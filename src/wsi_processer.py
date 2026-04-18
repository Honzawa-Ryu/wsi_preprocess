import numpy as np
import cv2
import h5py
import tiffslide
from tqdm import tqdm
from pathlib import Path

class WSIProcessor:
    def __init__(self, slide_path, patch_size=256):
        print(f"Initializing WSIProcessor with slide: {slide_path}")
        self.slide_path = Path(slide_path)
        self.slide = tiffslide.TiffSlide(str(self.slide_path))
        self.patch_size = patch_size
        
        # 実行結果を保持する属性
        self.results = {
            "thumbnail": None,
            "scale": None,
            "global_threshold": None,
            "patch_coords": None,    # 保存されたパッチの座標 (N, 2)
            "slice_ids": None,       # 各パッチのスライスID (N,)
            "num_slices": 0
        }

    def run(self, out_dir):
        """前処理をグリッドベースで実行し、高速に H5 保存する"""
        print(f"Processing slide: {self.slide_path.name}")
        
        # 1. サムネイルの取得とマスク作成
        level = min(2, len(self.slide.level_dimensions) - 1)
        thumbnail = np.array(self.slide.read_region((0, 0), level, self.slide.level_dimensions[level]).convert("RGB"))
        gray_thumb = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)
        
        # 大津法による二値化
        thresh, _ = cv2.threshold(gray_thumb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = (gray_thumb < thresh).astype(np.uint8) * 255
        
        # 連結成分解析（スライスID用）
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        
        scale = self.slide.level_downsamples[level]
        
        # --- 修正ポイント：グリッド座標の計算 ---
        # サムネイル上でのパッチサイズ（歩幅）を計算
        thumb_step = int(self.patch_size / scale)
        h, w = mask.shape
        
        final_patches = []
        final_coords = []
        final_slice_ids = []

        # グリッド状に座標を生成（パッチサイズずつ飛ばしてループ）
        # c, r はサムネイル上の座標
        for r in tqdm(range(0, h - thumb_step, thumb_step), desc="Rows"):
            for c in range(0, w - thumb_step, thumb_step):
                
                # パッチ範囲のマスクを確認（ここが「事前チェック」）
                mask_patch = mask[r:r+thumb_step, c:c+thumb_step]
                if np.mean(mask_patch > 0) < 0.2: # 20%以上が組織でなければスキップ
                    continue
                
                # スライスIDの取得（中心点などのラベルを採用）
                slice_id = labels[r + thumb_step//2, c + thumb_step//2]
                if slice_id == 0 or stats[slice_id, cv2.CC_STAT_AREA] * (scale**2) < 10000: 
                    continue

                # レベル0での座標に変換して読み込み
                x, y = int(c * scale), int(r * scale)
                patch = np.array(self.slide.read_region((x, y), 0, (self.patch_size, self.patch_size)).convert("RGB"))
                
                # リストに一時保存（H5への頻繁なアクセスを避ける）
                final_patches.append(patch)
                final_coords.append([x, y])
                final_slice_ids.append(slice_id - 1)

        # 2. まとめて HDF5 に保存
        out_path = Path(out_dir) / f"{self.slide_path.stem}.h5"
        with h5py.File(out_path, "w") as f:
            if final_patches:
                # 一気に配列に変換して書き込む（これが最速）
                f.create_dataset("images", data=np.array(final_patches), 
                                 dtype='uint8', compression="gzip", chunks=True)
                f.create_dataset("coords", data=np.array(final_coords), dtype='int32')
                f.create_dataset("slice_ids", data=np.array(final_slice_ids), dtype='int32')

        # 実行結果をキャッシュ
        self.results.update({
            "thumbnail": thumbnail,
            "scale": scale,
            "global_threshold": thresh,
            "patch_coords": np.array(final_coords),
            "slice_ids": np.array(final_slice_ids),
            "num_slices": num_labels - 1
        })
        return out_path

    def visualize(self, save_path=None):
        """self.results に保存されたデータを使って可視化する"""
        if self.results["thumbnail"] is None:
            print("Error: Run the processor first.")
            return

        vis_img = self.results["thumbnail"].copy()
        scale = self.results["scale"]
        
        # スライスごとに色を変えてパッチ位置をプロット
        # colors[0]は背景、1以降が各スライス
        colors = np.random.randint(0, 255, (self.results["num_slices"] + 1, 3))
        
        for i, (x, y) in enumerate(self.results["patch_coords"]):
            cx, cy = int(x / scale), int(y / scale)
            sid = self.results["slice_ids"][i]
            color = colors[sid + 1].tolist()
            cv2.circle(vis_img, (cx, cy), 2, color, -1)

        if save_path:
            cv2.imwrite(str(save_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        
        return vis_img
    
if __name__ == "__main__":
    import argparse
    print("Starting WSI Patch Extraction...")
    parser = argparse.ArgumentParser(description="WSI Patch Extraction")
    parser.add_argument("slide_path", type=str, help="Path to the WSI file")
    parser.add_argument("out_dir", type=str, help="Directory to save the output H5 file")
    parser.add_argument("--visualize", action="store_true", help="Whether to save visualization image")
    args = parser.parse_args()

    processor = WSIProcessor(args.slide_path)
    h5_path = processor.run(args.out_dir)
    print(f"Saved patches to: {h5_path}")

    if args.visualize:
        vis_img = processor.visualize(save_path=Path(args.out_dir) / f"{Path(args.slide_path).stem}_vis.png")
        print(f"Saved visualization to: {Path(args.out_dir) / f'{Path(args.slide_path).stem}_vis.png'}")