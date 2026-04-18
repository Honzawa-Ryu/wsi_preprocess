import numpy as np
import cv2
import h5py
import cupy as cp  # GPU加速用のNumPy代替
import cucim
from tqdm import tqdm
from pathlib import Path

class WSIProcessorCUcim:
    def __init__(self, slide_path, patch_size=256):
        self.slide_path = Path(slide_path)
        # CuImageを使用してスライドをロード
        self.slide = cucim.CuImage(str(self.slide_path))
        self.patch_size = patch_size
        
        # 実行結果を保持
        self.results = {
            "thumbnail": None,
            "scale": None,
            "global_threshold": None,
            "patch_coords": None,
            "slice_ids": None,
            "num_slices": 0
        }

    def run(self, out_dir, batch_size=128):
        """
        グリッドベースのバッチ処理により、GPUの性能を最大化してパッチを抽出する
        """
        print(f"Processing slide with CUcim (Optimized): {self.slide_path.name}")
        
        # 1. サムネイル取得とマスク作成
        level = min(2, self.slide.resolutions['level_count'] - 1)
        dims = self.slide.resolutions['level_dimensions'][level]
        
        # サムネイルをGPUで読み込み
        thumb_cupy = self.slide.read_region((0, 0), dims, level)
        thumbnail = cp.asnumpy(thumb_cupy)
        
        # 閾値判定 (CPUで十分高速)
        gray_thumb = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)
        thresh, _ = cv2.threshold(gray_thumb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = (gray_thumb < thresh).astype(np.uint8) * 255
        
        # 連結成分解析
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        scale = self.slide.resolutions['level_downsamples'][level]
        
        # --- 2. 処理対象の座標を「グリッド状」にリストアップ ---
        thumb_step = int(self.patch_size / scale)
        h, w = mask.shape
        candidate_coords = []

        for r in range(0, h - thumb_step, thumb_step):
            for c in range(0, w - thumb_step, thumb_step):
                # パッチ範囲のマスクを確認
                if np.mean(mask[r:r+thumb_step, c:c+thumb_step] > 0) > 0.5:
                    slice_id = labels[r + thumb_step // 2, c + thumb_step // 2]
                    # 小さすぎる領域はスキップ
                    if slice_id > 0 and stats[slice_id, cv2.CC_STAT_AREA] > 100:
                        x, y = int(c * scale), int(r * scale)
                        candidate_coords.append((x, y, slice_id - 1))

        print(f"Total candidate patches: {len(candidate_coords)}")

        # --- 3. バッチ処理による抽出と保存 ---
        final_coords = []
        final_slice_ids = []
        out_path = Path(out_dir) / f"{self.slide_path.stem}.h5"
        
        with h5py.File(out_path, "w") as f:
            # chunkサイズを指定して圧縮効率と速度を両立
            img_db = f.create_dataset("images", (0, self.patch_size, self.patch_size, 3), 
                                     maxshape=(None, self.patch_size, self.patch_size, 3), 
                                     dtype='uint8', compression="gzip", 
                                     chunks=(batch_size, self.patch_size, self.patch_size, 3))
            
            idx = 0
            # tqdmでバッチ処理の進捗を表示
            for i in tqdm(range(0, len(candidate_coords), batch_size), desc="Batch processing"):
                batch = candidate_coords[i : i + batch_size]
                
                batch_patches = []
                batch_meta = []

                for x, y, sid in batch:
                    try:
                        # GPUで読み込み
                        patch_cupy = self.slide.read_region((x, y), (self.patch_size, self.patch_size), 0)
                        
                        # GPU上で二段階目チェック (HSVのSやGrayなど、必要に応じて)
                        # ここでは簡易的にGrayで判定
                        patch_gray = 0.299 * patch_cupy[:,:,0] + 0.587 * patch_cupy[:,:,1] + 0.114 * patch_cupy[:,:,2]
                        if cp.mean(patch_gray < thresh) > 0.5:
                            # 合格したものだけCPUに戻す準備
                            batch_patches.append(cp.asnumpy(patch_cupy))
                            batch_meta.append(((x, y), sid))
                    except:
                        continue

                # バッチ分をまとめてH5に書き込み
                if batch_patches:
                    num_valid = len(batch_patches)
                    img_db.resize((idx + num_valid, self.patch_size, self.patch_size, 3))
                    img_db[idx : idx + num_valid] = np.array(batch_patches)
                    
                    for (x, y), sid in batch_meta:
                        final_coords.append([x, y])
                        final_slice_ids.append(sid)
                    
                    idx += num_valid

            # 座標情報を記録
            f.create_dataset("coords", data=np.array(final_coords), dtype='int32')
            f.create_dataset("slice_ids", data=np.array(final_slice_ids), dtype='int32')

        # 4. 結果を保持
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
        """可視化ロジック (既存コードを継承)"""
        if self.results["thumbnail"] is None:
            print("Error: Run the processor first.")
            return

        vis_img = self.results["thumbnail"].copy()
        scale = self.results["scale"]
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
    parser = argparse.ArgumentParser(description="WSI Patch Extraction with CUcim")
    parser.add_argument("slide_path", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    processor = WSIProcessorCUcim(args.slide_path)
    h5_path = processor.run(args.out_dir)
    print(f"Saved patches to: {h5_path}")

    if args.visualize:
        vis_path = Path(args.out_dir) / f"{Path(args.slide_path).stem}_vis.png"
        processor.visualize(save_path=vis_path)
        print(f"Saved visualization to: {vis_path}")