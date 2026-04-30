import numpy as np
import cv2
import h5py
import tiffslide
from tqdm import tqdm
from pathlib import Path
import argparse

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
            "patch_coords": None,
            "slice_ids": None,
            "blur_scores": None,
            "num_slices": 0
        }

    def calculate_blur_score_val(self, patch_rgb):
        """ラプラシアンの分散を用いて鮮明さを計算する"""
        gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return score

    def calculate_blur_score_mean(self, patch_rgb):
        gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
        # near8相当の処理
        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
        edge = cv2.filter2D(gray, cv2.CV_32F, kernel=kernel)
        return np.mean(np.abs(edge))

    def run(self, out_dir):
        """前処理を実行し、逐次 HDF5 に保存する"""
        print(f"Processing slide: {self.slide_path.name}")
        out_path = Path(out_dir) / f"{self.slide_path.stem}.h5"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. サムネイルの取得とマスク作成
        level = min(2, len(self.slide.level_dimensions) - 1)
        thumbnail = np.array(self.slide.read_region((0, 0), level, self.slide.level_dimensions[level]).convert("RGB"))
        gray_thumb = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)
        
        # 大津法で組織部を抽出
        thresh, mask = cv2.threshold(gray_thumb, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 連結成分解析（スライスごとのID付け）
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        scale = self.slide.level_downsamples[level]
        slide_w, slide_h = self.slide.dimensions
        h_thumb, w_thumb = mask.shape
        
        step = self.patch_size
        # サムネイル上での「組織として認める最小面積」の計算
        threshold_area_thumb = (self.patch_size**2 * 500) / (scale**2)

        # HDF5の準備（逐次書き込み用に maxshape を設定）
        with h5py.File(out_path, "w") as f:
            dset_img = f.create_dataset("images", (0, step, step, 3), maxshape=(None, step, step, 3), 
                                        dtype='uint8', compression="gzip", chunks=(1, step, step, 3))
            dset_coords = f.create_dataset("coords", (0, 2), maxshape=(None, 2), dtype='int32')
            dset_slices = f.create_dataset("slice_ids", (0,), maxshape=(None,), dtype='int32')
            dset_blur_mean = f.create_dataset("blur_scores_mean", (0,), maxshape=(None,), dtype='float32')
            dset_blur_val = f.create_dataset("blur_scores_val", (0,), maxshape=(None,), dtype='float32')

            final_coords = []
            final_slice_ids = []
            final_blur_scores_mean = []
            final_blur_scores_val = []
            count = 0

            # グリッド走査
            for y0 in tqdm(range(0, slide_h - step, step), desc="Rows"):
                for x0 in range(0, slide_w - step, step):
                    
                    # サムネイル座標への換算
                    r_thumb, c_thumb = int(y0 / scale), int(x0 / scale)
                    step_thumb = int(step / scale)
                    
                    if r_thumb + step_thumb > h_thumb or c_thumb + step_thumb > w_thumb:
                        continue

                    # 組織含有率のチェック（マスク上で20%以上が組織であること）
                    mask_patch = mask[r_thumb : r_thumb + step_thumb, c_thumb : c_thumb + step_thumb]
                    if np.mean(mask_patch > 0) < 0.2:
                        continue
                    
                    # スライスIDの取得（パッチの中心点を使用）
                    slice_id = labels[r_thumb + step_thumb // 2, c_thumb + step_thumb // 2]
                    if slice_id == 0 or stats[slice_id, cv2.CC_STAT_AREA] < threshold_area_thumb:
                        continue

                    # パッチ読み込みとスコア計算
                    patch = np.array(self.slide.read_region((x0, y0), 0, (step, step)).convert("RGB"))
                    blur_score_mean = self.calculate_blur_score_mean(patch)
                    blur_score_val = self.calculate_blur_score_val(patch)

                    # HDF5にパッチを保存
                    dset_img.resize((count + 1, step, step, 3))
                    dset_img[count] = patch
                    
                    # スコアも保存
                    dset_blur_mean.resize((count + 1,))
                    dset_blur_mean[count] = blur_score_mean
                    dset_blur_val.resize((count + 1,))
                    dset_blur_val[count] = blur_score_val

                    final_coords.append([x0, y0])
                    final_slice_ids.append(slice_id - 1)
                    final_blur_scores_mean.append(blur_score_mean)
                    final_blur_scores_val.append(blur_score_val)
                    count += 1

            # 座標とIDを一括保存
            if count > 0:
                dset_coords.resize((count, 2))
                dset_coords[:] = np.array(final_coords)
                dset_slices.resize((count,))
                dset_slices[:] = np.array(final_slice_ids)

        # 実行結果を保持（可視化用）
        self.results.update({
            "thumbnail": thumbnail,
            "scale": scale,
            "patch_coords": np.array(final_coords),
            "slice_ids": np.array(final_slice_ids),
            "blur_scores_mean": np.array(final_blur_scores_mean),
            "blur_scores_val": np.array(final_blur_scores_val),
            "num_slices": num_labels - 1
        })
        return out_path

    def visualize(self, save_path=None):
        """抽出されたパッチの位置をサムネイル上にプロットする"""
        if self.results["thumbnail"] is None or self.results["patch_coords"] is None:
            print("Error: No results to visualize. Run the processor first.")
            return None

        vis_img = self.results["thumbnail"].copy()
        scale = self.results["scale"]
        
        # スライスごとに異なる色を生成
        np.random.seed(42) # 色を固定
        colors = np.random.randint(50, 255, (self.results["num_slices"] + 1, 3))
        circle_radius = 3

        for i, (x, y) in enumerate(self.results["patch_coords"]):
            x_center = x + self.patch_size // 2
            y_center = y + self.patch_size // 2

            # --- [修正ポイント2] サムネイル座標への換算 ---
            cx_thumb = int(x_center / scale)
            cy_thumb = int(y_center / scale)
            sid = self.results["slice_ids"][i]
            color = colors[sid + 1].tolist()
            
            # パッチの存在箇所に矩形を描画
            s = int(self.patch_size / scale)
            cv2.circle(vis_img, (cx_thumb, cy_thumb), 
                                radius=circle_radius, color=color, thickness=-1)

        if save_path:
            cv2.imwrite(str(save_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        
        return vis_img

# --- デバッグ・実行用コード ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSI Patch Extraction with Blur Scoring")
    parser.add_argument("slide_path", type=str, help="Path to the WSI file (e.g., .ndpi, .svs, .tif)")
    parser.add_argument("out_dir", type=str, help="Directory to save the results")
    parser.add_argument("--patch_size", type=int, default=256, help="Size of the patches")
    parser.add_argument("--no_vis", action="store_true", help="Disable visualization")
    
    args = parser.parse_args()

    # インスタンス化
    processor = WSIProcessor(args.slide_path, patch_size=args.patch_size)
    
    # 実行
    try:
        h5_path = processor.run(args.out_dir)
        print(f"Successfully saved {len(processor.results['patch_coords'])} patches to: {h5_path}")
        
        # 可視化の実行
        if not args.no_vis:
            vis_path = Path(args.out_dir) / f"{Path(args.slide_path).stem}_vis.png"
            processor.visualize(save_path=vis_path)
            print(f"Visualization saved to: {vis_path}")
            
    except Exception as e:
        print(f"An error occurred: {e}")