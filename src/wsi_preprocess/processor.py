from pathlib import Path
from typing import Generator, Any

import cv2
import h5py
import numpy as np
import tiffslide
from tqdm import tqdm

from wsi_preprocess.blur import (
    calculate_blur_score_mean,
    calculate_blur_score_val,
    is_patch_blurry,
)
from wsi_preprocess.normalize import get_color_normalizer
from wsi_preprocess.segment import segment_tissue_otsu


class WSIProcessor:
    """WSI の前処理（組織検出・ボケ評価・色味正規化・パッチ抽出および HDF5 出力）を行うクラス。"""

    def __init__(
        self,
        slide_path: str | Path,
        patch_size: int = 256,
        tissue_threshold: float = 0.2,
        min_tissue_area_thumb: float = 500.0,
        remove_pen: bool = False,
        min_blur_score_val: float | None = None,
        min_blur_score_mean: float | None = None,
        color_normalizer: Any | str | None = None,
    ):
        """
        Args:
            slide_path: WSI 画像ファイルへのパス
            patch_size: パッチの1辺のサイズ（ピクセル）
            tissue_threshold: パッチが組織領域と見なされるための組織含有率（0.0 ~ 1.0）
            min_tissue_area_thumb: サムネイル上での最小組織面積しきい値
            remove_pen: マーカーペン跡の除去マスクを適用するかどうか
            min_blur_score_val: ボケ除外の分散しきい値（下回る場合はパッチをスキップ）
            min_blur_score_mean: ボケ除外の平均絶対しきい値
            color_normalizer: 色味正規化インスタンス、または正規化手法名 ('reinhard', 'macenko', None)
        """
        self.slide_path = Path(slide_path)
        self.slide = tiffslide.TiffSlide(str(self.slide_path))
        self.patch_size = patch_size
        self.tissue_threshold = tissue_threshold
        self.min_tissue_area_thumb = min_tissue_area_thumb
        self.remove_pen = remove_pen
        self.min_blur_score_val = min_blur_score_val
        self.min_blur_score_mean = min_blur_score_mean

        if isinstance(color_normalizer, str):
            self.color_normalizer = get_color_normalizer(color_normalizer)
        else:
            self.color_normalizer = color_normalizer

        self.results: dict[str, Any] = {
            "thumbnail": None,
            "scale": None,
            "patch_coords": None,
            "slice_ids": None,
            "blur_scores_mean": None,
            "blur_scores_val": None,
            "num_slices": 0,
        }

    def prepare_tissue_mask(self):
        """サムネイル画像および組織セグメンテーションマスクを取得する。"""
        level = min(2, len(self.slide.level_dimensions) - 1)
        thumb_img = self.slide.read_region(
            (0, 0), level, self.slide.level_dimensions[level]
        ).convert("RGB")
        thumbnail = np.array(thumb_img)
        scale = self.slide.level_downsamples[level]

        # サムネイル上での面積しきい値をスケーリングに応じて計算
        threshold_area_thumb = (self.patch_size**2 * self.min_tissue_area_thumb) / (
            scale**2
        )

        mask, labels, stats, num_labels = segment_tissue_otsu(
            thumbnail,
            min_area_px=threshold_area_thumb,
            remove_pen=self.remove_pen,
        )

        self.results["thumbnail"] = thumbnail
        self.results["scale"] = scale
        self.results["num_slices"] = max(0, num_labels - 1)

        return thumbnail, mask, labels, stats, scale

    def iter_patches(self, progress: bool = False) -> Generator[dict[str, Any], None, None]:
        """パッチをオンデマンドで生成して辞書形式で返すジェネレータ関数。"""
        thumbnail, mask, labels, stats, scale = self.prepare_tissue_mask()
        slide_w, slide_h = self.slide.dimensions
        h_thumb, w_thumb = mask.shape
        step = self.patch_size
        step_thumb = int(step / scale)

        threshold_area_thumb = (self.patch_size**2 * self.min_tissue_area_thumb) / (
            scale**2
        )

        row_range = range(0, slide_h - step, step)
        if progress:
            row_range = tqdm(row_range, desc=f"Extracting [{self.slide_path.name}]")

        for y0 in row_range:
            for x0 in range(0, slide_w - step, step):
                r_thumb, c_thumb = int(y0 / scale), int(x0 / scale)

                if r_thumb + step_thumb > h_thumb or c_thumb + step_thumb > w_thumb:
                    continue

                mask_patch = mask[
                    r_thumb : r_thumb + step_thumb, c_thumb : c_thumb + step_thumb
                ]
                if np.mean(mask_patch > 0) < self.tissue_threshold:
                    continue

                slice_id = labels[r_thumb + step_thumb // 2, c_thumb + step_thumb // 2]
                if (
                    slice_id == 0
                    or stats[slice_id, cv2.CC_STAT_AREA] < threshold_area_thumb
                ):
                    continue

                patch = np.array(
                    self.slide.read_region((x0, y0), 0, (step, step)).convert("RGB")
                )

                # ボケ判定（しきい値設定時のみフィルタ）
                if is_patch_blurry(
                    patch, self.min_blur_score_val, self.min_blur_score_mean
                ):
                    continue

                blur_score_mean = calculate_blur_score_mean(patch)
                blur_score_val = calculate_blur_score_val(patch)

                # 色味の正規化・補正
                if self.color_normalizer is not None:
                    patch = self.color_normalizer.transform(patch)

                yield {
                    "patch": patch,
                    "coords": [x0, y0],
                    "slice_id": int(slice_id - 1),
                    "blur_score_mean": blur_score_mean,
                    "blur_score_val": blur_score_val,
                }

    def run(
        self,
        out_dir: str | Path,
        progress: bool = True,
        save_images: bool = True,
    ) -> Path:
        """前処理を実行して結果を HDF5 に保存する（MahmoodLab TRIDENT / CLAM 互換仕様）。

        Args:
            out_dir: 保存先のディレクトリ
            progress: 進捗バーの表示可否
            save_images: 画像パッチ（RGB）を保存するかどうか。
                         False の時は座標データと属性のみ保存（TRIDENT/CLAM互換の軽量 H5 フォーマット）

        Returns:
            Path: 作成された HDF5 ファイルパス
        """
        out_path = Path(out_dir) / f"{self.slide_path.stem}.h5"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        step = self.patch_size

        with h5py.File(out_path, "w") as f:
            dset_img = None
            if save_images:
                dset_img = f.create_dataset(
                    "images",
                    (0, step, step, 3),
                    maxshape=(None, step, step, 3),
                    dtype="uint8",
                    compression="gzip",
                    chunks=(1, step, step, 3),
                )
            dset_coords = f.create_dataset(
                "coords", (0, 2), maxshape=(None, 2), dtype="int32"
            )
            dset_slices = f.create_dataset(
                "slice_ids", (0,), maxshape=(None,), dtype="int32"
            )
            dset_blur_mean = f.create_dataset(
                "blur_scores_mean", (0,), maxshape=(None,), dtype="float32"
            )
            dset_blur_val = f.create_dataset(
                "blur_scores_val", (0,), maxshape=(None,), dtype="float32"
            )

            # MahmoodLab CLAM / TRIDENT 互換ヘッダ属性の格納
            dset_coords.attrs["patch_size"] = int(self.patch_size)
            dset_coords.attrs["patch_level"] = 0
            dset_coords.attrs["downsample"] = 1.0

            f.attrs["patch_size"] = int(self.patch_size)
            f.attrs["patch_level"] = 0
            f.attrs["downsample"] = 1.0
            f.attrs["slide_path"] = str(self.slide_path)
            f.attrs["slide_name"] = str(self.slide_path.name)

            final_coords = []
            final_slice_ids = []
            final_blur_scores_mean = []
            final_blur_scores_val = []
            count = 0

            for item in self.iter_patches(progress=progress):
                if dset_img is not None:
                    dset_img.resize((count + 1, step, step, 3))
                    dset_img[count] = item["patch"]

                dset_blur_mean.resize((count + 1,))
                dset_blur_mean[count] = item["blur_score_mean"]
                dset_blur_val.resize((count + 1,))
                dset_blur_val[count] = item["blur_score_val"]

                final_coords.append(item["coords"])
                final_slice_ids.append(item["slice_id"])
                final_blur_scores_mean.append(item["blur_score_mean"])
                final_blur_scores_val.append(item["blur_score_val"])
                count += 1

            if count > 0:
                dset_coords.resize((count, 2))
                dset_coords[:] = np.array(final_coords)
                dset_slices.resize((count,))
                dset_slices[:] = np.array(final_slice_ids)

        self.results.update(
            {
                "patch_coords": np.array(final_coords) if final_coords else np.zeros((0, 2)),
                "slice_ids": np.array(final_slice_ids) if final_slice_ids else np.zeros(0),
                "blur_scores_mean": np.array(final_blur_scores_mean) if final_blur_scores_mean else np.zeros(0),
                "blur_scores_val": np.array(final_blur_scores_val) if final_blur_scores_val else np.zeros(0),
            }
        )

        return out_path

    def visualize(self, save_path: str | Path | None = None) -> np.ndarray | None:
        """抽出されたパッチ位置をサムネイル上にプロットして可視化する。"""
        if self.results["thumbnail"] is None or self.results["patch_coords"] is None:
            print("Error: No results to visualize. Run `run()` or `prepare_tissue_mask()` first.")
            return None

        vis_img = self.results["thumbnail"].copy()
        scale = self.results["scale"]

        np.random.seed(42)
        colors = np.random.randint(50, 255, (max(1, self.results["num_slices"] + 2), 3))
        circle_radius = 3

        coords = self.results["patch_coords"]
        slice_ids = self.results["slice_ids"]

        for i, (x, y) in enumerate(coords):
            x_center = x + self.patch_size // 2
            y_center = y + self.patch_size // 2
            cx_thumb = int(x_center / scale)
            cy_thumb = int(y_center / scale)

            sid = slice_ids[i] if i < len(slice_ids) else 0
            color = colors[(sid + 1) % len(colors)].tolist()

            cv2.circle(
                vis_img,
                (cx_thumb, cy_thumb),
                radius=circle_radius,
                color=color,
                thickness=-1,
            )

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

        return vis_img
