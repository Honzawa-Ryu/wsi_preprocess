import cv2
import numpy as np


class ReinhardNormalizer:
    """Reinhard 法 (L*a*b* 色空間の平均および標準偏差を合わせる手法) による染色正規化クラス。"""

    def __init__(
        self,
        target_means: tuple[float, float, float] | None = None,
        target_stds: tuple[float, float, float] | None = None,
    ):
        # デフォルトは一般的なH&E病理スライド組織パッチの経験的 L*a*b* 平均・標準偏差
        self.target_means = np.array(
            target_means if target_means is not None else [145.0, 135.0, 133.0],
            dtype=np.float32,
        )
        self.target_stds = np.array(
            target_stds if target_stds is not None else [38.0, 10.0, 7.0],
            dtype=np.float32,
        )

    def fit(self, reference_rgb: np.ndarray) -> "ReinhardNormalizer":
        """参照RGB画像からターゲット統計量を算出・保持する。"""
        lab = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        means = np.mean(lab, axis=(0, 1))
        stds = np.std(lab, axis=(0, 1))
        # 0割防止
        stds = np.maximum(stds, 1e-5)
        self.target_means = means
        self.target_stds = stds
        return self

    def transform(self, img_rgb: np.ndarray) -> np.ndarray:
        """RGB画像に対して Reinhard 色正規化を適用する。"""
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        means = np.mean(lab, axis=(0, 1))
        stds = np.std(lab, axis=(0, 1))
        stds = np.maximum(stds, 1e-5)

        # 正規化: (X - mean) * (target_std / std) + target_mean
        lab_norm = (lab - means) * (self.target_stds / stds) + self.target_means
        lab_norm = np.clip(lab_norm, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab_norm, cv2.COLOR_LAB2RGB)


class MacenkoNormalizer:
    """Macenko 法 (光学濃度 Optical Density 空間の SVD による染色分離・正規化) クラス。"""

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.15,
        Io: float = 240.0,
        target_stain_matrix: np.ndarray | None = None,
        target_max_c: np.ndarray | None = None,
    ):
        self.alpha = alpha
        self.beta = beta
        self.Io = Io
        # デフォルトは標準的なH&E染色マトリクスと最大濃度
        self.target_stain_matrix = (
            target_stain_matrix
            if target_stain_matrix is not None
            else np.array(
                [[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]],
                dtype=np.float32,
            )
        )
        self.target_max_c = (
            target_max_c
            if target_max_c is not None
            else np.array([1.9705, 1.0308], dtype=np.float32)
        )

    def _convert_rgb_to_od(self, img_rgb: np.ndarray) -> np.ndarray:
        img_rgb = img_rgb.astype(np.float32)
        img_rgb = np.maximum(img_rgb, 1.0)
        return -np.log10(img_rgb / self.Io)

    def transform(self, img_rgb: np.ndarray) -> np.ndarray:
        """RGBパッチに対してMacenko染色正規化を適用する。"""
        h, w, _ = img_rgb.shape
        od = self._convert_rgb_to_od(img_rgb)
        od_flat = od.reshape((-1, 3))

        # 背景・低光学濃度のピクセルを除外してベクトルの主成分を計算
        od_hat = od_flat[np.all(od_flat > self.beta, axis=1)]
        if len(od_hat) < 100:
            # 組織ピクセルが極端に少ない場合は元画像をそのまま返す安全処理
            return img_rgb

        try:
            _, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
            eigvecs = eigvecs[:, [1, 2]]
            if eigvecs[0, 0] < 0:
                eigvecs[:, 0] *= -1
            if eigvecs[0, 1] < 0:
                eigvecs[:, 1] *= -1

            that = np.dot(od_hat, eigvecs)
            phi = np.arctan2(that[:, 1], that[:, 0])
            min_phi = np.percentile(phi, self.alpha)
            max_phi = np.percentile(phi, 100 - self.alpha)

            v1 = np.dot(eigvecs, np.array([np.cos(min_phi), np.sin(min_phi)]))
            v2 = np.dot(eigvecs, np.array([np.cos(max_phi), np.sin(max_phi)]))

            if v1[0] > v2[0]:
                HE = np.array([v1, v2]).T
            else:
                HE = np.array([v2, v1]).T

            HE = HE / np.linalg.norm(HE, axis=0, keepdims=True)

            # 濃度Cの計算と正規化
            Y = od_flat.T
            C = np.linalg.lstsq(HE, Y, rcond=None)[0]
            max_c = np.percentile(C, 99, axis=1)
            max_c = np.maximum(max_c, 1e-5)

            C_norm = C * (self.target_max_c[:, np.newaxis] / max_c[:, np.newaxis])
            OD_norm = np.dot(self.target_stain_matrix, C_norm)

            img_norm = self.Io * np.exp(-OD_norm * np.log(10))
            img_norm = np.clip(img_norm.T.reshape((h, w, 3)), 0, 255).astype(np.uint8)
            return img_norm
        except Exception:
            # 万が一特異値分解等で不安定になった場合のフォールバック
            return img_rgb


class ColorAdjuster:
    """明度・コントラスト・ガンマ補正を行うクラス。"""

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 1.0,
        gamma: float = 1.0,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.gamma = gamma

    def transform(self, img_rgb: np.ndarray) -> np.ndarray:
        img = img_rgb.astype(np.float32)
        img = img * self.contrast + self.brightness

        if abs(self.gamma - 1.0) > 1e-3:
            img = np.clip(img, 0, 255) / 255.0
            img = (img ** (1.0 / self.gamma)) * 255.0

        return np.clip(img, 0, 255).astype(np.uint8)


def get_color_normalizer(method: str | None, **kwargs):
    """指定文字列から対応する正規化クラスのインスタンスを取得するファクトリ関数。"""
    if not method or method.lower() in ("none", "raw", ""):
        return None
    method = method.lower()
    if method == "reinhard":
        return ReinhardNormalizer(**kwargs)
    elif method == "macenko":
        return MacenkoNormalizer(**kwargs)
    elif method == "adjust":
        return ColorAdjuster(**kwargs)
    else:
        raise ValueError(f"Unknown color normalization method: {method}")
