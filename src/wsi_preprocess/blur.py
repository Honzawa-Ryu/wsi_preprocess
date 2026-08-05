import cv2
import numpy as np


def calculate_blur_score_val(patch_rgb: np.ndarray) -> float:
    """ラプラシアンフィルタの分散を用いてパッチ画像の鮮明さ（ボケ度合い）を計算する。

    スコアが高いほど鮮明（エッジが豊富）、低いほどボケていることを示す。

    Args:
        patch_rgb (np.ndarray): RGB画像 (H, W, 3), uint8

    Returns:
        float: ラプラシアンの分散値
    """
    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(score)


def calculate_blur_score_mean(patch_rgb: np.ndarray) -> float:
    """エッジフィルタの平均絶対値を用いてパッチ画像の鮮明さを計算する。

    Args:
        patch_rgb (np.ndarray): RGB画像 (H, W, 3), uint8

    Returns:
        float: 8近傍ラプラシアン相当エッジ応答の絶対値平均
    """
    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    edge = cv2.filter2D(gray, cv2.CV_32F, kernel=kernel)
    return float(np.mean(np.abs(edge)))


def is_patch_blurry(
    patch_rgb: np.ndarray,
    min_blur_score_val: float | None = None,
    min_blur_score_mean: float | None = None,
) -> bool:
    """パッチが指定されたボケしきい値を下回る（ボケている）かどうかを判定する。

    Args:
        patch_rgb (np.ndarray): RGB画像
        min_blur_score_val (float | None): Laplacian 分散の最小しきい値。下回る場合はボケと判定
        min_blur_score_mean (float | None): エッジ平均の最小しきい値。下回る場合はボケと判定

    Returns:
        bool: ボケている場合 True, 鮮明な場合 False
    """
    if min_blur_score_val is not None:
        score_val = calculate_blur_score_val(patch_rgb)
        if score_val < min_blur_score_val:
            return True

    if min_blur_score_mean is not None:
        score_mean = calculate_blur_score_mean(patch_rgb)
        if score_mean < min_blur_score_mean:
            return True

    return False
