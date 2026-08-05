import cv2
import numpy as np


def remove_pen_marks(img_rgb: np.ndarray) -> np.ndarray:
    """HSV色空間を用いて、病理WSI上のペン跡（緑・青・極端な赤など）を検出してマスクを作成する。

    Args:
        img_rgb (np.ndarray): サムネイルRGB画像 (H, W, 3), uint8

    Returns:
        np.ndarray: ペン跡ではない領域を表すマスク (255=正常領域, 0=ペン跡)
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # 彩度が高すぎる領域の中で、H&E 染色色相範囲外（緑・青・濃黒インク等）をペン跡として検出
    # 緑・青系のマーカー色相 (およそ H: 35~150) かつ彩度 S > 80
    pen_mask_green_blue = ((h >= 35) & (h <= 150) & (s > 80)).astype(np.uint8) * 255

    # 極端に彩度が高くかつ明度が極端に低い黒・青インクなど
    pen_mask_dark = ((v < 40) & (s > 150)).astype(np.uint8) * 255

    pen_mask = cv2.bitwise_or(pen_mask_green_blue, pen_mask_dark)
    # モルフォロジー処理でノイズ除去
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pen_mask = cv2.morphologyEx(pen_mask, cv2.MORPH_DILATE, kernel)

    # 正常領域マスク（ペン跡以外が 255）
    valid_mask = cv2.bitwise_not(pen_mask)
    return valid_mask


def segment_tissue_otsu(
    thumbnail_rgb: np.ndarray,
    min_area_px: float = 0.0,
    remove_pen: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """大津法 (Otsu thresholding) を用いてサムネイル画像から組織領域を抽出する。

    Args:
        thumbnail_rgb (np.ndarray): サムネイルRGB画像 (H, W, 3), uint8
        min_area_px (float): 組織として見なす最小面積（ピクセル単位）。これ未満の領域は除外。
        remove_pen (bool): マーカーペン跡の除去処理を適用するかどうか

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, int]:
            - mask (np.ndarray): 組織領域マスク (H, W), uint8 (255=組織, 0=背景)
            - labels (np.ndarray): 連結成分ラベル画像 (H, W), int32
            - stats (np.ndarray): 連結成分の統計情報
            - num_labels (int): ラベル数
    """
    gray = cv2.cvtColor(thumbnail_rgb, cv2.COLOR_RGB2GRAY)
    thresh, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if remove_pen:
        valid_mask = remove_pen_marks(thumbnail_rgb)
        mask = cv2.bitwise_and(mask, valid_mask)

    # モルフォロジー処理による微小ノイズの除去と穴埋め
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    # 面積が min_area_px 未満の成分を除外する
    if min_area_px > 0:
        filtered_mask = np.zeros_like(mask)
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area >= min_area_px:
                filtered_mask[labels == label_id] = 255
        mask = filtered_mask
        # 再度連結成分を計算
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    return mask, labels, stats, num_labels
