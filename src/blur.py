"""パッチのぼやけ具合を定量する指標をまとめたモジュール。

生WSIからの切り出し(wsi_processer)とTRIDENT出力への追記(add_blur_scores)の
双方から呼ばれるため、指標の実装はここに一本化する。
"""
import cv2
import numpy as np

# near8相当のラプラシアンカーネル
_NEAR8_KERNEL = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)


def blur_score_val(patch_rgb):
    """ラプラシアンの分散を用いて鮮明さを計算する。値が大きいほど鮮明。"""
    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def blur_score_mean(patch_rgb):
    """near8フィルタ応答の絶対値平均。値が大きいほど鮮明。"""
    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    edge = cv2.filter2D(gray, cv2.CV_32F, kernel=_NEAR8_KERNEL)
    return float(np.mean(np.abs(edge)))


# HDF5に書き出す際のデータセット名 -> 計算関数
METRICS = {
    "blur_scores_val": blur_score_val,
    "blur_scores_mean": blur_score_mean,
}


def compute_all(patch_rgb):
    """全指標をまとめて計算し、{データセット名: スコア} を返す。"""
    return {name: fn(patch_rgb) for name, fn in METRICS.items()}
