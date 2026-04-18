import sys, os
import numpy as np
import cv2
from openslide import OpenSlide
from tqdm import tqdm
kernels = {
    'near4': np.array([[0,1,0], [1,-4,1], [0,1,0]]),
    'near8': np.array([[1,1,1], [1,-8,1], [1,1,1]])
}

def get_blur(image: OpenSlide, patch_size: int, kernel: str='near4', show_tqdm=False):
    """
    画像内のぼやけている領域を判定する関数。
    画像内のエッジを認識するラプラシアンフィルタをかけ, patchごとの絶対値の平均値で
    計算している。

    Parameterrs
    -----------
    image: openslide.OpenSlide
        WSIのtifファイル
    patch_size: int
        patchの1辺の長さ
    kernel: Optional[str]
        ラプラシアンフィルタでは,各ピクセルの周囲との差を計算するが, カーネルには 
           "near4"       "near8"
        [[ 0, 1, 0],  [[ 1, 1, 1],
         [ 1,-4, 1],   [ 1,-8, 1],
         [ 0, 1, 0]]  [[ 1, 1, 1]]
        の2つが使われているので, "near4"または"near8"のどちらかを指定する。
        TG-GATEs病理画像ではどちらもあまり変わらなかったです。
    show_tqdm: bool
        Trueの場合, 1列処理ごとに進むプログレスバーを表示する。
    
    Returns
    -------
    laplacian: np.array(int)[wsi_height//patch_size, wsi_width//patch_size]
        各patchのラプラシアンフィルタの平均値。
        左上からpatchごとに処理していくので, 右端・下端は無視される可能性があります。
    """

    psizex = image.dimensions[0]//patch_size
    psizey = image.dimensions[1]//patch_size
    if kernel not in kernels:
        raise ValueError(f"Kernel {kernel} is not supported in get_blur().")
    kernel = kernels[kernel]
    filters = np.zeros((psizey, psizex))

    if show_tqdm:
        from tqdm import tqdm
        xs = tqdm(range(psizex))
    else:
        xs = range(psizex)

    for px in xs:
        for py in range(psizey):
            patch = image.read_region((px*patch_size, py*patch_size), level=0,
                size=(patch_size, patch_size))
            patch = cv2.cvtColor(np.array(patch), cv2.COLOR_BGR2GRAY)
            edge = cv2.filter2D(patch, cv2.CV_32F, kernel=kernel)
            filters[py, px] = np.mean(np.abs(edge))
    return filters