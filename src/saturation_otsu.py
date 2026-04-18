import cv2
import numpy as np
from openslide import OpenSlide

def get_slice_idx(image, patch_size, threshold=None, slice_min_patch = 500, 
    adjacent_cells = [(-1, 0), (1, 0), (0, -1), (0, 1)]):
    """
    get_patchesに加え, WSI内に複数ある切片を分けて認識するようにする。

    Parameters
    ----------
    iamge_file: openslide.OpenSlide
        読み込むWSIのtifファイル
    patch_size: int
        1つのpatchの大きさ(pixel単位)
    threshold: float or None
        各patchが背景かどうかを判定する彩度のthreshold.
        Noneの場合, OTSU法で決定する。
    slice_min_patch: int
        patch数がslice_min_patch未満の切片は小さすぎるとして除去する。
    adjacent_cells: List[Tuple[int]]
        隣接すると判定する領域の指定。
        デフォルトでは上下左右に隣接するpatchは同じ切片に属すると判定する。

    Returns
    -------
    slice_idx: np.array(int)[wsi_height, wsi_width]
        各patchがどの切片に属するか(-1=背景)
    n_slice: int
        切片の総数
    """
    level = image.get_best_level_for_downsample(patch_size)
    downsample = image.level_downsamples[level]
    ratio = patch_size / downsample
    whole = image.read_region(location=(0,0), level=level,
        size = image.level_dimensions[level]).convert('HSV')
    whole = whole.resize((int(whole.width / ratio), int(whole.height / ratio)))
    whole = np.array(whole, dtype=np.uint8)
    saturation = whole[:,:,1]

    if threshold is None:
        threshold, _ = cv2.threshold(saturation, 0, 255, cv2.THRESH_OTSU)
    mask = saturation > threshold
    left_mask = np.full((mask.shape[0]+1, mask.shape[1]+1), fill_value=False, dtype=bool)
    left_mask[:-1, :-1] = mask
    n_left_mask = np.sum(left_mask)
    slice_idx = np.full_like(mask, fill_value=-1, dtype=int)
    i_slice = 0
    while n_left_mask > 0:
        mask_i = np.where(left_mask)
        slice_left_indices = [(mask_i[0][0], mask_i[1][0])]
        while len(slice_left_indices) > 0:
            i, j = slice_left_indices.pop()
            if left_mask[i, j]:
                slice_idx[i, j] = i_slice
                for adj_i, adj_j in adjacent_cells:
                    if left_mask[i+adj_i, j+adj_j]:
                        slice_left_indices.append((i+adj_i, j+adj_j))
                left_mask[i, j] = False
                n_left_mask -= 1
        slice_mask = slice_idx == i_slice
        if np.sum(slice_mask) < slice_min_patch:
            slice_idx[slice_mask] = -1
        else:
            i_slice += 1
    n_slice = i_slice
    return slice_idx, n_slice


def get_threshold(image_files, patch_size):
    """
    Decides threshold of saturation between object and background based on OTSU method.
    Multiple images are used.

    Parameters
    ----------
    image_files: List[Str]
        File paths of tif image to be processed.
    patch_size: int
        Size of each patch.

    Returns
    -------
    threshold: int

    """
    wsi_saturaions = []
    for image_file in image_files:
        image = OpenSlide(image_file)
        level = image.get_best_level_for_downsample(patch_size)
        downsample = image.level_downsamples[level]
        ratio = patch_size / downsample
        whole = image.read_region(location=(0,0), level=level,
            size=image.level_dimensions[level]).convert('HSV')
        whole = whole.resize((int(whole.width / ratio), int(whole.height / ratio)))
        whole = np.array(whole, dtype=np.uint8)
        wsi_saturaions.append(whole[:,:,1].ravel())
    wsi_saturaions = np.concatenate(wsi_saturaions)
    threshold, _ = cv2.threshold(wsi_saturaions, 0, 255, cv2.THRESH_OTSU)
    return threshold