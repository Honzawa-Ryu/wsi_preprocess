import sys, os
import numpy as np
import pandas as pd # CSV操作用に追加
from openslide import OpenSlide
import tqdm
from blur_laplacian import get_blur
from saturation_otsu import get_slice_idx

home_dir = "/workspace/01-toxicological-data-extractor/data/collected_data"
output_base_dir = "/workspace/02_inhouse-vqvae/VQVAE/data/preprocessed_patches"

files = [f for f in os.listdir(home_dir) if f.endswith(('.svs', '.tif', '.ndpi'))]

for file in tqdm.tqdm(files, desc="Total Slides"):
    image_number = os.path.splitext(file)[0]
    wsi_path = os.path.join(home_dir, file)
    
    slide_save_dir = os.path.join(output_base_dir, "out_csv")
    os.makedirs(slide_save_dir, exist_ok=True)

    wsi = OpenSlide(wsi_path)
    patch_size = 256
    kernel = 'near4'
    
    # 1. 切片の識別（背景マスクの取得）
    # slice_idx: (H, W) の配列。背景は -1, 切片は 0, 1, 2...
    slice_idx, n_slice = get_slice_idx(wsi, patch_size)

    # 2. ラプラシアンフィルタによるBlurスコア取得
    blur = get_blur(wsi, patch_size, kernel, show_tqdm=True)

    # 3. 背景を除外して結果を格納
    results = []
    # blurの形状に合わせてループ
    h, w = blur.shape
    for i in range(h):
        for j in range(w):
            s_id = slice_idx[i, j]
            
            # 背景（-1）でなければリストに追加
            if s_id != -1:
                results.append({
                    "patch_name": f"patch_{i}_{j}.jpeg",
                    "coord_y": i,
                    "coord_x": j,
                    "slice_id": s_id, # どの切片か
                    "laplacian_score": blur[i, j]
                })

    # DataFrameに変換してCSV保存
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(slide_save_dir, f"{image_number}_blur_scores.csv")
        df.to_csv(csv_path, index=False)