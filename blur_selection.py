import sys, os
import numpy as np
import matplotlib.pyplot as plt
from openslide import OpenSlide
import shutil
import tqdm
import pandas as pd
# 自作モジュール
from blur_laplacian import get_blur

# パスの設定
home_dir = "/workspace/01-toxicological-data-extractor/data/collected_data"
base_target_dir = "/workspace/03-vq-patho-anomaly-detection/data/shionogi_wsi/pathch_taa"
# 出力先を整理
output_base_dir = "/workspace/02_inhouse-vqvae/VQVAE/data/preprocessed_patches"

files = [f for f in os.listdir(home_dir) if f.endswith(('.svs', '.tif', '.ndpi'))] # 拡張子でフィルタリングすると安全
save_dict = {}

for file in tqdm.tqdm(files):
    # 拡張子を除いたファイル名を取得 (修正ポイント)
    image_number = os.path.splitext(file)[0]
    
    patch_size = 256
    kernel = 'near4'

    # WSIの読み込み
    wsi_path = os.path.join(home_dir, file)
    wsi = OpenSlide(wsi_path)
    
    # ブラー（ボケ）判定の計算
    blur = get_blur(wsi, patch_size, kernel, show_tqdm=True)

    save_dict[f"{file}"] = blur.mean()  # 平均値を保存（必要に応じて変更）]
    print(f"Processed {file}: Mean Blur = {blur.mean():.2f}")

    # 既存パッチの格納ディレクトリ
    # current_image_dir = os.path.join(base_target_dir, image_number)
    
    # if not os.path.exists(current_image_dir):
    #     print(f"No dir exist: {current_image_dir}")
    #     continue

    # # ブラーの閾値を超えたパッチ名のセットを作成
    # target_set = set([f"patch_{i}_{j}.jpeg" for i, j in np.argwhere(blur > 30)])
    
    # # 既存のディレクトリ（malignantなど）を走査
    # for sub_dir in os.listdir(current_image_dir):
    #     sub_dir_path = os.path.join(current_image_dir, sub_dir)
    #     if not os.path.isdir(sub_dir_path):
    #         continue
            
    #     # 出力先ディレクトリの作成
    #     save_dir = os.path.join(output_base_dir, image_number, sub_dir)
    #     os.makedirs(save_dir, exist_ok=True)

    #     for patch_file in os.listdir(sub_dir_path):
    #         if patch_file in target_set:
    #             src_path = os.path.join(sub_dir_path, patch_file)
    #             dst_path = os.path.join(save_dir, patch_file)
                
    #             # ファイルのコピー
    #             shutil.copy2(src_path, dst_path)
# ブラーの平均値をCSVに保存
blur_df = pd.DataFrame(list(save_dict.items()), columns=['filename', 'mean_blur'])
blur_df.to_csv(os.path.join(output_base_dir, "blur_scores_TGGATE.csv"), index=False)