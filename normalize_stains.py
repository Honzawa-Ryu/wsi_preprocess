import staintools
import os
import glob
import cv2

# 基準となる画像を設定する
target = staintools.read_image("/workspace/inhouse-vqvae/VQVAE/data/preprocessed/27527/slice_0/patch_2_70.jpeg")
target = staintools.LuminosityStandardizer.standardize(target)

# ノーマライザーの定義
# Vahadaneを指定している
normalizer = staintools.StainNormalizer(method='vahadane')
normalizer.fit(target)

# 入力画像と出力画像のディレクトリを指定
INPUT_DIR = "/workspace/inhouse-vqvae/VQVAE/data/preprocessed"
OUTPUT_DIR = "/workspace/inhouse-vqvae/VQVAE/data/normalize_vahadane"

search_pattern = os.path.join(INPUT_DIR, "**", "*.jpeg")
image_paths = glob.glob(search_pattern, recursive=True)

for path in image_paths:
    relative_path = os.path.relpath(path, INPUT_DIR)
    output_path = os.path.join(OUTPUT_DIR, relative_path)
    output_dir = os.path.dirname(output_path)

    os.makedirs(output_dir, exist_ok=True)

    to_transform = staintools.read_image(path)
    to_transform = staintools.LuminosityStandardizer.standardize(to_transform)

    transformed = normalizer.transform(to_transform)

    cv2.imwrite(output_path, transformed)
