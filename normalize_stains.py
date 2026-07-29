import os
import glob
import random
from numpy.linalg import LinAlgError
from tiatoolbox.tools import stainnorm
from tiatoolbox.utils.misc import imread, imwrite

target = imread("/workspace/inhouse-vqvae/VQVAE/data/preprocessed/27527/slice_0/patch_2_70.jpeg")
# target = staintools.LuminosityStandardizer.standardize(target)

normalizer = stainnorm.VahadaneNormalizer()
normalizer.fit(target)

INPUT_DIR = "/workspace/inhouse-vqvae/VQVAE/data/preprocessed"
OUTPUT_DIR = "/workspace/inhouse-vqvae/VQVAE/data/normalize_vahadane"

dirs = os.listdir(INPUT_DIR)
sorted_dirs = sorted(dirs)

for sorted_dir in sorted_dirs:
    print(sorted_dir)
    search_pattern = os.path.join(INPUT_DIR, sorted_dir, "**", "*.jpeg")
    image_paths = glob.glob(search_pattern, recursive=True)
    # random.seed(42)
    # random.shuffle(image_paths)
    image_paths = sorted(image_paths)
    print(f"{sorted_dir} から {len(image_paths)} 枚の画像をノーマライズします")
    output_dir = os.path.join(OUTPUT_DIR, sorted_dir)
    os.makedirs(output_dir, exist_ok=True)

    for path in image_paths:
        try:
            to_transform = imread(path)
            # to_transform = staintools.LuminosityStandardizer.standardize(to_transform)

            transformed = normalizer.transform(to_transform)
            
            file_name = os.path.basename(path)
            output_path = os.path.join(output_dir, file_name)
            imwrite(output_path, transformed)
        except LinAlgError as e:
            print(f"警告: {path} のSVDが収束せずスキップします。エラー: {e}")
            continue
        except Exception as e:
            print(f"その他のエラー: {path} をスキップします。エラー: {e}")
            continue

