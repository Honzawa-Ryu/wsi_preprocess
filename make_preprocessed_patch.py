import numpy as np
import openslide
from openslide.deepzoom import DeepZoomGenerator
import os
from saturation_otsu import get_slice_idx
from PIL import Image

# （ユーザー提供の）get_slice_idx関数をここに貼り付けるか、インポートしてください

def save_patches_from_wsi(image_path, patch_size, output_dir, slice_min_patch=500):
    """
    WSIから組織切片のパッチをオフラインで作成・保存する関数

    Parameters
    ----------
    image_path: str
        WSIファイルのパス
    patch_size: int
        1つのパッチの大きさ（ピクセル単位）
    output_dir: str
        パッチを保存するディレクトリ
    slice_min_patch: int
        切片として認識する最小パッチ数
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    slide = openslide.OpenSlide(image_path)
    
    # get_slice_idxでパッチのインデックス情報を取得
    slice_idx, n_slice = get_slice_idx(slide, patch_size, slice_min_patch=slice_min_patch)

    if n_slice == 0:
        print("組織切片が見つかりませんでした。")
        return

    # DeepZoomGeneratorを使って効率的にパッチを生成
    # level=0が最低解像度
    # tile_size=patch_size, overlap=0, limit_bounds=False
    # を設定することで、get_slice_idxで定義されたパッチと一致させる
    tiles = DeepZoomGenerator(slide, tile_size=patch_size, overlap=0, limit_bounds=False)
    
    # 組織切片ごとにパッチを保存
    for i_slice in range(n_slice):
        print(f"--- 切片 {i_slice+1}/{n_slice} のパッチを保存中 ---")
        
        # 該当する切片のパッチの座標を取得
        # whereで取得されるのは (y座標のリスト, x座標のリスト) のタプル
        patch_coords = np.where(slice_idx == i_slice)
        
        # 保存先ディレクトリを作成
        slice_output_dir = os.path.join(output_dir, f"slice_{i_slice}")
        if not os.path.exists(slice_output_dir):
            os.makedirs(slice_output_dir)

        # 各パッチを読み込み、保存
        for y, x in zip(patch_coords[0], patch_coords[1]):
            # openslide.OpenSlide.read_region(location, level, size)
            # locationは最高解像度レベル(level=0)での座標
            location_x = int(x * patch_size)
            location_y = int(y * patch_size)

            # tilesオブジェクトから直接タイル（パッチ）を取得
            # tiles.get_tile(level, address)
            # addressは (x, y) のタプル
            # tiles.level_count - 1は最高解像度レベル（最小のダウンサンプルレベル）
            try:
                # パッチをPIL Imageとして取得
                patch = tiles.get_tile(tiles.level_count - 1, (x, y))
                
                # 画像を保存
                patch.save(os.path.join(slice_output_dir, f"patch_{y}_{x}.jpeg"))

            except Exception as e:
                print(f"パッチ ({y}, {x}) の保存中にエラーが発生しました: {e}")
                continue

    print("パッチの保存が完了しました。")

# get_slice_idx 関数は、このコードの上部または別のファイルからインポートされていることを前提とします。
# def get_slice_idx(...): ...

def process_all_wsi_in_directory(input_dir, output_root_dir, patch_size, slice_min_patch=500):
    """
    指定されたディレクトリ内のすべてのWSIファイルを処理し、パッチを保存する関数。

    Parameters
    ----------
    input_dir: str
        WSIファイルが保存されているディレクトリのパス
    output_root_dir: str
        処理結果を保存するルートディレクトリのパス
    patch_size: int
        1つのパッチの大きさ（ピクセル単位）
    slice_min_patch: int
        切片として認識する最小パッチ数
    """
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)

    # 指定されたディレクトリ内のすべてのファイル名をリストアップ
    for filename in os.listdir(input_dir):
        # ファイルの拡張子をチェックし、WSIファイル（.svs, .tifなど）か確認
        if filename.lower().endswith(('.svs', '.tif', '.tiff', '.ndpi')):
            image_path = os.path.join(input_dir, filename)
            
            # 各WSIファイルごとに専用の出力ディレクトリを作成
            # 例: "sample1.svs" -> "output_patches/sample1"
            base_filename = os.path.splitext(filename)[0]
            output_dir = os.path.join(output_root_dir, base_filename)
            
            print(f"--- {filename} の処理を開始します ---")
            
            try:
                # WSIを処理する関数（前回の回答で示した関数）を呼び出す
                # `save_patches_from_wsi`関数をこのコード内で定義するか、インポートする必要があります
                save_patches_from_wsi(image_path, patch_size, output_dir, slice_min_patch)
                print(f"--- {filename} の処理が完了しました ---")
            except Exception as e:
                print(f"--- エラー: {filename} の処理中にエラーが発生しました: {e} ---")
                continue

# 実行例
if __name__ == "__main__":
    # WSIファイルが保存されているディレクトリ
    input_directory = "/workspace/Liver"
    
    # 全てのパッチを保存するルートディレクトリ
    output_root_directory = "/workspace/Liver/preprocessed"
    
    # パッチのサイズ
    p_size = 256
    
    # 関数を実行
    process_all_wsi_in_directory(input_directory, output_root_directory, p_size)