# wsi_preprocess

病理画像 (WSI) の前処理・組織検出・ボケ領域フィルタリング・色味調整（染色正規化）・パッチ抽出ツールおよび Python ライブラリです。

## 主な機能

- **組織領域セグメンテーション**: 大津の二値化＋モルフォロジー処理、およびマーカーペン跡（インク）除去オプション (`remove_pen=True`) による組織抽出。
- **ボケ領域（Blur）のフィルタリング**: ラプラシアン分散やエッジ絶対値平均を用いたボケ度合い評価と、指定しきい値以下のボケパッチ除外。
- **色味調整・染色正規化 (Stain Normalization)**:
  - **Reinhard 法**: L*a*b* 色空間における平均・標準偏差ベースの高速正規化
  - **Macenko 法**: 光学濃度 (OD) 空間におけるSVDベースの染色分離・正規化
  - **ColorAdjuster**: 明度・コントラスト・ガンマ補正
- **柔軟なパッチ出力**:
  - 従来通りの HDF5 (`.h5`) 出力
  - メモリ内ジェネレータ (`iter_patches`) による PyTorch データセット等の AI パイプラインへの直接連携

---

## インストール方法

openslide などの関連ツールをインストールの上、本パッケージをインストールしてください。

```bash
apt install -y openslide-tools
pip install -e .
```

---

## コマンドライン実行 (CLI ツール)

インストールすると `wsi-preprocess` コマンド、または `python -m wsi_preprocess` が利用できます。

### 基本例（ディレクトリ内の全WSIを一括処理）
```bash
wsi-preprocess -i ./data -o ./output --patch-size 256 --n-workers 8
```

### 色味の正規化（Reinhard法）やボケ除去・可視化画像をあわせて実行
```bash
wsi-preprocess \
  --input ./data/findings \
  --out-dir ./results \
  --patch-size 256 \
  --color-norm reinhard \
  --min-blur-val 50.0 \
  --remove-pen \
  --visualize \
  --n-workers 8
```

#### CLI オプション一覧
- `-i, --input`: WSIファイルまたはWSIを含むフォルダパス
- `-o, --out-dir`: 出力ディレクトリ
- `--patch-size`: パッチの一辺のピクセルサイズ (デフォルト: 256)
- `--tissue-thresh`: パッチ内の組織含有率のしきい値 (デフォルト: 0.2)
- `--min-tissue-area`: サムネイル上の最小組織連結領域面積 (デフォルト: 500.0)
- `--remove-pen`: マーカーペン跡の除去マスク適用
- `--min-blur-val`: ラプラシアン分散しきい値（これ未満のボケパッチをスキップ）
- `--color-norm`: 色味調整手法 (`none`, `reinhard`, `macenko`, `adjust`)
- `-n, --n-workers`: 並列プロセス数
- `--visualize`: サムネイル上に抽出パッチをプロットした可視化画像の保存
- `--coords-only`: 画像データを保存せず、座標とメタデータのみを保存 (MahmoodLab CLAM / TRIDENT 互換の軽量 H5 モード)

---

## MahmoodLab TRIDENT / CLAM との互換性

本ツールが出力する HDF5 (`.h5`) ファイルは、**Mahmood Lab** の **TRIDENT**, **CLAM**, **UNI**, **CONCH** 等の特徴量抽出パイプラインと完全互換です。

### 仕様特徴
1. **メタデータ属性 (`attrs`) の完全準拠**:
   - `f['coords'].attrs['patch_size']`, `f['coords'].attrs['patch_level']`, `f['coords'].attrs['downsample']` を自動格納
   - `f.attrs` にも同様にスライド情報やパッチメタデータを記録
2. **軽量座標オンリーモード (`--coords-only`)**:
   - TRIDENT/CLAM で一般的な「座標 H5 (`coords.h5`) を事前生成し、特徴抽出時に元SVSから動的にパッチを切り出す」ワークフローに対応
   - 実行時に `--coords-only` オプションを付与することで画像ストレージを大幅に削減できます

```bash
wsi-preprocess -i ./data -o ./coords_h5 --patch-size 256 --coords-only -n 8
```

---

## Python ライブラリとしての使い方

### 1. HDF5 への保存と可視化
```python
from wsi_preprocess import WSIProcessor

processor = WSIProcessor(
    slide_path="sample.svs",
    patch_size=256,
    tissue_threshold=0.2,
    remove_pen=True,
    min_blur_score_val=50.0,
    color_normalizer="reinhard",  # "reinhard", "macenko", あるいはインスタンスを渡す
)

# H5ファイルの生成
h5_path = processor.run(out_dir="output/")

# 可視化画像の生成
processor.visualize(save_path="output/sample_vis.png")
```

### 2. パッチイテレータを用いたオンメモリ取得 (深層学習パイプライン等への接続)
```python
from wsi_preprocess import WSIProcessor

processor = WSIProcessor("sample.svs", patch_size=256, color_normalizer="reinhard")

for item in processor.iter_patches(progress=True):
    patch_rgb = item["patch"]            # (256, 256, 3) uint8 RGB画像
    coords = item["coords"]              # [x, y] Level 0座標
    slice_id = item["slice_id"]          # 組織連結領域ID
    blur_score = item["blur_score_val"]  # 鮮明度スコア
    # ここで直接PyTorchモデルに入力等が可能
```
