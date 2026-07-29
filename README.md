# wsi_preprocess
病理画像(WSI)の前処理用のツールを集めたもの。

WSIからパッチを切り出し, ぼやけを定量し, そこから学習用データセットを構築するまでを扱う。

Slurm環境での実行手順（`.sif`の作成からデータセット書き出しまで）は
[docs/howtouse.md](docs/howtouse.md) にまとめてある。

# 環境構築
WSIの読み出しには [tiffslide](https://github.com/Bayer-Group/tiffslide) を使っている。
openslideと違いシステムパッケージのインストールは不要で, Pythonパッケージのみで完結する。

```
uv sync
```

LMDB形式でデータセットを出力する場合のみ, 追加で以下が必要。
```
uv add lmdb
```
WebDataset形式(tar)の書き出しは標準ライブラリのみで行うため追加パッケージは不要。
読み出し側で `webdataset` を使う場合は `uv add webdataset`。

# 全体の流れ

パッチの出所によって2通りの入口がある。どちらも最終的には同じ形式のh5になり,
以降の手順を共通で使える。

```
[A] 生WSI (.svs 等)                    [B] TRIDENTでパッチ切り出し済み
        |                                       |
        | scripts/make_patches.py               | (coordsのみを持つh5)
        v                                       v
   h5 (images + coords + blur)          scripts/add_blur_scores.py
        |                                       | 元WSIから画素を読み直して
        |                                       | blurスコアをh5に追記
        +------------------+--------------------+
                           v
                 scripts/build_dataset.py
              閾値でフィルタ -> スライドごとにN枚を抽出
                           v
              WebDataset (tar) または LMDB
```

# 含まれているツール

## 背景の判別・除去 + パッチ切り出し (src/wsi_processer.py)
WSIから背景部分を除き, 組織が含まれるpatchを取り出す。
大津法で組織部を抽出し, 連結成分解析でスライスごとにIDを振る。
切り出しと同時にぼやけスコアも計算してh5に保存する。

```
python scripts/make_patches.py <slide_dir> <out_dir> --n_workers 8
```

`<slide_dir>` の中がフォルダ分けされていても再帰的に探索し, 出力側にも同じ相対パスを
再現する。深さは問わない。

```
data/caseA/2024/slide001.svs  ->  out/caseA/2024/slide001.h5
data/caseB/slide002.svs       ->  out/caseB/slide002.h5
data/slide003.svs             ->  out/slide003.h5
```

## ぼやけの定量 (src/blur.py, scripts/add_blur_scores.py)
ぼやけの指標は2種類あり, いずれも **値が大きいほど鮮明**。

| データセット名 | 指標 |
| --- | --- |
| `blur_scores_val` | ラプラシアンの分散 |
| `blur_scores_mean` | near8フィルタ応答の絶対値平均 |

TRIDENTなどが出力した `coords` のみのh5に対しては, 元WSIから画素を読み直して
スコアを計算し, 同じh5に追記する。

```
python scripts/add_blur_scores.py <h5_dir> --wsi_dir <wsi_dir> --n_workers 8
```

- パッチサイズと読み出しレベルはh5の属性(`patch_size` / `patch_level`)から自動取得する。
  属性が無い, あるいは上書きしたい場合は `--patch_size` / `--patch_level` で指定する。
- h5が既に画素(`images`)を持つ場合は `--wsi_dir` は不要。
- すでにスコアがあるh5はスキップされる。再計算したい場合は `--overwrite`。
- `<h5_dir>` も `--wsi_dir` もフォルダ分けされていてよく, 再帰的に探索する。
  同名のWSIが複数フォルダにある場合は, h5と同じ相対フォルダにあるものを優先して
  対応付ける。

## 学習用データセットの構築 (src/dataset_builder.py, scripts/build_dataset.py)
blurスコアの閾値でパッチを絞り込み, スライドごとにN枚をランダム抽出してまとめる。

```
python scripts/build_dataset.py <h5_dir> <out_dir> \
    --wsi_dir <wsi_dir> \
    --threshold 100 --n_per_slide 200 \
    --format webdataset --n_workers 8
```

主なオプション:

| オプション | 説明 |
| --- | --- |
| `--metric` | 閾値判定に使う指標 (既定: `blur_scores_val`) |
| `--threshold` | 絶対閾値。これ以上のスコアのパッチを候補にする |
| `--threshold_percentile` | スライドごとの分位点で閾値を決める(0-100) |
| `--n_per_slide` | スライドあたりの抽出枚数。未指定なら閾値を満たす全パッチ |
| `--format` | `webdataset`(既定) または `lmdb` |
| `--codec` | `png`(既定) または `npy`。いずれも可逆 |
| `--seed` | サンプリングの乱数シード (既定: 42) |

閾値については, スキャナや染色の違いでスコアの絶対値が動くため,
複数施設のデータを混ぜる場合は `--threshold_percentile` の方が安定する。

符号化はPNGとnpyのいずれも可逆で, JPEGは採用していない。
非可逆圧縮は高周波成分を落とすため, ぼやけの評価そのものを壊してしまうため。

出力先には `manifest.json` が生成され, 使った閾値やseed, スライドごとの採用枚数が記録される。

`<h5_dir>` の中はフォルダ分けされていてよい。`slide_id` は通常h5のファイル名(stem)だが,
別フォルダに同名のh5があるときだけ, 衝突するものを `caseA_slide001` のように相対パス由来の
IDに置き換えて一意にする(衝突していないスライドのIDは変わらない)。
元のh5は `manifest.json` の `per_slide[slide_id]["path"]` から辿れる。

### 出力の読み出し

WebDataset:
```python
import webdataset as wds, glob

ds = wds.WebDataset(sorted(glob.glob("out_dir/*.tar"))).decode("rgb8").to_tuple("png", "json")
for img, meta in ds:      # img: (H, W, 3) uint8
    print(meta["slide_id"], meta["blur_scores_val"])
```

LMDB:
```python
import lmdb, pickle, cv2, numpy as np

env = lmdb.open("out_dir", readonly=True, lock=False)
with env.begin() as txn:
    n = pickle.loads(txn.get(b"__len__"))
    rec = pickle.loads(txn.get(b"%08d" % 0))
    img = cv2.cvtColor(cv2.imdecode(np.frombuffer(rec["data"], np.uint8), cv2.IMREAD_COLOR),
                       cv2.COLOR_BGR2RGB)
```

# 実行例
exampleディレクトリ内にツールを使っている例があります(処理する病理画像は含まれていません)。
