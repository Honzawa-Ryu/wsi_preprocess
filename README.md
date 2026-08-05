# wsi_preprocess
病理画像(WSI)の前処理用のツールを集めたもの。

WSIからパッチを切り出し, ぼやけを定量し, そこから学習用データセットを構築するまでを扱う。
コアのセグメンテーション・パッチ座標抽出・特徴量抽出は [TRIDENT](https://github.com/mahmoodlab/TRIDENT)
に委譲し, ぼやけスコア付与と学習用パッチ抽出をその前後に挟み込むラッパー構成になっている
(実装方針は [PLAN.md](PLAN.md) を参照)。TRIDENTはリポジトリに同梱せず, `pyproject.toml` の
`[tool.uv.sources]` でGitHubから直接インストールする通常の依存パッケージとして扱っている。

Slurm環境での実行手順（`.sif`の作成からデータセット書き出しまで）は
[docs/howtouse.md](docs/howtouse.md) にまとめてある。

# 環境構築
WSIの読み出しには [tiffslide](https://github.com/Bayer-Group/tiffslide) を使っている
(独自処理側。TRIDENT自体はopenslideを使う)。tiffslideはopenslideと違いシステムパッケージの
インストールは不要で, Pythonパッケージのみで完結する。

```
uv sync
```
TRIDENTは `[tool.uv.sources]` で固定したコミットからビルドされるため, これだけで
`import trident` が使えるようになる。トップレベルにTRIDENT本体は展開されない
(`.venv/lib/.../site-packages/trident/` に入る)。

LMDB形式でデータセットを出力する場合のみ, 追加で以下が必要。
```
uv add lmdb
```
WebDataset形式(tar)の書き出しは標準ライブラリのみで行うため追加パッケージは不要。
読み出し側で `webdataset` を使う場合は `uv add webdataset`。

# 統合パイプライン (main.py)

TRIDENTでのセグメンテーション+座標抽出から, ぼやけスコア付与, 学習用パッチ抽出までを
1コマンドで実行できる。内部的には後述の各ツールを順番に呼び出しているだけなので,
個別に実行したい場合は下の「含まれているツール」を直接使えばよい。

```bash
python main.py \
    --wsi_dir data/wsis \
    --out_dir results/trident \
    --calc_blur \
    --extract_patches 1000
```

| ステップ | 内容 | 実行条件 |
| --- | --- | --- |
| Step1 | TRIDENTでセグメンテーション+パッチ座標抽出 | 常に実行 |
| Step2 | ぼやけスコアの計算・付与 | `--calc_blur` |
| Step3 | TRIDENTで特徴量抽出 | `--patch_encoder <name>` (例: `uni_v1`) を指定したときのみ |
| Step4 | 学習用パッチの抽出(画像ファイル+CSV) | `--extract_patches N` を指定したときのみ |

Step2〜4を既定で無効にしているのは, 無条件にパッチ画像や特徴量を書き出すとディスクと
実行時間を圧迫しやすいため。`--segmenter` / `--device` は未指定ならGPUの有無で自動選択する
(GPUが無ければ `otsu` + CPU)。主なオプションは `python main.py --help` を参照。

Python APIとしても呼べる。
```python
from main import run_pipeline

run_pipeline(
    wsi_dir="data/wsis",
    out_dir="results/trident",
    calc_blur=True,
    extract_patches=1000,
)
```

WebDataset/LMDBとしてまとめたい場合は, Step4の代わりに後述の
`scripts/build_dataset.py` を `--out_dir` 配下の座標h5(`<out_dir>/<mag>x_<ps>px_.../patches`)
に対して実行する。

# 全体の流れ(個別ツールを使う場合)

座標(coords)のみを持つh5を入口にして, 以降の手順を共通で使える。
そのh5は `main.py` のStep1(TRIDENT呼び出し, GPUがあれば`hest`, 無ければ`otsu`でCPUのみでも動く)
で作ってもよいし, 別途TRIDENTの `run_batch_of_slides.py` を直接実行して作ってもよい。
どちらで作ったcoords h5でも, 以降は同じスクリプトで扱える。

```
TRIDENT (main.py Step1 / run_batch_of_slides.py)
        |
        | coordsのみを持つh5
        v
scripts/add_blur_scores.py         (main.py --calc_blur も同じ処理)
        | 元WSIから画素を読み直して
        | blurスコアをh5に追記
        v
scripts/build_dataset.py
   閾値でフィルタ -> スライドごとにN枚を抽出
        v
   WebDataset (tar) または LMDB
```

# 含まれているツール

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
TRIDENTが書き出す `<slide>_patches.h5` という名前も, 末尾の `_patches` を自動で外して
`slide_id`(=元のWSI名)として扱う。

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
