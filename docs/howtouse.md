# Slurm環境での実行手順

`.sif` の作成から学習用データセットの書き出しまで、上から順にコピペすれば通るように書いてある。
ノード指定は `-p large-preproc` / `-t 4:00:00` を仮に置いているので、環境に合わせて読み替えること。

---

## 0. 前提とディレクトリ構成

入力WSIは `data/` に置く。**中でフォルダ分けされていても再帰的に探索する**ので、
症例ごと・年度ごとなど好きに切ってよい。出力側にも同じ相対パスが再現される。

```
wsi_preprocess/
├── .env/
│   ├── env.def          # Apptainerの定義ファイル（リポジトリに同梱）
│   └── env.sif          # ← これから作る（.gitignore済み）
├── data/                # 入力WSI。深さは問わない
│   ├── caseA/2024/slide001.svs
│   ├── caseB/slide002.svs
│   └── slide003.svs
├── scripts/
├── src/
├── logs/                # ← ログの出力先。作っておかないとジョブが落ちる
└── results/
    ├── h5/              # パッチh5。dataのフォルダ構造を保つ
    │   ├── caseA/2024/slide001.h5
    │   ├── caseB/slide002.h5
    │   └── slide003.h5
    └── dataset/         # 最終成果物（WebDataset tar）
```

探索対象の拡張子は `.svs .ndpi .tif .tiff .mrxs .scn .vms .bif`。パッチh5は `.h5 .hdf5`。
探索の挙動は3つのスクリプトで共通で、次のようになっている。

- 拡張子の大文字小文字は区別しない（`.SVS` のような表記が混ざっていても拾う）
- シンボリックリンクされたディレクトリも辿る（実体を別ボリュームに置いた構成に対応）
- ディレクトリの代わりに単一ファイルを渡してもよい

`data/A/slide001.svs` と `data/B/slide001.svs` のように**別フォルダに同名のスライド**が
あってもよい。その場合の扱いは手順5と6に書いてある。

まず作業ディレクトリを決めておく。以降のコマンドはすべてここが起点。

```bash
export PROJ=/path/to/wsi_preprocess     # ← 自分のパスに変更
cd "$PROJ"
mkdir -p logs results
```

---

## 1. `.sif` の作成

ログインノード上で1回だけ実行する。`--fakeroot` が使えない環境ではシステム管理者に確認すること。

```bash
cd "$PROJ"

# ビルド中の一時ファイルは数GBになる。/tmp が小さい環境だと ENOSPC で落ちるので退避させる
export APPTAINER_TMPDIR="$PROJ/.apptainer_tmp"
export APPTAINER_CACHEDIR="$PROJ/.apptainer_cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

apptainer build --fakeroot .env/env.sif .env/env.def
```

ビルドが終わったら一時ディレクトリは消してよい。

```bash
rm -rf "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
```

確認:

```bash
apptainer exec .env/env.sif uv --version
```

> `.env/env.def` は CUDA イメージがベースになっているが、この前処理はすべてCPUで動く。
> GPUを使う他の作業と `.sif` を共用する想定でなければ、`Bootstrap` 行を
> `ubuntu:24.04` に置き換えるとイメージが大幅に小さくなる。

---

## 2. Python環境の準備

`uv sync` でプロジェクト直下に `.venv` を作る。これもログインノードで1回だけ。
計算ノードはネットワークが閉じていることが多いので、**ジョブ投入前に済ませておく**。

```bash
cd "$PROJ"
apptainer exec --bind "$PROJ:$PROJ" .env/env.sif uv sync
```

LMDB形式で出力したい場合のみ追加する（WebDataset形式なら不要）。

```bash
apptainer exec --bind "$PROJ:$PROJ" .env/env.sif uv add lmdb
```

確認:

```bash
apptainer exec --bind "$PROJ:$PROJ" .env/env.sif \
    .venv/bin/python -c "import tiffslide, h5py, cv2; print('ok')"
```

以降、コンテナ内では `uv run` ではなく `.venv/bin/python` を直接叩く。
`uv run` は実行のたびに依存解決を試みるため、ネットワークの無い計算ノードで詰まることがある。

---

## 3. 動作確認（1枚だけ対話実行）

いきなり全件流す前に、`srun` で1枚だけ通しておくと事故が減る。

```bash
cd "$PROJ"

srun -p large-preproc -t 0:30:00 -c 4 --mem=16G --pty \
apptainer exec --bind "$PROJ:$PROJ" .env/env.sif \
    .venv/bin/python scripts/make_patches.py \
        data results/h5_test --n_workers 4 --visualize
```

`results/h5_test/` 以下に `.h5` と `_vis.png` ができていれば通っている。
`_vis.png` はサムネイル上に採用パッチを点で描いたもので、背景除去が効いているかの確認に使える。

中身の確認:

```bash
apptainer exec --bind "$PROJ:$PROJ" .env/env.sif .venv/bin/python - <<'EOF'
import glob, h5py
for p in sorted(glob.glob("results/h5_test/**/*.h5", recursive=True)):
    with h5py.File(p) as f:
        print(p, "| patches:", len(f["coords"]), "| keys:", list(f))
EOF
```

パッチが0枚なら「4-1. パッチが0枚になる」を参照。

---

## 4. 経路A: 生WSIからパッチを切り出す

`data/` 以下のWSIを再帰的に処理し、`results/h5/` に同じ構造で `.h5` を書き出す。
このスクリプトは切り出しと同時にぼやけスコアも計算するので、次は手順6に進んでよい。

```bash
cd "$PROJ"
cat > run_make_patches.sh <<'EOF'
#!/bin/bash
#SBATCH -p large-preproc
#SBATCH -t 4:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -J make_patches
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# --n_workers を渡さなければ SLURM_CPUS_PER_TASK を自動で拾う
apptainer exec --bind "$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR" .env/env.sif \
    .venv/bin/python scripts/make_patches.py \
        data \
        results/h5
EOF

sbatch run_make_patches.sh
```

`--visualize` を足すとスライドごとに確認用PNGも出る（枚数が多いと相応に時間とディスクを食う）。

これが一番重い工程で、4時間に収まらないことがある。その場合は手順7の分割実行へ。

---

## 5. 経路B: TRIDENTなどで切り出し済みのh5にスコアを足す

すでに `coords` だけを持つh5がある場合はこちら。元WSIから画素を読み直してスコアを計算し、
**同じh5に追記する**（h5は上書きされるので、心配なら先にコピーを取ること）。

`--wsi_dir` に渡したディレクトリからも、h5のファイル名を手がかりにWSIを再帰的に探す。

```bash
cd "$PROJ"
cat > run_add_blur.sh <<'EOF'
#!/bin/bash
#SBATCH -p large-preproc
#SBATCH -t 4:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -J add_blur
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

apptainer exec --bind "$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR" .env/env.sif \
    .venv/bin/python scripts/add_blur_scores.py \
        results/h5 \
        --wsi_dir data \
        --n_workers "$SLURM_CPUS_PER_TASK"
EOF

sbatch run_add_blur.sh
```

- パッチサイズと読み出しレベルはh5の属性から自動取得する。属性が無い場合は
  `--patch_size 256 --patch_level 0` のように明示する。
- すでにスコアがあるh5はスキップされる。再計算したいときは `--overwrite`。
- 経路Aで作ったh5に対して実行しても、スキップされるだけで害はない。
- h5側もフォルダ分けされていてよい。`--wsi_dir` 以下に同名のWSIが複数ある場合は、
  **h5と同じ相対フォルダにあるものを優先**して対応付ける（`results/h5/caseA/x.h5`
  なら `data/caseA/x.svs`）。該当が無ければ末尾のフォルダ名の一致で拾い、それも
  無ければ拡張子の優先度とパス順で決める。取り違えると別スライドのスコアが入るので、
  同名スライドがある場合は起動時の警告が出たら対応関係を確認しておくこと。

---

## 6. 学習用データセットの構築

ぼやけスコアで足切りし、スライドごとにN枚を抽出してWebDataset(tar)にまとめる。

```bash
cd "$PROJ"
cat > run_build_dataset.sh <<'EOF'
#!/bin/bash
#SBATCH -p large-preproc
#SBATCH -t 4:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -J build_dataset
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

apptainer exec --bind "$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR" .env/env.sif \
    .venv/bin/python scripts/build_dataset.py \
        results/h5 \
        results/dataset \
        --threshold_percentile 50 \
        --n_per_slide 200 \
        --format webdataset \
        --codec png \
        --seed 42 \
        --n_workers "$SLURM_CPUS_PER_TASK"
EOF

sbatch run_build_dataset.sh
```

閾値の指定は2通りあり、**どちらか一方のみ**指定する。

| 指定 | 意味 | 使いどころ |
| --- | --- | --- |
| `--threshold 100` | 絶対閾値。スコアがこれ以上のパッチを候補にする | 単一施設・単一スキャナ |
| `--threshold_percentile 50` | スライドごとの分位点。上位(100-p)%を候補にする | 複数施設が混ざる場合 |

スコアの絶対値はスキャナや染色で動くため、素性の違うデータを混ぜるなら分位点の方が安定する。
まず `--threshold_percentile` で通し、`results/dataset/manifest.json` の `per_slide` を見て
スライドごとの採用枚数に極端な偏りが無いか確認するとよい。

経路B（h5が `coords` のみ）の場合は、ここでも元WSIが要るので `--wsi_dir data` を足す。

```bash
        --wsi_dir data \
```

主なオプションは `README.md` の表を参照。符号化は `png` / `npy` のいずれも可逆で、
JPEGは採用していない（非可逆圧縮は高周波を落とし、ぼやけの評価そのものを壊すため）。

---

## 7. まとめて流す

依存関係を付けて一気に投げる場合。前段が正常終了したときだけ次が走る。

```bash
cd "$PROJ"
JID1=$(sbatch --parsable run_make_patches.sh)
JID2=$(sbatch --parsable --dependency=afterok:"$JID1" run_build_dataset.sh)
echo "make_patches=$JID1  build_dataset=$JID2"
```

経路Bなら `run_add_blur.sh` を間に挟む。

```bash
JID1=$(sbatch --parsable run_add_blur.sh)
JID2=$(sbatch --parsable --dependency=afterok:"$JID1" run_build_dataset.sh)
```

### スライドが多くて4時間に収まらない場合

`data/` 直下のサブディレクトリ単位でジョブ配列に分ける。
出力構造は保たれるので、あとから `results/h5` 全体を1つとして扱える。

```bash
cd "$PROJ"

# 処理単位のリストを作る（data/直下のディレクトリ名）
find data -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort > subsets.txt
wc -l < subsets.txt

cat > run_make_patches_array.sh <<'EOF'
#!/bin/bash
#SBATCH -p large-preproc
#SBATCH -t 4:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -J make_patches_arr
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

SUBSET=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" subsets.txt)
echo "subset: $SUBSET"

apptainer exec --bind "$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR" .env/env.sif \
    .venv/bin/python scripts/make_patches.py \
        "data/$SUBSET" \
        "results/h5/$SUBSET"
EOF

# 同時実行数は %8 の部分で制限する（I/O帯域に合わせて調整）
sbatch --array=0-$(($(wc -l < subsets.txt) - 1))%8 run_make_patches_array.sh
```

---

## 8. 進捗の確認

```bash
squeue -u "$USER"
tail -f logs/make_patches_*.out
```

`make_patches` は10秒おきにCPU/RAM使用率を `.out` に吐くので、長時間ジョブではログが伸びる。
不要なら `scripts/make_patches.py` の `monitor_resources` スレッド起動行をコメントアウトする。

終了サマリは各スクリプトの末尾に出る。失敗したスライドがあれば終了コードが1になるので、
`sacct` でも判別できる。

```bash
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS
```

---

## 9. 出力の読み出し

WebDataset:

```python
import webdataset as wds, glob

ds = (wds.WebDataset(sorted(glob.glob("results/dataset/*.tar")))
        .decode("rgb8")
        .to_tuple("png", "json"))
for img, meta in ds:      # img: (256, 256, 3) uint8
    print(meta["slide_id"], meta["x"], meta["y"], meta["blur_scores_val"])
    break
```

`results/dataset/manifest.json` に、使った閾値・seed・スライドごとの採用枚数が記録されている。
データセットの素性はここを見れば追える。

LMDB形式で出した場合の読み出しは `README.md` を参照。

---

## 10. つまずきやすいところ

### 10-1. パッチが0枚になる

組織として認める最小面積が `patch_size**2 * 500`（256なら約3270万px、レベル0換算で
おおよそ5700×5700px相当）に固定されている。これを下回る組織しか無いスライド
——生検の小片、TMA、切り出し済みの小さい画像——では、全パッチが落ちて0枚になる。

`--visualize` を付けて `_vis.png` を確認し、組織が写っているのに点が乗らないならこれが原因。
`src/wsi_processer.py` の `threshold_area_thumb` の係数 `500` を下げる。

ピラミッド（縮小画像）を持たない単層TIFFでも同じことが起きる。サムネイルのレベル決定が
`min(2, レベル数-1)` なので、単層だと等倍がサムネイル扱いになり面積閾値が実質無限大になる。

### 10-2. 同名のh5が複数フォルダにある

`slide_id` は通常ファイル名(stem)をそのまま使うが、`results/h5/A/x.h5` と
`results/h5/B/x.h5` のように衝突する場合だけ、`build_dataset.py` が相対パスを繋いだ
`A_x` / `B_x` に置き換える（衝突していないスライドのIDは変わらない）。

置き換えが起きると起動時に `Note: 同名のh5が複数フォルダにあります` と対応表が出る。
元のh5は `manifest.json` の `per_slide[slide_id]["path"]` から辿れる。
素のスライド名でIDを揃えたいなら、事前にリネームしてから流すこと。

### 10-3. 計算ノードでファイルが見えない

Apptainerが自動でバインドするのは `$HOME` とカレントディレクトリ程度。
データが別のファイルシステムにあるなら明示的に足す。

```bash
apptainer exec --bind "$PROJ:$PROJ" --bind /mnt/storage:/mnt/storage .env/env.sif ...
```

### 10-4. `uv sync` が計算ノードで失敗する

計算ノードからは外部ネットワークに出られないことが多い。手順2をログインノードで済ませ、
ジョブ内では `.venv/bin/python` を直接呼ぶこと（ジョブスクリプト内で `uv run` を使わない）。

### 10-5. ビルド中に `No space left on device`

`APPTAINER_TMPDIR` が小さい `/tmp` を向いている。手順1のとおり容量のある場所に退避させる。
