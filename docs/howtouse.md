# Slurm環境での実行手順

`.sif` の作成から学習用データセットの書き出しまで、上から順にコピペすれば通るように書いてある。
ノード指定は `-p large-preproc` / `-t 4:00:00` を仮に置いているので、環境に合わせて読み替えること。
GPUノードのパーティション名(`-p gpu` としている箇所)も同様に仮置きなので読み替えること。

パッチ切り出しの経路は2通りある。迷ったら経路C(`main.py`, TRIDENTラッパー)から試すとよい。

| 経路 | 何をする | GPU |
| --- | --- | --- |
| B (手順4) | 外部で作った(TRIDENTなど)coordsのみのh5にスコアを後付け | 呼び出し側の責任 |
| C (手順5) | `main.py`でTRIDENT本体を呼び, セグメンテーション〜ぼやけ付与まで一括 | `hest`(既定)使用時のみ必要, `otsu`ならCPUのみで可 |

いずれの経路も最終的に同じ形式のh5(`coords` + ぼやけスコア)になり、以降の手順(6. データセット構築)を共通で使える。
(かつてはOtsu実装を自前で持つ「経路A」もあったが、TRIDENTの`--segmenter otsu`が同じことをより成熟した形で
行うため廃止した。生WSIから直接切り出したい場合は経路Cで`--segmenter otsu --device cpu`を使う。)

---

## 0. 前提とディレクトリ構成

入力WSIは `data/` に置く。**中でフォルダ分けされていても再帰的に探索する**ので、
症例ごと・年度ごとなど好きに切ってよい。出力側にも同じ相対パスが再現される。

```
wsi_preprocess/
├── .env/
│   ├── env.def          # Apptainerの定義ファイル（リポジトリに同梱）
│   └── env.sif          # ← これから作る（.gitignore済み）
├── main.py               # 統合パイプライン(経路C)のエントリーポイント
├── data/                 # 入力WSI。深さは問わない
│   ├── caseA/2024/slide001.svs
│   ├── caseB/slide002.svs
│   └── slide003.svs
├── scripts/
├── src/
├── logs/                 # ← ログの出力先。作っておかないとジョブが落ちる
└── results/
    ├── h5/               # 経路Bのパッチh5(TRIDENTを別途直接叩いた出力など)。dataのフォルダ構造を保つ
    │   ├── caseA/2024/slide001.h5
    │   ├── caseB/slide002.h5
    │   └── slide003.h5
    ├── trident/          # 経路C(main.py)の出力先(TRIDENTのjob_dirを兼ねる)
    └── dataset/          # 最終成果物（WebDataset tar）
```

探索対象の拡張子は `.svs .ndpi .tif .tiff .mrxs .scn .vms .bif`。パッチh5は `.h5 .hdf5`。
探索の挙動はスクリプト間で共通で、次のようになっている。

- 拡張子の大文字小文字は区別しない（`.SVS` のような表記が混ざっていても拾う）
- シンボリックリンクされたディレクトリも辿る（実体を別ボリュームに置いた構成に対応）
- ディレクトリの代わりに単一ファイルを渡してもよい

`data/A/slide001.svs` と `data/B/slide001.svs` のように**別フォルダに同名のスライド**が
あってもよい。その場合の扱いは手順4と6に書いてある。

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

> `.env/env.def` はCUDAイメージがベース。経路Bだけを使う(TRIDENT本体のセグメンテーション/
> 特徴量抽出モデルを動かさない)なら実行自体はCPUのみで完結するが、経路C(`main.py`)で既定の
> `--segmenter hest` や `--patch_encoder` を使う場合はGPU + CUDAが要るため、このイメージの
> ままにしておくこと。経路B専用で運用する(TRIDENTのGPUモデルを一切使わない)なら
> `Bootstrap` 行を `ubuntu:24.04` に置き換えてイメージを小さくしてもよいが、その場合は
> 経路Cで `--segmenter otsu --device cpu` を明示する必要がある。

---

## 2. Python環境の準備

`uv sync` でプロジェクト直下に `.venv` を作る。これもログインノードで1回だけ。
計算ノードはネットワークが閉じていることが多いので、**ジョブ投入前に済ませておく**。

```bash
cd "$PROJ"
apptainer exec --bind "$PROJ:$PROJ" .env/env.sif uv sync
```

TRIDENTは`pyproject.toml`の`[tool.uv.sources]`で固定したGitHubのコミットから直接ビルドされる
(リポジトリにはコードを同梱していない)。そのため`uv sync`は**GitHubにもアクセスできるネットワーク**
で実行する必要がある(社内プロキシ配下でPyPIのみ許可、GitHubは不可、という環境だと失敗する)。
torch含め1GB近くダウンロードするので、初回は数分かかる。

LMDB形式で出力したい場合のみ追加する（WebDataset形式なら不要）。

```bash
apptainer exec --bind "$PROJ:$PROJ" .env/env.sif uv add lmdb
```

確認:

```bash
apptainer exec --bind "$PROJ:$PROJ" .env/env.sif \
    .venv/bin/python -c "import tiffslide, h5py, cv2, trident; print('ok', trident.__version__)"
```

GPUを使う経路C(既定の`--segmenter hest`や`--patch_encoder`)を試すなら、GPUノード上で
CUDAが見えることも確認しておく(`--nv`を忘れるとCPUにフォールバックし気づきにくい)。

```bash
srun -p gpu --gres=gpu:1 -t 0:10:00 --pty \
apptainer exec --nv --bind "$PROJ:$PROJ" .env/env.sif \
    .venv/bin/python -c "import torch; print(torch.cuda.is_available())"
```

TRIDENTの一部のパッチエンコーダ(`uni_v1`など)はHugging Face上でgatedなモデルで、
アクセス許可の取得と`huggingface-cli login`がログインノード側(ネットワークが開いている側)で
事前に必要。ログイン情報は`$HOME/.cache/huggingface`に保存され、Apptainerは`$HOME`を
自動でバインドするため計算ノードのジョブからもそのまま使える。経路Cでも
セグメンテーション+ぼやけ付与だけ(`--patch_encoder`を指定しない)なら不要。

以降、コンテナ内では `uv run` ではなく `.venv/bin/python` を直接叩く。
`uv run` は実行のたびに依存解決を試みるため、ネットワークの無い計算ノードで詰まることがある。

---

## 3. 動作確認（1枚だけ対話実行）

いきなり全件流す前に、`srun` で1枚だけ通しておくと事故が減る。GPUノードの確保待ちを避けるため、
まずはCPUのみで動く`--segmenter otsu`で通し方を確認するのが手早い。

```bash
cd "$PROJ"

srun -p large-preproc -t 0:30:00 -c 4 --mem=16G --pty \
apptainer exec --bind "$PROJ:$PROJ" .env/env.sif \
    .venv/bin/python main.py \
        --wsi_dir data \
        --out_dir results/trident_test \
        --segmenter otsu --device cpu \
        --mag 20 --patch_size 256 \
        --calc_blur
```

`results/trident_test/20x_256px_0px_overlap/patches/` 以下に `<slide>_patches.h5` ができ、
`results/trident_test/contours/<slide>.jpg` に組織検出のオーバーレイ画像ができていれば通っている
（背景除去が効いているかはこの画像で確認できる）。

中身の確認:

```bash
apptainer exec --bind "$PROJ:$PROJ" .env/env.sif .venv/bin/python - <<'EOF'
import glob, h5py
for p in sorted(glob.glob("results/trident_test/**/*_patches.h5", recursive=True)):
    with h5py.File(p) as f:
        print(p, "| patches:", len(f["coords"]), "| keys:", list(f))
EOF
```

パッチが0枚なら「10-4. 経路Cでコマンドは通るのにパッチ/特徴量が0件」を参照。

---

## 4. 経路B: 外部で切り出し済みのh5にスコアを足す

すでに `coords` だけを持つh5がある場合(TRIDENTを別途手動で走らせた、他のメンバーの成果物を
使う、など)はこちら。元WSIから画素を読み直してスコアを計算し、**同じh5に追記する**
（h5は上書きされるので、心配なら先にコピーを取ること）。

自分でこれからTRIDENTを走らせるだけなら、この手順を手で行う必要はない。手順5(経路C)の
`main.py --calc_blur` が同じ処理をセグメンテーション・座標抽出とまとめて1コマンドで行う。

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
- h5側もフォルダ分けされていてよい。`--wsi_dir` 以下に同名のWSIが複数ある場合は、
  **h5と同じ相対フォルダにあるものを優先**して対応付ける（`results/h5/caseA/x.h5`
  なら `data/caseA/x.svs`）。該当が無ければ末尾のフォルダ名の一致で拾い、それも
  無ければ拡張子の優先度とパス順で決める。取り違えると別スライドのスコアが入るので、
  同名スライドがある場合は起動時の警告が出たら対応関係を確認しておくこと。
- TRIDENTが書き出す `<slide>_patches.h5` という名前は自動で認識し、`<slide>` 部分だけで
  WSIを探す(`_patches`サフィックスは無視される)。

---

## 5. 経路C: 統合パイプライン(main.py)でTRIDENTを実行する

TRIDENT本体でのセグメンテーション+パッチ座標抽出から, ぼやけスコア付与, (任意で)学習用パッチ
抽出までを`main.py`の1コマンドで行う。内部で呼んでいるのは手順4・6に出てくるのと同じ処理
(TRIDENTの`Processor`, `scripts/add_blur_scores.py`相当, `src/patch_extractor.py`)。
詳細は[README.md](../README.md)の「統合パイプライン (main.py)」および[PLAN.md](../PLAN.md)を参照。

### 5-1. CPUのみで動かす場合(`--segmenter otsu`)

GPUノードが確保できない、またはまず動作を確認したいとき。既存の`large-preproc`パーティションで動く。

```bash
cd "$PROJ"
cat > run_trident_cpu.sh <<'EOF'
#!/bin/bash
#SBATCH -p large-preproc
#SBATCH -t 4:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -J trident_cpu
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

apptainer exec --bind "$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR" .env/env.sif \
    .venv/bin/python main.py \
        --wsi_dir data \
        --out_dir results/trident \
        --segmenter otsu --device cpu \
        --mag 20 --patch_size 256 \
        --calc_blur \
        --blur_n_workers "$SLURM_CPUS_PER_TASK"
EOF

sbatch run_trident_cpu.sh
```

### 5-2. GPUで動かす場合(既定の`--segmenter hest`、特徴量抽出も使う場合)

`apptainer exec`に**`--nv`を付け忘れるとGPUが見えず、`main.py`は自動でCPU(`otsu`)側に
フォールバックしてしまう**(エラーにならず気づきにくいので注意)。

```bash
cd "$PROJ"
cat > run_trident_gpu.sh <<'EOF'
#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -J trident_gpu
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

apptainer exec --nv --bind "$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR" .env/env.sif \
    .venv/bin/python main.py \
        --wsi_dir data \
        --out_dir results/trident \
        --mag 20 --patch_size 256 \
        --calc_blur \
        --patch_encoder uni_v1 \
        --blur_n_workers "$SLURM_CPUS_PER_TASK"
EOF

sbatch run_trident_gpu.sh
```

`--patch_encoder`(Step3, 特徴量抽出)は省略可能。指定する場合は、選ぶエンコーダに応じて
`--mag`/`--patch_size`を正しい組み合わせに揃えること(例: `uni_v1`は`--patch_size 256 --mag 20`、
`conch_v15`は`--patch_size 512 --mag 20`)。ずれると特徴量として意味を成さない値になる。
選択肢と対応表は`.claude/skills/trident/reference.md`(Patch encoders節)を参照。

### 5-3. 出力先

```
results/trident/
├── contours_geojson/                    セグメンテーション結果
├── 20x_256px_0px_overlap/
│   ├── patches/<slide>_patches.h5       座標 + (--calc_blurなら)ぼやけスコア
│   ├── visualization/<slide>.jpg
│   └── features_uni_v1/<slide>.h5       --patch_encoderを指定した場合のみ
├── patches/                             --extract_patches を指定した場合のみ(画像+CSV)
└── summary.md                           TRIDENT側の実行サマリ
```

`--extract_patches N`を足すと、スライドごとに最大N枚を`results/trident/patches/`に
画像ファイル(png)として書き出す(手順6のWebDataset/LMDB化とは別の、簡易な取り出し方)。
WebDataset/LMDBとしてまとめたい場合は、`results/trident/20x_256px_0px_overlap/patches`を
入力に手順6の`scripts/build_dataset.py`を実行する(`--extract_patches`は使わない)。

再実行は`--out_dir`が同じであれば再開される(完了済みスライドはスキップ)。ただし
`--mag`/`--patch_size`/`--overlap`やエンコーダを変えると、TRIDENTは別ディレクトリ
(`<mag>x_<ps>px_...`)を新規に使うため、再開ではなく新規実行になる。

---

## 6. 学習用データセットの構築

ぼやけスコアで足切りし、スライドごとにN枚を抽出してWebDataset(tar)にまとめる。
経路B/Cのいずれで作ったh5でも(`coords` + ぼやけスコアのデータセットがあれば)使える。

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
        --wsi_dir data \
        --threshold_percentile 50 \
        --n_per_slide 200 \
        --format webdataset \
        --codec png \
        --seed 42 \
        --n_workers "$SLURM_CPUS_PER_TASK"
EOF

sbatch run_build_dataset.sh
```

経路C(`main.py`)の出力を使う場合は、1つ目の引数を座標h5のディレクトリに読み替える。

```bash
        results/trident/20x_256px_0px_overlap/patches \
        results/dataset \
```

閾値の指定は2通りあり、**どちらか一方のみ**指定する。

| 指定 | 意味 | 使いどころ |
| --- | --- | --- |
| `--threshold 100` | 絶対閾値。スコアがこれ以上のパッチを候補にする | 単一施設・単一スキャナ |
| `--threshold_percentile 50` | スライドごとの分位点。上位(100-p)%を候補にする | 複数施設が混ざる場合 |

スコアの絶対値はスキャナや染色で動くため、素性の違うデータを混ぜるなら分位点の方が安定する。
まず `--threshold_percentile` で通し、`results/dataset/manifest.json` の `per_slide` を見て
スライドごとの採用枚数に極端な偏りが無いか確認するとよい。

経路B・経路Cのh5は画素(`images`)を持たず`coords`のみなので、`--wsi_dir data` が必須。

主なオプションは `README.md` の表を参照。符号化は `png` / `npy` のいずれも可逆で、
JPEGは採用していない（非可逆圧縮は高周波を落とし、ぼやけの評価そのものを壊すため）。

---

## 7. まとめて流す

依存関係を付けて一気に投げる場合。前段が正常終了したときだけ次が走る。

```bash
cd "$PROJ"
JID1=$(sbatch --parsable run_add_blur.sh)          # または run_trident_cpu.sh / run_trident_gpu.sh
JID2=$(sbatch --parsable --dependency=afterok:"$JID1" run_build_dataset.sh)
echo "add_blur=$JID1  build_dataset=$JID2"
```

経路Cは1ジョブでセグメンテーション+座標抽出+ぼやけ付与まで完結するので、後段の
`build_dataset.sh`だけを繋げばよい(手順4(経路B)に相当する部分は`run_trident_*.sh`単体で完了する)。

### スライドが多くて4時間に収まらない場合

`data/` 直下のサブディレクトリ単位でジョブ配列に分ける。
出力構造は保たれるので、あとから `results/h5`(または`results/trident`) 全体を1つとして扱える。
経路Bの`run_add_blur.sh`を例に示す。

```bash
cd "$PROJ"

# 処理単位のリストを作る（data/直下のディレクトリ名）
find data -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort > subsets.txt
wc -l < subsets.txt

cat > run_add_blur_array.sh <<'EOF'
#!/bin/bash
#SBATCH -p large-preproc
#SBATCH -t 4:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -J add_blur_arr
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

SUBSET=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" subsets.txt)
echo "subset: $SUBSET"

apptainer exec --bind "$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR" .env/env.sif \
    .venv/bin/python scripts/add_blur_scores.py \
        "results/h5/$SUBSET" \
        --wsi_dir "data/$SUBSET" \
        --n_workers "$SLURM_CPUS_PER_TASK"
EOF

# 同時実行数は %8 の部分で制限する（I/O帯域に合わせて調整）
sbatch --array=0-$(($(wc -l < subsets.txt) - 1))%8 run_add_blur_array.sh
```

経路Cで同様に分割する場合は、`main.py`呼び出しの`--wsi_dir "data/$SUBSET"`
`--out_dir "results/trident/$SUBSET"`に読み替えれば同じ配列ジョブの形で書ける。

---

## 8. 進捗の確認

```bash
squeue -u "$USER"
tail -f logs/add_blur_*.out
```

経路C(`main.py`)はTRIDENT側の進捗バー(`tqdm`)がそのまま`.err`に出るほか、
`results/trident/summary.md`と`results/trident/wsi_states/<slide>__*.json`にも
スライドごとの状態(成功/スキップ/エラーとその理由)が残る。

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

経路Cの`--extract_patches`で書き出した画像は`results/trident/patches/<slide_id>/*.png`と
`results/trident/patches/patches.csv`(座標・スコアの一覧)にそのまま並んでいる。

---

## 10. つまずきやすいところ

### 10-1. 同名のh5が複数フォルダにある

`slide_id` は通常ファイル名(stem)をそのまま使うが、`results/h5/A/x.h5` と
`results/h5/B/x.h5` のように衝突する場合だけ、`build_dataset.py` が相対パスを繋いだ
`A_x` / `B_x` に置き換える（衝突していないスライドのIDは変わらない）。TRIDENT形式の
`<slide>_patches.h5` は自動で`_patches`を除いた`<slide>`として扱われる。

置き換えが起きると起動時に `Note: 同名のh5が複数フォルダにあります` と対応表が出る。
元のh5は `manifest.json` の `per_slide[slide_id]["path"]` から辿れる。
素のスライド名でIDを揃えたいなら、事前にリネームしてから流すこと。

### 10-2. 計算ノードでファイルが見えない

Apptainerが自動でバインドするのは `$HOME` とカレントディレクトリ程度。
データが別のファイルシステムにあるなら明示的に足す。

```bash
apptainer exec --bind "$PROJ:$PROJ" --bind /mnt/storage:/mnt/storage .env/env.sif ...
```

### 10-3. `uv sync` が計算ノードで失敗する

計算ノードからは外部ネットワークに出られないことが多い。手順2をログインノードで済ませ、
ジョブ内では `.venv/bin/python` を直接呼ぶこと（ジョブスクリプト内で `uv run` を使わない）。
TRIDENTはGitHubから取得する依存パッケージなので、PyPIだけ許可されたプロキシ環境では
これだけ失敗することがある(手順2参照)。

### 10-4. 経路Cでコマンドは通るのにパッチ/特徴量が0件

TRIDENTの`--task`は前段の結果を前提にする(セグメンテーションが無いと座標抽出は
`GeoJSON not found`で何もせずスキップする)。`main.py`は内部で`segment()`→`extract_coords()`の
順に呼んでいるので通常は問題にならないが、`results/trident/`を使い回して個別に
`src.trident_runner.TridentRunner`を直接叩くようなスクリプトを書く場合は、この順序を崩さないこと。

セグメンテーション自体が空になるケースもある。`results/trident/wsi_states/<slide>__*.json`の
`"reason"`を見る。代表的な原因:
- スライドのピラミッドが薄い/単層で、`hest`が組織を検出できない → `--seg_conf_thresh 0.4`を試す、
  またはCPUで動く`--segmenter otsu`に切り替える。
- 生検の小片やTMAなど組織領域が小さい → `--min_tissue_proportion`を下げる。

### 10-5. GPUを指定したのにCPUで動いている(遅い)

`apptainer exec`に`--nv`を付け忘れると、コンテナ内から見えるGPUが無くなり
`torch.cuda.is_available()`がFalseになる。`main.py`はGPUが見えない場合エラーにはせず
自動的に`--segmenter otsu --device cpu`相当にフォールバックするため、静かに遅い経路へ
切り替わって気づきにくい。手順5-2のように`--nv`を必ず付けること。

### 10-6. Hugging Faceのgatedモデルでエラーになる(経路Cで`--patch_encoder`使用時)

`uni_v1`など一部のパッチエンコーダはHugging Face上でアクセス許可が必要なモデル。
`huggingface-cli login`とモデルページでのアクセス申請をログインノード側で先に済ませておくこと
(手順2参照)。計算ノードにはネットワークが無いことが多く、ジョブの中で初めて認証しようとしても失敗する。

### 10-7. ビルド中に `No space left on device`

`APPTAINER_TMPDIR` が小さい `/tmp` を向いている。手順1のとおり容量のある場所に退避させる。
