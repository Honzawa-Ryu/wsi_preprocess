WSI前処理ライブラリの実装計画書を作成しました。方針に基づき、TRIDENTをコアとしつつ、独自処理を透過的に実行できるラッパー型の構成としています。

---

# WSI前処理パイプライン 実装計画書

## 1. 目的と要件

グループ内におけるWSI（Whole Slide Image）の前処理手順を統一し、再現性の確保と作業コストの削減を図る。

* **コアエンジン:** TRIDENTを利用したセグメンテーション、座標抽出、特徴量抽出。
* **独自要件1（ぼやけスコア付与）:** 各パッチのぼやけ度合いを定量化し、メタデータとして付与する（フィルタリングは後段の任意処理とする）。
* **独自要件2（学習用パッチ抽出）:** 指定した条件・枚数に基づき、学習用のパッチ画像を物理的またはデータセットとして抽出・構築する。
* **利用形態:** ユーザーは内部構造を意識せず、単一のコマンドまたはAPIで一連の処理を実行できること。

## 2. システム構成とディレクトリ構造

> **実装メモ:** 当初はGit SubmoduleでTRIDENTを取り込む案だったが、TRIDENTはPyPIには無いものの
> 素のPythonパッケージとして`pip`/`uv`から直接installできる（`pyproject.toml`の
> `[tool.uv.sources]`でGitHubのコミットを指定するだけでよい）ため、リポジトリにコードを
> 同梱するSubmodule化は不要と判断し取りやめた。`uv sync`で依存関係の一つとして解決される。
> また独自モジュール名も`wsi_pipeline`ではなく、実装時の既存構成に合わせて`src/`を使っている。
> なお、TRIDENTを使わない独自Otsu実装(`wsi_processer.py`, 経路A)は、TRIDENT本体の
> `--segmenter otsu`(CPU専用のクラシックなOtsu実装)と機能が完全に重複していたため削除した。
> 生WSIから直接切り出したい場合は `main.py --segmenter otsu --device cpu` を使う。

TRIDENTは通常の依存パッケージとしてinstallし、その上に自作のPythonモジュール（`src/`）を被せる構成とします。

```text
wsi_preprocess/
├── pyproject.toml              # dependencies に trident を追加、[tool.uv.sources] でGitHubを指定
├── main.py                     # パイプライン統括・CLIエントリーポイント (run_pipeline)
├── src/                        # 独自ライブラリのコアディレクトリ
│   ├── __init__.py
│   ├── trident_runner.py       # TRIDENTのPython API(Processor)呼び出し用ラッパー
│   ├── blur.py                 # ぼやけスコア算出関数
│   ├── patch_extractor.py      # パッチ抽出モジュール
│   ├── dataset_builder.py      # WebDataset/LMDB構築 (scripts/build_dataset.py が使う)
│   └── patch_source.py         # h5への統一アクセス層 (TRIDENT形式 / 画素埋め込み形式の両対応)
├── scripts/                    # 個別ステップを単体で叩くためのCLI
└── README.md

```

## 3. 処理パイプラインの詳細

パイプライン実行時、以下のステップを順次処理します。

### Step 1: 組織セグメンテーションとパッチ座標抽出 (TRIDENT)

* **処理:** `trident_runner.py` からTRIDENTの処理を呼び出し、WSIから背景を除去したパッチ座標（`.h5`ファイル）を生成。
* **出力:** 座標情報が格納された `coords.h5`

### Step 2: ぼやけスコアの算出と付与 (独自処理)

* **処理:** `blur_scorer.py` が `coords.h5` の座標リストを読み込む。OpenSlide等で該当座標の画像をオンメモリで展開し、OpenCVを用いてぼやけスコア（例：ラプラシアンの分散値など）を計算する。
* **出力:** 計算結果を元の `coords.h5` 内の新しいデータセット（例: `blur_score`）として書き込むか、同一階層に `coords_meta.csv` として出力する。

### Step 3: 特徴量抽出 (TRIDENT)

* **処理:** スコア付与後の座標ファイルを指定し、TRIDENTの推論用スクリプトを呼び出して特徴量を抽出。
* **出力:** 特徴量が格納された `.pt` ファイル。

### Step 4: 学習用パッチの抽出 (独自処理)

* **処理:** `patch_extractor.py` を使用し、指定されたサンプリング数（`N枚`）のパッチをランダム、またはスコア等の条件に基づいて抽出する。
* **出力:** 指定ディレクトリに画像ファイル（`.png` または `.jpg`）として保存、あるいは抽出結果をまとめたCSV/H5ファイルを出力する。

## 4. インターフェース設計案

CLIおよびPython APIの両方から実行可能な設計とします。

> **実装メモ:** `wsi_pipeline`パッケージではなく、リポジトリ直下の`main.py`を
> エントリーポイントにしている(下記コマンドは実装済み、そのまま動く)。

**CLI（コマンドライン）での実行イメージ:**

```bash
python main.py \
  --wsi_dir ./data/wsis \
  --out_dir ./data/processed \
  --calc_blur \
  --extract_patches 1000

```

**Pythonコードでの実行イメージ:**

```python
from main import run_pipeline

run_pipeline(
    wsi_dir="./data/wsis",
    out_dir="./data/processed",
    calc_blur=True,           # ぼやけスコアを計算・付与
    extract_patches=1000      # 学習用に各スライドから1000枚パッチを抽出
)

```

## 5. 開発フェーズ

* **フェーズ1: 基礎環境構築**
* リポジトリ作成、TRIDENTを依存パッケージとして`pyproject.toml`に追加、環境構築（`uv sync`で完結）。
* ラッパーの基本構造（TRIDENTをPythonからキックできる状態）の作成。


* **フェーズ2: 独自機能の組み込み**
* H5ファイル読み書き処理の実装。
* 画像切り出しおよびぼやけスコア計算ロジック（`blur_scorer.py`）の実装。


* **フェーズ3: 抽出機能とパイプライン統合**
* 指定枚数のパッチ画像保存機能（`patch_extractor.py`）の実装。
* すべてのステップを直列で実行する `main.py` の完成。


* **フェーズ4: テストと運用ドキュメント作成**
* 少数のWSIを用いたエンドツーエンドの動作確認。
* グループメンバー向けのREADMEおよび利用手順書の作成。



## 6. 実装上の注意点

* **I/Oのボトルネック回避:** ぼやけスコアの計算時はパッチ画像をメモリに読み込むため、並列処理（`multiprocessing`等）を導入して計算時間を短縮することを検討する。
* **ストレージ容量:** パッチ画像そのものを抽出（Step 4）する場合、データ容量が肥大化しやすいため、デフォルトでは画像を保存せず座標とスコアのリストのみを生成するモードを用意しておくと安全。

---