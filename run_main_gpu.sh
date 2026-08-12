#!/bin/bash
#SBATCH -p x-large-creator
#SBATCH -t 12:00:00        # 実際の計算時間に合わせて調整してください
#SBATCH -c 8               # ワーカー数に合わせて調整（例: 8コア）
#SBATCH --mem=32G          # 必要メモリ量に合わせて調整
#SBATCH --gpus=1           # ※お使いのクラスタの指定形式に応じて --gres=gpu:1 等に変更してください
#SBATCH -o logs/gpu_%j.out
#SBATCH -e logs/gpu_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJ="$SLURM_SUBMIT_DIR"
else
    PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$PROJ"

if [ ! -f .env/env.sif ]; then
    echo "Error: Singularity image .env/env.sif not found. Please build it first."
    exit 1
fi


# 染色正規化の目標パッチ画像。既定では data/baseline に置いた基準パッチを使う。
# 別の画像を使いたい場合は明示的に指定する。例:
#   STAIN_TARGET=/path/to/target_patch.jpg sbatch run_main_gpu.sh
BASELINE_DIR="$PROJ/data/baseline"
STAIN_TARGET="${STAIN_TARGET:-}"
if [ -z "$STAIN_TARGET" ]; then
    # 対応拡張子はTRIDENTのWSI探索と揃える必要はない(こちらは通常の画像1枚)ので
    # png/jpg/jpegのみ見る。複数あればソート順で先頭を採用する。
    STAIN_TARGET="$(find "$BASELINE_DIR" -maxdepth 1 -type f \
        \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | sort | head -n1)"
fi

if [ -z "$STAIN_TARGET" ] || [ ! -f "$STAIN_TARGET" ]; then
    echo "Error: 染色正規化の目標パッチが見つかりません($BASELINE_DIR)。" \
         "STAIN_TARGET=/path/to/target.png で明示的に指定してください。"
    exit 1
fi
echo "Stain normalization target: $STAIN_TARGET"

# ---- GPUノードでの本計算実行 ----
apptainer exec --nv --bind "$PROJ:$PROJ" .env/env.sif \
    .venv/bin/python main.py \
        --wsi_dir data/raw_slide \
        --out_dir results/trident \
        --mag 20 --patch_size 256 \
        --calc_blur \
        --patch_encoder uni_v2 \
        --stain_normalize_target "$STAIN_TARGET" \
        --blur_n_workers "${SLURM_CPUS_PER_TASK:-1}"