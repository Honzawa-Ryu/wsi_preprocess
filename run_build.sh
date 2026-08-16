#!/bin/bash
#SBATCH -p large-creator
#SBATCH -J build_env
#SBATCH -t 3:00:00
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -o logs/build_%j.out
#SBATCH -e logs/build_%j.err

set -euo pipefail

STRAIGHT_RUN=${STRAIGHT_RUN:-false}

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJ="$SLURM_SUBMIT_DIR"
else
    PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$PROJ"

mkdir -p logs results

# ---- 1. .sif の作成(無ければ) ----
if [ -f .env/env.sif ]; then
    echo "[skip] .env/env.sif は既に存在します"
else
    echo "[build] .env/env.sif を作成します"
    SCRATCH_BUILD_DIR="/scratch/$USER/wsi_preprocess_build/${SLURM_JOB_ID:-$$}"
    export APPTAINER_TMPDIR="$SCRATCH_BUILD_DIR/tmp"
    export APPTAINER_CACHEDIR="$SCRATCH_BUILD_DIR/cache"
    mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
    apptainer build --fakeroot .env/env.sif .env/env.def
    rm -rf "$SCRATCH_BUILD_DIR"
fi

# ---- 2. uv syncでPython環境を準備 ----
echo "[uv sync] .venv を同期します"
apptainer exec --bind "$PROJ:$PROJ" .env/env.sif uv sync

# ---- 3. GPU実行ジョブの投入 ----
if [ "$STRAIGHT_RUN" = "true" ]; then
    echo "[info] STRAIGHT_RUN が有効なため、run_main_gpu.sh の投入を試みます"
    
    if [ -f run_main_gpu.sh ]; then
        echo "[sbatch] main.py 実行ジョブを投入します"
        sbatch run_main_gpu.sh
    else
        echo "[warn] run_main_gpu.sh が存在しないため、ジョブ投入をスキップします"
    fi
else
    echo "[info] STRAIGHT_RUN が無効なため、run_main_gpu.sh の投入を行いません"
fi