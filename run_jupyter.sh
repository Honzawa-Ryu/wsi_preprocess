#!/bin/bash
#SBATCH -p large-creator
#SBATCH -J jupyter
#SBATCH -t 4:00:00        # large-creatorパーティションの上限(4時間)に合わせています。必要に応じて調整してください
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -o logs/jupyter_%j.out
#SBATCH -e logs/jupyter_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJ="$SLURM_SUBMIT_DIR"
else
    PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$PROJ"

if [ ! -f .env/env.sif ]; then
    echo "Error: Singularity image .env/env.sif not found. Please run run_build.sh first."
    exit 1
fi

# ジョブIDから決めることで、同時に複数ジョブを走らせてもポートが衝突しにくくする
PORT=$((10000 + SLURM_JOB_ID % 20000))
NODE="$SLURMD_NODENAME"

echo "==================================================================="
echo "Jupyter Lab starting on node: $NODE  port: $PORT"
echo "手元のPCから、ログインノード経由でポートフォワードしてください:"
echo "  ssh -L ${PORT}:${NODE}:${PORT} <あなたのログインノードのSSHホスト名/エイリアス>"
echo "接続後、ブラウザで http://localhost:${PORT} を開いてください"
echo "(アクセス用トークンは下のJupyterログに出力されます)"
echo "==================================================================="

apptainer exec --bind "$PROJ:$PROJ" .env/env.sif \
    .venv/bin/jupyter lab --no-browser --ip=0.0.0.0 --port="$PORT" --notebook-dir="$PROJ"
