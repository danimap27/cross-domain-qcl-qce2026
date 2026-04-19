#!/bin/bash
# slurm_ablation.sh — Lanza el estudio de ablación secuencialmente en un único nodo
#
# Para lanzar: sbatch core/slurm_ablation.sh

#SBATCH --job-name=qcl_ablation
#SBATCH --partition=standard
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# 1. Conda activation
HERCULES_CONDA="/lustre/software/easybuild/common/software/Miniconda3/4.9.2"
CONDA_ENV="${CONDA_ENV:-qcl}"

if [ -f "$HERCULES_CONDA/etc/profile.d/conda.sh" ]; then
    source "$HERCULES_CONDA/etc/profile.d/conda.sh"
else
    # Fallback
    CONDA_BASE_AUTO=$(conda info --base 2>/dev/null)
    CONDA_BASE_FALLBACK="${CONDA_BASE_AUTO:-$HOME/miniconda3}"
    if [ -f "$CONDA_BASE_FALLBACK/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE_FALLBACK/etc/profile.d/conda.sh"
    fi
fi

source activate "$CONDA_ENV" || { echo "[ERROR] Cannot activate env: $CONDA_ENV"; exit 1; }

# 2. Execution
cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs

echo "============================================================"
echo "  Desplegando Estudio de Ablación Completo QCL en Hércules"
echo "  Date: $(date)"
echo "  Node: $SLURMD_NODENAME"
echo "============================================================"

# Evitar colisión de buffers
python -u ablation_study.py

echo "============================================================"
echo "  Finish: $(date)"
echo "============================================================"
