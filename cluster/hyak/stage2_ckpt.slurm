#!/bin/bash
#SBATCH --job-name=tqm_stage2
#SBATCH --partition=gpu-l40s
#SBATCH --account=stf
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
set -x

PROJECT_ROOT="${PROJECT_ROOT:-/gscratch/scrubbed/${USER}/theta_quench_magic_lab}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.conda-hyak/bin/python}"

mkdir -p "${PROJECT_ROOT}/slurm_logs"
cd "$PROJECT_ROOT"

TMP_BASE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
JOB_TMP="${TMP_BASE}/${USER}/theta_quench_stage2_${SLURM_JOB_ID}"
mkdir -p "${JOB_TMP}/mpl" "${JOB_TMP}/tmp"

export MPLCONFIGDIR="${JOB_TMP}/mpl"
export TMPDIR="${JOB_TMP}/tmp"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

make PYTHON="$PYTHON_BIN" stage2-universal
