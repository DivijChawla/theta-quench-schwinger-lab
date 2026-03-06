#!/bin/bash
#SBATCH --job-name=tqm_setup_env
#SBATCH --partition=compute
#SBATCH --account=stf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
set -x

PROJECT_ROOT="${PROJECT_ROOT:-/gscratch/scrubbed/${USER}/theta_quench_magic_lab}"

mkdir -p "${PROJECT_ROOT}/slurm_logs"
cd "$PROJECT_ROOT"

TMP_BASE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
JOB_TMP="${TMP_BASE}/${USER}/theta_quench_setup_${SLURM_JOB_ID}"
mkdir -p "${JOB_TMP}/tmp"

export TMPDIR="${JOB_TMP}/tmp"
export PYTHONUNBUFFERED=1

bash cluster/hyak/bootstrap_env.sh
