#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gscratch/scrubbed/${USER}/theta_quench_magic_lab}"
CONDA_BIN="${CONDA_BIN:-${HOME}/miniconda3/bin/conda}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-${HOME}/.conda-envs/theta-quench-magic-lab}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv-hyak}"

cd "$PROJECT_ROOT"

if [ -x "$CONDA_BIN" ]; then
  mkdir -p "$(dirname "$CONDA_ENV_DIR")"
  "$CONDA_BIN" create -y -p "$CONDA_ENV_DIR" python=3.11 pip
  "$CONDA_ENV_DIR/bin/python" -m pip install --upgrade pip
  "$CONDA_ENV_DIR/bin/python" -m pip install -r requirements.txt
  "$CONDA_ENV_DIR/bin/python" -m pip install -e .
  echo "Hyak conda environment ready at $CONDA_ENV_DIR"
  exit 0
fi

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

echo "Hyak environment ready at $VENV_DIR"
