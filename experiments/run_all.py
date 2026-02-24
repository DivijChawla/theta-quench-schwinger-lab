from __future__ import annotations

import argparse

from tqm.cli import run_all
from tqm.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full theta-quench + magic + NNQS pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_all(cfg)


if __name__ == "__main__":
    main()
