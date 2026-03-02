from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compute budget markdown table")
    parser.add_argument("--glob", type=str, default="outputs/*/*/compute_budget.csv")
    parser.add_argument("--out", type=str, default="report/compute_budget.md")
    args = parser.parse_args()

    files = sorted(Path().glob(args.glob))
    lines = [
        "# Compute Budget",
        "",
        "| Study | Phase | Step | Label | Runtime (s) |",
        "|---|---|---|---|---:|",
    ]

    if not files:
        lines.append("| n/a | n/a | n/a | no compute_budget.csv found | n/a |")
    else:
        for path in files:
            df = pd.read_csv(path)
            parts = path.parts
            study = parts[-3] if len(parts) >= 3 else "unknown"
            phase = parts[-2] if len(parts) >= 2 else "unknown"
            for r in df.to_dict(orient="records"):
                runtime = float(r.get("runtime_seconds", float("nan")))
                lines.append(
                    f"| {study} | {phase} | {r.get('step')} | {r.get('label')} | {runtime:.2f} |"
                )

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
