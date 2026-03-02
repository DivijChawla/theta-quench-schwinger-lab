from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str((Path.cwd() / ".mplconfig").resolve())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_meta(payload: dict, arch_hint: str | None = None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key, val in payload.get("meta", {}).items():
        if ":" in key:
            out[key] = val
        else:
            arch = arch_hint or "unknown"
            out[f"{arch}:{key}"] = val
    return out


def _verdict_char(verdict: str) -> str:
    mapping = {"supported": "S", "mixed": "M", "unsupported": "U", "insufficient": "I"}
    return mapping.get(verdict, "?")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build regime/size boundary heatmap for magic->valNLL")
    ap.add_argument("--out", type=str, default="outputs/figs/regime_boundary_magic_valnll.png")
    ap.add_argument("--baseline", type=str, default="outputs/novelty/novelty_summary.json")
    ap.add_argument("--alt-baseline", type=str, default="outputs/novelty_alt_regime/novelty_summary.json")
    ap.add_argument(
        "--alt-heavy-mass",
        type=str,
        default="outputs/novelty_alt_regime_heavy_mass/novelty_summary.json",
    )
    ap.add_argument(
        "--alt-strong-coupling",
        type=str,
        default="outputs/novelty_alt_regime_strong_coupling/novelty_summary.json",
    )
    ap.add_argument("--n10", type=str, default="outputs/novelty_n10_exact_power/novelty_summary.json")
    args = ap.parse_args()

    metric = "pearson_magic_vs_val_nll"

    regimes = [
        ("baseline", _extract_meta(_load(Path(args.baseline)))),
        ("alt_baseline", _extract_meta(_load(Path(args.alt_baseline)))),
        ("alt_heavy_mass", _extract_meta(_load(Path(args.alt_heavy_mass)))),
        ("alt_strong_coupling", _extract_meta(_load(Path(args.alt_strong_coupling)))),
        ("N10_power", _extract_meta(_load(Path(args.n10)))),
    ]

    arch_seen: set[str] = set()
    for _, meta in regimes:
        for key in meta.keys():
            if ":" in key:
                arch_seen.add(key.split(":", 1)[0])
    preferred = ["gru", "made", "rbm", "independent"]
    archs = [a for a in preferred if a in arch_seen] + sorted(a for a in arch_seen if a not in preferred)

    vals = np.full((len(archs), len(regimes)), np.nan, dtype=float)
    labels = [["" for _ in regimes] for _ in archs]

    for j, (_, meta) in enumerate(regimes):
        for i, arch in enumerate(archs):
            key = f"{arch}:{metric}"
            rec = meta.get(key)
            if rec is None:
                continue
            vals[i, j] = float(rec.get("pooled_r", np.nan))
            labels[i][j] = _verdict_char(str(rec.get("verdict", "")))

    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(len(regimes)))
    ax.set_yticks(np.arange(len(archs)))
    ax.set_xticklabels([r for (r, _) in regimes], rotation=20, ha="right")
    ax.set_yticklabels(archs)
    ax.set_title("Boundary map: magic -> val NLL (pooled r)")

    for i in range(len(archs)):
        for j in range(len(regimes)):
            text = labels[i][j]
            if np.isfinite(vals[i, j]):
                text = f"{text}\n{vals[i, j]:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("pooled Pearson r")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
