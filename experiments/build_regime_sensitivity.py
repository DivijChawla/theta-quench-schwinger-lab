from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_meta(specs: list[str]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for spec in specs:
        if "=" in spec:
            arch, path = spec.split("=", 1)
            arch = arch.strip()
            payload = load(path.strip())
            for k, v in payload.get("meta", {}).items():
                if ":" in k:
                    merged[k] = v
                else:
                    merged[f"{arch}:{k}"] = v
        else:
            payload = load(spec)
            for k, v in payload.get("meta", {}).items():
                if ":" in k:
                    merged[k] = v
                else:
                    merged[f"gru:{k}"] = v
    return merged


def parse_labeled_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"Expected labeled spec 'label=path', got: {spec}")
    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label:
        raise ValueError(f"Empty label in spec: {spec}")
    if not path:
        raise ValueError(f"Empty path in spec: {spec}")
    return label, path


def get_meta(meta: dict[str, dict], arch: str, metric: str) -> dict | None:
    key = f"{arch}:{metric}"
    return meta.get(key)


def fmt(x: float) -> str:
    if x is None:
        return "n/a"
    try:
        x = float(x)
    except Exception:
        return str(x)
    if x != x:
        return "nan"
    if x == 0.0:
        return "<1e-300"
    if abs(x) < 1e-3:
        return f"{x:.2e}"
    return f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build regime-sensitivity report for novelty claims")
    ap.add_argument(
        "--baseline",
        action="append",
        default=[],
        help="Baseline spec: 'arch=path/to/summary.json' (repeatable)",
    )
    ap.add_argument(
        "--alt",
        action="append",
        default=[],
        help="Alt-regime spec: 'label=path/to/summary.json' (repeatable)",
    )
    ap.add_argument("--out", type=str, default="report/regime_sensitivity.md")
    args = ap.parse_args()

    base_specs = list(args.baseline) or [
        "gru=outputs/novelty/novelty_summary.json",
        "independent=outputs/novelty_arch_independent/novelty_summary.json",
        "rbm=outputs/novelty_arch_rbm/novelty_summary.json",
    ]
    alt_specs = list(args.alt) or [
        "alt_baseline=outputs/novelty_alt_regime/novelty_summary.json",
        "alt_heavy_mass=outputs/novelty_alt_regime_heavy_mass/novelty_summary.json",
        "alt_strong_coupling=outputs/novelty_alt_regime_strong_coupling/novelty_summary.json",
    ]

    base_meta = merge_meta(base_specs)
    alt_map: dict[str, dict[str, dict]] = {}
    for spec in alt_specs:
        label, path = parse_labeled_spec(spec)
        alt_map[label] = merge_meta([path])

    archs = ["gru", "made", "rbm", "independent"]
    metrics = [
        "pearson_magic_vs_val_nll",
        "pearson_magic_vs_val_nll_partial_entropy",
        "pearson_entropy_vs_val_nll",
    ]

    lines: list[str] = []
    lines.append("# Regime Sensitivity Report")
    lines.append("")
    lines.append("Compares baseline regime vs multiple alternate `(m,g)` regimes for core claims.")
    lines.append("")
    lines.append("| Regime | Architecture | Metric | Baseline Verdict | Baseline r | Alt Verdict | Alt r |")
    lines.append("|---|---|---|---|---:|---|---:|")

    flips: list[str] = []
    flip_count = {arch: 0 for arch in archs}
    total_count = {arch: 0 for arch in archs}
    for reg_label, alt_meta in alt_map.items():
        for arch in archs:
            for metric in metrics:
                b = get_meta(base_meta, arch, metric)
                a = get_meta(alt_meta, arch, metric)
                if not b or not a:
                    continue
                b_v = b.get("verdict", "n/a")
                a_v = a.get("verdict", "n/a")
                lines.append(
                    f"| {reg_label} | {arch} | {metric.replace('pearson_', '').replace('_', ' ')} | {b_v} | {fmt(b.get('pooled_r'))} | {a_v} | {fmt(a.get('pooled_r'))} |"
                )
                if metric == "pearson_magic_vs_val_nll":
                    total_count[arch] += 1
                    if b_v != a_v:
                        flip_count[arch] += 1
                        flips.append(f"{reg_label} | {arch}: `{b_v}` -> `{a_v}`")

    lines.append("")
    lines.append("## Interpretation")
    if flips:
        lines.append("- The `magic -> validation NLL` claim is regime-sensitive:")
        for f in flips:
            lines.append(f"- {f}")
    else:
        lines.append("- No verdict flip detected for `magic -> validation NLL` between regimes.")

    lines.append("")
    lines.append("## Flip Summary (magic -> val nll)")
    lines.append("| Architecture | flips / regimes |")
    lines.append("|---|---:|")
    for arch in archs:
        lines.append(f"| {arch} | {flip_count[arch]} / {total_count[arch]} |")

    lines.append("- Practical claim boundary: treat magic-learnability relation as conditional on parameter regime, not universal.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
