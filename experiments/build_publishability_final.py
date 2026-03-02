from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


METRICS = [
    "pearson_magic_vs_val_nll",
    "pearson_magic_vs_val_nll_partial_entropy",
    "pearson_entropy_vs_val_nll",
    "pearson_magic_vs_kl",
    "pearson_magic_vs_tv",
]


def _load_json(path: Path) -> dict:
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


def _parse_spec(spec: str) -> tuple[str | None, Path]:
    if "=" in spec:
        left, right = spec.split("=", 1)
        return left.strip(), Path(right.strip())
    return None, Path(spec.strip())


def _fmt(x: float | int | None) -> str:
    if x is None:
        return "n/a"
    try:
        y = float(x)
    except Exception:
        return str(x)
    if y != y:
        return "nan"
    if y == 0.0:
        return "<1e-300"
    if abs(y) < 1e-3:
        return f"{y:.2e}"
    return f"{y:.3f}"


def _metric_label(metric: str) -> str:
    return metric.replace("pearson_", "").replace("_", " ")


def _get(meta: Dict[str, dict], arch: str, metric: str) -> dict | None:
    return meta.get(f"{arch}:{metric}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build consolidated publishability report")
    ap.add_argument("--baseline", action="append", default=[])
    ap.add_argument("--seed", type=str, default="outputs/novelty_gru_seeds8_n6_fulltheta/novelty_summary.json")
    ap.add_argument("--n10", type=str, default="outputs/novelty_n10_exact_power/novelty_summary.json")
    ap.add_argument(
        "--regime",
        action="append",
        default=[],
        help="Regime spec: label=path/to/novelty_summary.json",
    )
    ap.add_argument("--out", type=str, default="report/publishability_status.md")
    args = ap.parse_args()

    baseline_specs = args.baseline or [
        "gru=outputs/novelty/novelty_summary.json",
        "independent=outputs/novelty_arch_independent/novelty_summary.json",
        "rbm=outputs/novelty_arch_rbm/novelty_summary.json",
    ]
    regime_specs = args.regime or [
        "alt_baseline=outputs/novelty_alt_regime/novelty_summary.json",
        "alt_heavy_mass=outputs/novelty_alt_regime_heavy_mass/novelty_summary.json",
        "alt_strong_coupling=outputs/novelty_alt_regime_strong_coupling/novelty_summary.json",
    ]

    base_meta: dict[str, dict] = {}
    for spec in baseline_specs:
        arch_hint, path = _parse_spec(spec)
        base_meta.update(_extract_meta(_load_json(path), arch_hint=arch_hint))

    seed_payload = _load_json(Path(args.seed))
    seed_meta = _extract_meta(seed_payload, arch_hint="gru")
    n10_meta = _extract_meta(_load_json(Path(args.n10)))

    regime_meta: dict[str, dict[str, dict]] = {}
    for spec in regime_specs:
        label, path = _parse_spec(spec)
        if label is None:
            label = path.parent.name
        regime_meta[label] = _extract_meta(_load_json(path))

    archs = sorted({k.split(":", 1)[0] for k in base_meta.keys()})
    lines: list[str] = []
    lines.append("# Consolidated Publishability Status (Final)")
    lines.append("")
    lines.append("## Baseline Cross-Architecture Evidence")
    lines.append("| Architecture | Metric | Verdict | pooled r | 95% CI | q-value |")
    lines.append("|---|---|---|---:|---|---:|")
    for arch in archs:
        for m in METRICS:
            v = _get(base_meta, arch, m)
            if not v:
                continue
            lines.append(
                "| "
                + f"{arch} | {_metric_label(m)} | {v.get('verdict')} | "
                + f"{_fmt(v.get('pooled_r'))} | "
                + f"[{_fmt(v.get('ci_low'))}, {_fmt(v.get('ci_high'))}] | "
                + f"{_fmt(v.get('combined_q'))} |"
            )

    lines.append("")
    lines.append("## Robustness Upgrades")
    seed_sites = seed_payload.get("n_sites", [])
    seed_theta = seed_payload.get("theta1_values", [])
    seed_count = len(seed_payload.get("seeds", []))
    seed_sites_label = ",".join(str(x) for x in seed_sites) if seed_sites else "n/a"
    seed_theta_label = "full theta grid" if len(seed_theta) >= 4 else f"{len(seed_theta)} theta points"
    lines.append(
        f"### Seed Robustness (N={seed_sites_label}, {seed_theta_label}, {seed_count} seeds, GRU)"
    )
    for m in ["pearson_magic_vs_val_nll", "pearson_magic_vs_val_nll_partial_entropy"]:
        v = seed_meta.get(f"gru:{m}")
        if v:
            lines.append(
                f"- `{_metric_label(m)}`: {v.get('verdict')} | r={_fmt(v.get('pooled_r'))} | q={_fmt(v.get('combined_q'))}"
            )

    lines.append("### N=10 Exact-Magic Stress (power run)")
    for arch in ["gru", "made", "rbm", "independent"]:
        v = n10_meta.get(f"{arch}:pearson_magic_vs_val_nll")
        if v:
            lines.append(
                f"- `{arch}` magic vs val nll: {v.get('verdict')} | r={_fmt(v.get('pooled_r'))} | q={_fmt(v.get('combined_q'))}"
            )

    lines.append("")
    lines.append("## Regime Boundary (magic -> val nll)")
    lines.append("| Architecture | Regime | Verdict | pooled r | q-value |")
    lines.append("|---|---|---|---:|---:|")

    arch_gate = ["gru", "made", "rbm", "independent"]
    regime_counts: dict[str, dict[str, int]] = {
        a: {"supported": 0, "mixed": 0, "unsupported": 0, "other": 0} for a in arch_gate
    }

    for reg_label, meta in regime_meta.items():
        for arch in arch_gate:
            v = meta.get(f"{arch}:pearson_magic_vs_val_nll")
            if not v:
                continue
            verdict = str(v.get("verdict", "other"))
            if verdict not in regime_counts[arch]:
                verdict = "other"
            regime_counts[arch][verdict] += 1
            lines.append(
                f"| {arch} | {reg_label} | {v.get('verdict')} | {_fmt(v.get('pooled_r'))} | {_fmt(v.get('combined_q'))} |"
            )

    lines.append("")
    lines.append("### Regime Verdict Counts")
    lines.append("| Architecture | supported | mixed | unsupported |")
    lines.append("|---|---:|---:|---:|")
    for arch in arch_gate:
        c = regime_counts[arch]
        lines.append(f"| {arch} | {c['supported']} | {c['mixed']} | {c['unsupported']} |")

    base_supported = sum(
        1
        for arch in arch_gate
        if _get(base_meta, arch, "pearson_magic_vs_val_nll")
        and _get(base_meta, arch, "pearson_magic_vs_val_nll").get("verdict") == "supported"
    )
    seed_supported = (
        seed_meta.get("gru:pearson_magic_vs_val_nll", {}).get("verdict") == "supported"
    )
    n10_supported = sum(
        1
        for arch in arch_gate
        if n10_meta.get(f"{arch}:pearson_magic_vs_val_nll", {}).get("verdict") == "supported"
    )

    lines.append("")
    lines.append("## Claim Gate")
    if base_supported >= 2 and seed_supported and n10_supported == 0:
        lines.append(
            "- **Publishable scoped claim**: strong small-N, architecture-robust evidence for `magic -> validation NLL`, with explicit failure to generalize to current N=10 setting and alternate regimes."
        )
        lines.append(
            "- Correct framing: `magic-learnability coupling is conditional (regime- and size-dependent), not universal`."
        )
    else:
        lines.append(
            "- Cross-checks are incomplete for a scoped publishable claim; avoid strong novelty statements until baseline/seed/N10 gates are all satisfied."
        )

    lines.append("- Do not claim universal hardness laws from current data.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
