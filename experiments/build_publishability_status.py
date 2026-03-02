from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_meta(payload: dict, arch_hint: str | None = None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    meta = payload.get("meta", {})
    for key, val in meta.items():
        if ":" in key:
            arch, metric = key.split(":", 1)
        else:
            arch = arch_hint or "unknown"
            metric = key
        out[f"{arch}:{metric}"] = val
    return out


def _fmt(v: float) -> str:
    if not isinstance(v, (int, float)):
        return str(v)
    if v != v:
        return "nan"
    if v == 0.0:
        return "<1e-300"
    if abs(v) < 1e-3 and v != 0.0:
        return f"{v:.2e}"
    return f"{v:.3f}"


def main() -> None:
    p = argparse.ArgumentParser(description="Build consolidated publishability status from robustness outputs")
    p.add_argument(
        "--summary",
        action="append",
        default=[],
        help="Summary spec: either '/path/to/novelty_summary.json' or 'arch=/path/to/novelty_summary.json'",
    )
    p.add_argument("--out", type=str, default="report/publishability_status.md")
    args = p.parse_args()

    summary_specs = list(args.summary)
    if not summary_specs:
        summary_specs = [
            "gru=outputs/novelty/novelty_summary.json",
            "independent=outputs/novelty_arch_independent/novelty_summary.json",
            "rbm=outputs/novelty_arch_rbm/novelty_summary.json",
        ]

    meta = {}
    for spec in summary_specs:
        if "=" in spec:
            arch_hint, path_raw = spec.split("=", 1)
            arch_hint = arch_hint.strip()
            path = Path(path_raw.strip())
        else:
            arch_hint = None
            path = Path(spec.strip())
        payload = _load(path)
        meta.update(_extract_meta(payload, arch_hint=arch_hint))

    architectures = sorted({k.split(":", 1)[0] for k in meta.keys() if ":" in k})

    key_metrics = [
        "pearson_magic_vs_val_nll",
        "pearson_magic_vs_val_nll_partial_entropy",
        "pearson_magic_vs_kl",
        "pearson_magic_vs_tv",
    ]

    lines: list[str] = []
    lines.append("# Consolidated Publishability Status")
    lines.append("")
    lines.append("## Evidence Table")
    lines.append("| Architecture | Metric | Verdict | pooled r | 95% CI | q-value |")
    lines.append("|---|---|---|---:|---|---:|")

    for arch in architectures:
        for metric in key_metrics:
            k = f"{arch}:{metric}"
            v = meta.get(k)
            if not v:
                continue
            lines.append(
                "| "
                + f"{arch} | {metric.replace('pearson_', '').replace('_', ' ')} | {v.get('verdict')} | "
                + f"{_fmt(v.get('pooled_r', float('nan')))} | "
                + f"[{_fmt(v.get('ci_low', float('nan')))}, {_fmt(v.get('ci_high', float('nan')))}] | "
                + f"{_fmt(v.get('combined_q', float('nan')))} |"
            )

    lines.append("")
    lines.append("## Can Claim Now")
    can_claim: list[str] = []
    val_supported_arch = [
        arch for arch in architectures if meta.get(f"{arch}:pearson_magic_vs_val_nll", {}).get("verdict") == "supported"
    ]
    if len(val_supported_arch) >= 2:
        can_claim.append(
            "Across multiple NNQS families, higher magic robustly predicts worse validation NLL in small-N exact Schwinger theta-quenches."
        )

    partial_supported_arch = [
        arch
        for arch in architectures
        if meta.get(f"{arch}:pearson_magic_vs_val_nll_partial_entropy", {}).get("verdict") in {"supported", "mixed"}
    ]
    if len(partial_supported_arch) >= 2:
        can_claim.append(
            "The magic-learnability signal persists after controlling for entanglement entropy (partial-correlation analysis) across architectures."
        )

    if not can_claim:
        can_claim.append("No cross-architecture claim currently passes the support threshold.")

    for c in can_claim:
        lines.append(f"- {c}")

    lines.append("")
    lines.append("## Do Not Claim Yet")
    weak_kl_arch = [
        arch for arch in architectures if meta.get(f"{arch}:pearson_magic_vs_kl", {}).get("verdict") != "supported"
    ]
    if weak_kl_arch:
        lines.append(
            "- Universal KL/TV hardness laws for expressive NNQS "
            f"(non-supported or mixed KL evidence in: {', '.join(weak_kl_arch)})."
        )
    else:
        lines.append("- Universal KL/TV hardness laws across expressive NNQS are still not a large-N claim.")
    lines.append("- Large-N universality (all evidence remains small-N exact regime).")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
