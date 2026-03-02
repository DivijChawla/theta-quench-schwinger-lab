from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ALL_ARCH_FORBIDDEN = [
    r"\ball-architecture universal law\b",
    r"\bmodel-class-universal law\b",
    r"\buniversal across all architectures\b",
]

EXPRESSIVE_FORBIDDEN_IF_FAIL = [
    r"\bexpressive universal law\b",
    r"\bexpressive-model universal law\b",
]

SAFE_NEGATIONS = [
    "not a",
    "not",
    "unsupported",
    "fails",
]


def _contains_any(text: str, patterns: list[str]) -> list[str]:
    lowered = text.lower()
    hits: list[str] = []
    for pat in patterns:
        for m in re.finditer(pat, lowered):
            window = lowered[max(0, m.start() - 24) : m.end() + 24]
            if any(neg in window for neg in SAFE_NEGATIONS):
                continue
            hits.append(m.group(0))
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2 claim gate for universal-law wording")
    parser.add_argument(
        "--registry",
        type=str,
        default="outputs/stage2_prlx/stage2/effect_registry.json",
    )
    parser.add_argument("--targets", type=str, default="README.md,report/main.tex,report/stage2_universal_law.md")
    args = parser.parse_args()

    payload = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    gates = payload.get("gate_flags", {})
    all_arch_ok = bool(gates.get("all_arch_universal_corridor_claim", False) and gates.get("all_arch_quantile_robust", False))
    expressive_ok = bool(
        gates.get("expressive_universal_corridor_claim", False) and gates.get("expressive_quantile_robust", False)
    )

    violations: list[str] = []
    for target in [x.strip() for x in args.targets.split(",") if x.strip()]:
        p = Path(target)
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8")
        if not all_arch_ok:
            hits = _contains_any(txt, ALL_ARCH_FORBIDDEN)
            if hits:
                violations.append(f"{target}: forbidden all-architecture wording {sorted(set(hits))}")
        if not expressive_ok:
            hits = _contains_any(txt, EXPRESSIVE_FORBIDDEN_IF_FAIL)
            if hits:
                violations.append(f"{target}: forbidden expressive wording {sorted(set(hits))}")

    if violations:
        for v in violations:
            print(f"stage2-claim-gate FAIL: {v}")
        raise SystemExit(2)
    print("stage2-claim-gate PASS")


if __name__ == "__main__":
    main()
