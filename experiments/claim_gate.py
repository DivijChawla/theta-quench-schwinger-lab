from __future__ import annotations

import argparse
import re
from pathlib import Path


FORBIDDEN_PATTERNS = [
    r"\buniversal law\b",
    r"\buniversality\b",
    r"\bproves? universal\b",
    r"\bthermodynamic[- ]limit proof\b",
]

SAFE_NEGATIONS = [
    "not universal",
    "non-universal",
    "avoid universal",
]


def _has_boundary_failures(publishability_text: str) -> bool:
    text = publishability_text.lower()
    return ("n=10" in text and "unsupported" in text) or ("regime" in text and "unsupported" in text)


def _contains_forbidden(text: str) -> list[str]:
    lowered = text.lower()
    hits: list[str] = []
    for pat in FORBIDDEN_PATTERNS:
        for m in re.finditer(pat, lowered):
            window = lowered[max(0, m.start() - 24) : m.end() + 24]
            if any(neg in window for neg in SAFE_NEGATIONS):
                continue
            hits.append(m.group(0))
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Claim-gate check for non-overclaiming language")
    parser.add_argument("--publishability", type=str, default="report/publishability_status.md")
    parser.add_argument("--targets", type=str, default="README.md,report/main.tex")
    args = parser.parse_args()

    pub_text = Path(args.publishability).read_text(encoding="utf-8")
    should_gate = _has_boundary_failures(pub_text)
    if not should_gate:
        print("claim-gate: no boundary failures detected; no language restriction enforced")
        return

    bad = []
    for target in [x.strip() for x in args.targets.split(",") if x.strip()]:
        path = Path(target)
        if not path.exists():
            continue
        hits = _contains_forbidden(path.read_text(encoding="utf-8"))
        if hits:
            bad.append((target, hits))

    if bad:
        for target, hits in bad:
            print(f"claim-gate FAIL: {target} contains forbidden universal phrasing: {sorted(set(hits))}")
        raise SystemExit(2)
    print("claim-gate PASS")


if __name__ == "__main__":
    main()
