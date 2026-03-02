from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check stage gate flags from effect registry")
    parser.add_argument("--registry", type=str, required=True)
    parser.add_argument("--require-primary-majority", action="store_true")
    parser.add_argument("--require-primary-headline", action="store_true")
    parser.add_argument("--require-entropy-control", action="store_true")
    args = parser.parse_args()

    payload = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    flags = payload.get("gate_flags", {})
    ok = True

    if args.require_primary_majority and not bool(flags.get("primary_supported_majority", False)):
        print("FAIL: primary_supported_majority gate not satisfied")
        ok = False
    if args.require_primary_headline and not bool(flags.get("primary_supported_headline_region", False)):
        print("FAIL: primary_supported_headline_region gate not satisfied")
        ok = False
    if args.require_entropy_control and not bool(flags.get("entropy_control_supported_any", False)):
        print("FAIL: entropy_control_supported_any gate not satisfied")
        ok = False

    if not ok:
        raise SystemExit(2)
    print("stage-gate PASS")


if __name__ == "__main__":
    main()
