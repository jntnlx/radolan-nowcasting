#!/usr/bin/env python
"""
Build the strict-v2 processed RADOLAN-YW cache.

Cache contains memory-mapped numpy arrays of (12, 128, 128)
log-rain patches:
  - train_event:  6 years * "warm season" months, event-filtered
  - val_event:    1 year * "warm season" months, event-filtered
  - val_all:      1 year * "warm season" months, reservoir-sampled
  - test_all:     1 year * "warm season" months, reservoir-sampled

Usage:
  # Full cache build:
  python scripts/build_cache.py

  # Quick test (1 day of frames, representative splits capped at 64):
  python scripts/build_cache.py --test-mode

  # Audit RADOLAN-YW units without building cache:
  python scripts/build_cache.py --audit-only
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory so `radolan_nowcasting` is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radolan_nowcasting.config import PROCESSED_CACHE_DIR, RAW_CACHE_DIR, TRAIN_YEARS, WARM_SEASON_MONTHS, VAL_YEAR, TEST_YEAR
from radolan_nowcasting.data import audit_radolan_units, build_strict_cache


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-mode", action="store_true",
                    help="Build a tiny cache from 1 day for smoke testing.")
    p.add_argument("--skip-download", action="store_true",
                    help="Only use already-downloaded archives.")
    p.add_argument("--audit-only", action="store_true",
                    help="Run unit audit without building cache.")
    p.add_argument("--output-dir", type=Path, default=PROCESSED_CACHE_DIR)
    p.add_argument("--raw-dir", type=Path, default=RAW_CACHE_DIR)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.audit_only:
        print("Running RADOLAN-YW unit audit...")
        audit_months = [
            (TRAIN_YEARS[0], WARM_SEASON_MONTHS[0]),
            (VAL_YEAR, WARM_SEASON_MONTHS[0]),
            (TEST_YEAR, WARM_SEASON_MONTHS[0]),
        ]
        report = audit_radolan_units(
            audit_months,
            raw_dir=args.raw_dir,
            skip_download=args.skip_download,
        )
        out = args.output_dir / "unit_audit.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Unit audit saved: {out}")
        ok = report["summary"]["all_expected_yw_depth_300s"]
        print(f"All expected YW depth/300s: {ok}")
        return 0 if ok else 1

    print(f"Building strict-v2 cache → {args.output_dir}")
    print(f"  test_mode={args.test_mode}")
    result = build_strict_cache(
        output_dir=args.output_dir,
        raw_dir=args.raw_dir,
        skip_download=args.skip_download,
        test_mode=args.test_mode,
    )

    for name, info in result["splits"].items():
        print(f"  {name}: {info['samples']} samples")
    print(f"Stats: {result['stats_path']}")
    print(f"Metadata: {result['metadata_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
