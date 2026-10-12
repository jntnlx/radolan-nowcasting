#!/usr/bin/env python
"""
Download RADKLIM-YW warm-season archives and build the lossless frame store.

    python scripts/build_store.py                    # all split years (resumable)
    python scripts/build_store.py --years 2021 2022  # selected seasons
    python scripts/build_store.py --download-only

The raw archives need ~2.8 GB per season; the store ~3 GB per season.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nowcast.config import SEASON_MONTHS, TEST_YEARS, TRAIN_YEARS, VAL_YEARS
from nowcast.radklim import download_archive
from nowcast.store import build


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=int, nargs="+", default=[*TRAIN_YEARS, *VAL_YEARS, *TEST_YEARS])
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--download-only", action="store_true")
    args = p.parse_args()

    for year in args.years:
        for month in SEASON_MONTHS:
            download_archive(year, month)
    if not args.download_only:
        build(args.years, workers=args.workers)


if __name__ == "__main__":
    main()
