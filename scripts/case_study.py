#!/usr/bin/env python
"""
Case-study figure: observed rain against the network ensemble, STEPS and extrapolation.

    python scripts/case_study.py --time 2021-07-14T15:00 --checkpoint runs/<run>/best.pt [--steps]
"""

import argparse
import contextlib
import io
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nowcast import baselines, plots
from nowcast.config import time_index
from nowcast.evaluate import LEADS, NetworkForecaster, load_case


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--time", required=True, help="analysis time (UTC), e.g. 2021-07-14T15:00")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--steps", action="store_true", help="include STEPS (several minutes on CPU)")
    p.add_argument("--members", type=int, default=20)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    year, t0 = time_index(datetime.fromisoformat(args.time))
    history, obs, context = load_case(year, t0)
    network = NetworkForecaster(args.checkpoint, args.members)
    forecasts = {"network": network(history, context, seed=t0).cpu().numpy()}
    with contextlib.redirect_stdout(io.StringIO()):
        if args.steps:
            forecasts["STEPS"] = baselines.steps(history, LEADS, members=args.members, seed=t0).astype("float32")
        forecasts["extrapolation"] = baselines.extrapolation(history, LEADS)
    out = args.out or Path(args.checkpoint).parent / f"case_{args.time.replace(':', '')}.png"
    plots.case_figure(args.time.replace("T", " "), history, obs, forecasts, path=out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
