#!/usr/bin/env python
"""
Visual gate: do individual members look like radar fields?

    python scripts/member_panel.py --checkpoint runs/<run>/best.pt [--out DIR]

For five fixed validation cases (the July 2021 flood case, the case on which the network is
worst and the one on which it is best relative to STEPS, a convective case and a median one)
and +30 / +60 / +120 min, one figure per case shows the observation, three members of the
network, one member of the displacement ensemble of the analysis with the training flow (the
untrained network; evaluate.py) and, if cached
in runs/panel_cache/, two STEPS members. Scores cannot replace this look: an ensemble can have
good marginal scores with members that no forecaster would accept as rain fields.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nowcast.config import RUNS_DIR, time_index  # noqa: E402
from nowcast.evaluate import DisplacementForecaster, NetworkForecaster, load_case  # noqa: E402

CASES = ("2021-07-14T14", "2022-10-02T17", "2021-07-24T05", "2021-06-29T15", "2022-06-20T07")
LEADS = (6, 12, 24)
WINDOW = 448  # km
LEVELS = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 1000]  # mm/h
CMAP = ListedColormap(matplotlib.colormaps["YlGnBu"](np.linspace(0.12, 1.0, len(LEVELS) - 1)))
NORM = BoundaryNorm(LEVELS, CMAP.N)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None, help="directory (default: the run directory)")
    args = p.parse_args()
    out = args.out or args.checkpoint.parent
    out.mkdir(parents=True, exist_ok=True)
    name = args.checkpoint.parent.name
    network, displaced = NetworkForecaster(args.checkpoint, 5), DisplacementForecaster(5, smoothness=1.0)

    for case in CASES:
        year, t0 = time_index(datetime.fromisoformat(case + ":00"))
        history, obs, context = load_case(year, t0, network.history)
        lead = [k - 1 for k in LEADS]
        # the window with the most observed rain at +60 min
        tiles = F.avg_pool2d(torch.from_numpy(np.nan_to_num(obs[11]))[None, None], WINDOW, stride=32)[0, 0]
        r, c = (32 * int(i) for i in np.unravel_index(int(tiles.argmax()), tiles.shape))
        cut = np.s_[..., r:r + WINDOW, c:c + WINDOW]
        members = network(history, context, seed=t0)[:3, lead].cpu().numpy()[cut]
        columns = [("observed", obs[lead][cut])] + [(f"{name}, member {i + 1}", m) for i, m in enumerate(members)]
        columns.append(("untrained (displaced analysis), member 1", displaced(history[-6:], context, seed=t0)[0, lead].cpu().numpy()[cut]))
        cache = RUNS_DIR / "panel_cache" / f"steps_{case}.npy"
        if cache.exists():
            columns += [(f"STEPS, member {i + 1}", m[cut]) for i, m in enumerate(np.load(cache).astype(np.float32))]

        fig, axes = plt.subplots(len(LEADS), len(columns), figsize=(3.1 * len(columns), 3.2 * len(LEADS)))
        for j, (title, fields) in enumerate(columns):
            for k, minutes in enumerate(5 * np.array(LEADS)):
                ax = axes[k, j]
                ax.imshow(np.ma.masked_less(np.nan_to_num(fields[k]), LEVELS[0]), origin="lower", cmap=CMAP, norm=NORM, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if k == 0:
                    ax.set_title(title, fontsize=9)
                if j == 0:
                    ax.set_ylabel(f"+{minutes} min")
        fig.suptitle(f"{case}:00 UTC, {WINDOW} km window; rain rate classes 0.1 / 0.5 / 1 / 2 / 5 / 10 / 20 / 50 mm/h", fontsize=10)
        fig.tight_layout()
        path = out / f"members_{name}_{case}.png"
        fig.savefig(path, dpi=75)
        plt.close(fig)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
