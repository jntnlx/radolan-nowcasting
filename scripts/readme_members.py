#!/usr/bin/env python
"""
README figure: one ensemble member of two model versions next to the observation and a STEPS member.

    python scripts/readme_members.py --before runs/<old>/best.pt --after runs/<new>/best.pt

Shows what changed when the ensemble spread moved from local noise fields to random displacement of
the observed field. The member shown is the first one drawn (seed = analysis time) for each model,
not a selection; the STEPS member is the first of the cached ensemble written for member_panel.py
(runs/panel_cache/steps_<case>.npy). The window is the 448 km square with the most observed rain at
+60 min, as in member_panel.py. Default case: 2021-06-29 15:00 UTC, a convective validation case.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nowcast.config import DOMAIN_ORIGIN, RUNS_DIR, time_index  # noqa: E402
from nowcast.evaluate import NetworkForecaster, load_case  # noqa: E402
from nowcast.plots import GRID_ORIGIN_KM, RADOLAN_CRS  # noqa: E402

INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
NO_DATA = ListedColormap(["#e4e3df"])
RAIN_LEVELS = [0.1, 1, 2, 5, 10, 20, 50, 1000]  # mm/h, as readme_figure.py
RAIN_CMAP = ListedColormap(matplotlib.colormaps["YlGnBu"](np.linspace(0.2, 1.0, len(RAIN_LEVELS) - 1)))
RAIN_NORM = BoundaryNorm(RAIN_LEVELS, RAIN_CMAP.N)
LEADS = (6, 12, 24)  # steps of 5 min; the leads cached for STEPS
WINDOW = 448  # km


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--before", type=Path, required=True, help="checkpoint of the earlier version")
    p.add_argument("--after", type=Path, required=True, help="checkpoint of the current version")
    p.add_argument("--case", default="2021-06-29T15")
    p.add_argument("--labels", nargs=2, default=("earlier version:\nspread from local noise",
                                                  "this version:\nspread from displacement"))
    p.add_argument("--out", type=Path, default=Path("docs/figures/members_2021-06-29.png"))
    p.add_argument("--motion-detail", type=float, default=None, help="current version: lead (min) until which the detail of a less smooth optical flow acts on the motion (default: as trained)")
    args = p.parse_args()

    t = datetime.fromisoformat(args.case + ":00")
    year, t0 = time_index(t)
    lead = [k - 1 for k in LEADS]
    before, after = NetworkForecaster(args.before, 1), NetworkForecaster(args.after, 1, motion_detail=args.motion_detail)
    history, obs, context = load_case(year, t0, max(before.history, after.history))
    members = []
    for net in (before, after):
        members.append(net(history[-net.history:], context, seed=t0)[0, lead].cpu().numpy())
        torch.cuda.empty_cache()
    steps = np.load(RUNS_DIR / "panel_cache" / f"steps_{args.case}.npy").astype(np.float32)[0]
    obs = obs[lead]

    tiles = F.avg_pool2d(torch.from_numpy(np.nan_to_num(obs[1]))[None, None], WINDOW, stride=32)[0, 0]
    r0, c0 = (32 * int(i) for i in np.unravel_index(int(tiles.argmax()), tiles.shape))
    r1, c1 = r0 + WINDOW, c0 + WINDOW
    cut = np.s_[..., r0:r1, c0:c1]
    x0, y0 = GRID_ORIGIN_KM[0] + DOMAIN_ORIGIN[1], GRID_ORIGIN_KM[1] + DOMAIN_ORIGIN[0]
    extent = [1e3 * (x0 + c0), 1e3 * (x0 + c1), 1e3 * (y0 + r0), 1e3 * (y0 + r1)]

    columns = [("observed", obs[cut]), (args.labels[0], members[0][cut]), (args.labels[1], members[1][cut]),
               ("STEPS (pysteps)", steps[cut])]
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(len(LEADS), len(columns), figsize=(10, 7.9), facecolor=SURFACE,
                             subplot_kw={"projection": RADOLAN_CRS}, gridspec_kw=dict(wspace=0.04, hspace=0.05))
    no_data = np.isnan(obs[cut])
    for j, (title, fields) in enumerate(columns):
        for k, minutes in enumerate(5 * np.array(LEADS)):
            ax = axes[k, j]
            ax.set_extent(extent, crs=RADOLAN_CRS)
            kw = dict(origin="lower", extent=extent, transform=RADOLAN_CRS, interpolation="nearest")
            field = np.nan_to_num(fields[k])
            ax.imshow(np.ma.masked_where(~no_data[k], no_data[k]), cmap=NO_DATA, **kw)
            ax.imshow(np.ma.masked_where(no_data[k] | (field < RAIN_LEVELS[0]), field), cmap=RAIN_CMAP, norm=RAIN_NORM, **kw)
            ax.add_feature(cfeature.BORDERS.with_scale("10m"), lw=0.5, edgecolor=MUTED)
            ax.spines["geo"].set_edgecolor(MUTED)
            ax.spines["geo"].set_linewidth(0.6)
            if k == 0:
                ax.set_title(title, fontsize=10, color=INK)
            if j == 0:
                ax.text(-0.06, 0.5, f"+{minutes} min", transform=ax.transAxes, rotation=90, ha="center",
                        va="center", fontsize=10, color=INK)

    box = axes[-1, 0].get_position()
    cax = fig.add_axes([box.x0 + 0.1 * box.width, box.y0 - 0.06, axes[-1, -1].get_position().x1 - box.x0 - 0.2 * box.width, 0.012])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=RAIN_NORM, cmap=RAIN_CMAP), cax=cax, orientation="horizontal",
                        ticks=RAIN_LEVELS[:-1])
    cbar.ax.set_xticklabels([f"{v:g}" for v in RAIN_LEVELS[:-1]])
    cbar.ax.tick_params(labelsize=8, colors=MUTED, length=2)
    cbar.outline.set_edgecolor(MUTED)
    cbar.set_label(f"rain rate (mm/h); {t:%d %b %Y, %H:%M} UTC analysis, {WINDOW} km window, member 1 of each ensemble",
                   fontsize=9, color=INK)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
