#!/usr/bin/env python
"""
README figure: observation, one ensemble member and the ensemble probability of
heavy rain for one case, on a section of the domain.

    python scripts/readme_figure.py --checkpoint runs/<run>/best.pt

Default case: 2021-07-14 14:00 UTC (July 2021 floods), the hour of that day with
the largest observed area above 5 mm/h; it is one of the 120 validation cases.
The member shown is the first one drawn (seed = analysis time), not a selection.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nowcast.config import DOMAIN_ORIGIN, time_index  # noqa: E402
from nowcast.evaluate import NetworkForecaster, load_case  # noqa: E402
from nowcast.plots import GRID_ORIGIN_KM, RADOLAN_CRS  # noqa: E402

INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
NO_DATA = ListedColormap(["#e4e3df"])

# Ordered ramps, monotone in lightness: rain rate (green-blue), probability (orange).
RAIN_LEVELS = [0.1, 1, 2, 5, 10, 20, 50, 1000]  # mm/h
RAIN_CMAP = ListedColormap(matplotlib.colormaps["YlGnBu"](np.linspace(0.2, 1.0, len(RAIN_LEVELS) - 1)))
RAIN_NORM = BoundaryNorm(RAIN_LEVELS, RAIN_CMAP.N)
PROB_LEVELS = [0.05, 0.2, 0.4, 0.6, 0.8, 1.0]
PROB_CMAP = ListedColormap(matplotlib.colormaps["Oranges"](np.linspace(0.15, 0.75, len(PROB_LEVELS) - 1)))
PROB_NORM = BoundaryNorm(PROB_LEVELS, PROB_CMAP.N)

AHR = (7.0, 50.5)  # lon, lat of the Ahr valley, the worst-hit area


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--time", default="2021-07-14T14:00", help="analysis time (UTC)")
    p.add_argument("--rows", type=int, nargs=2, default=(150, 662), help="domain rows (south -> north) of the section")
    p.add_argument("--cols", type=int, nargs=2, default=(40, 488), help="domain columns (west -> east) of the section")
    p.add_argument("--members", type=int, default=20)
    p.add_argument("--threshold", type=float, default=5.0, help="mm/h")
    p.add_argument("--out", type=Path, default=Path("docs/figures/case_2021-07-14.png"))
    p.add_argument("--motion-detail", type=float, default=None, help="lead (min) until which the detail of a less smooth optical flow acts on the motion (default: as trained)")
    args = p.parse_args()

    t = datetime.fromisoformat(args.time)
    year, t0 = time_index(t)
    network = NetworkForecaster(args.checkpoint, args.members, motion_detail=args.motion_detail)
    history, obs, context = load_case(year, t0, network.history)
    forecast = network(history, context, seed=t0).cpu().numpy()  # (M, K, H, W) mm/h

    (r0, r1), (c0, c1) = args.rows, args.cols
    cut = np.s_[..., r0:r1, c0:c1]
    history, obs, forecast = history[cut], obs[cut], forecast[cut]
    x0, y0 = GRID_ORIGIN_KM[0] + DOMAIN_ORIGIN[1], GRID_ORIGIN_KM[1] + DOMAIN_ORIGIN[0]
    extent = [1e3 * (x0 + c0), 1e3 * (x0 + c1), 1e3 * (y0 + r0), 1e3 * (y0 + r1)]

    def panel(ax, field, floor, cmap, norm, no_data):
        """Values below `floor` are left white; pixels without radar data are grey."""
        ax.set_extent(extent, crs=RADOLAN_CRS)
        kw = dict(origin="lower", extent=extent, transform=RADOLAN_CRS, interpolation="nearest")
        ax.imshow(np.ma.masked_where(~no_data, no_data), cmap=NO_DATA, **kw)
        ax.imshow(np.ma.masked_where(no_data | (field < floor), field), cmap=cmap, norm=norm, **kw)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), lw=0.5, edgecolor=MUTED)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), lw=0.5, edgecolor=MUTED)
        ax.spines["geo"].set_edgecolor(MUTED)
        ax.spines["geo"].set_linewidth(0.6)

    plt.rcParams.update({"font.size": 10})
    leads = (6, 12, 24)  # steps of 5 min
    aspect = (r1 - r0) / (c1 - c0)
    fig, axes = plt.subplots(3, 4, figsize=(9, 3 * 2.05 * aspect + 0.9), facecolor=SURFACE,
                             subplot_kw={"projection": RADOLAN_CRS}, gridspec_kw=dict(wspace=0.04, hspace=0.06))

    panel(axes[0, 0], np.nan_to_num(history[-1]), RAIN_LEVELS[0], RAIN_CMAP, RAIN_NORM, np.isnan(history[-1]))
    axes[0, 0].set_title(f"{t:%d %b %Y, %H:%M} UTC", fontsize=10, color=INK)
    ahr = RADOLAN_CRS.transform_point(*AHR, ccrs.PlateCarree())
    halo = [patheffects.withStroke(linewidth=2.5, foreground="white")]
    axes[0, 0].plot(*ahr, marker="o", ms=7, mfc="none", mec=INK, mew=1.2, path_effects=halo)  # axes use the RADOLAN projection
    axes[0, 0].annotate("Ahr", ahr, xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK, path_effects=halo)

    for c, k in enumerate(leads, start=1):
        o = obs[k - 1]
        no_data = np.isnan(o)
        panel(axes[0, c], np.nan_to_num(o), RAIN_LEVELS[0], RAIN_CMAP, RAIN_NORM, no_data)
        axes[0, c].set_title(f"+{5 * k} min", fontsize=10, color=INK)
        panel(axes[1, c], forecast[0, k - 1], RAIN_LEVELS[0], RAIN_CMAP, RAIN_NORM, no_data)
        probability = (forecast[:, k - 1] > args.threshold).mean(0)
        panel(axes[2, c], probability, PROB_LEVELS[0], PROB_CMAP, PROB_NORM, no_data)
        # Outline of the observed exceedance, lightly smoothed so that single pixels do not clutter it.
        event = gaussian_filter((np.nan_to_num(o) > args.threshold).astype(float), 1.5)
        axes[2, c].contour(event, levels=[0.5], origin="lower", extent=extent, transform=RADOLAN_CRS,
                           colors=INK, linewidths=0.55)

    labels = ["observed", "forecast member", "forecast probability"]
    for r, label in enumerate(labels):
        ax = axes[r, 0] if r == 0 else axes[r, 1]
        ax.text(-0.06, 0.5, label, transform=ax.transAxes, rotation=90, ha="center", va="center",
                fontsize=10, color=INK)

    # The cells below the analysis panel hold the two colour scales.
    for ax in axes[1:, 0]:
        ax.remove()
    for r, (cmap, norm, ticks, label) in enumerate([
            (RAIN_CMAP, RAIN_NORM, RAIN_LEVELS[:-1], "rain rate (mm/h)"),
            (PROB_CMAP, PROB_NORM, PROB_LEVELS, f"probability of R > {args.threshold:g} mm/h")], start=1):
        box = axes[r, 1].get_position()
        cax = fig.add_axes([box.x0 - box.width * 1.0, box.y0 + 0.62 * box.height, 0.78 * box.width, 0.06 * box.height])
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=ticks)
        cbar.ax.set_xticklabels([f"{v:g}" for v in ticks])
        cbar.ax.tick_params(labelsize=8, colors=MUTED, length=2)
        cbar.outline.set_edgecolor(MUTED)
        cax.set_title(label, fontsize=9, color=INK)
    box = axes[2, 1].get_position()
    fig.legend(handles=[Line2D([], [], color=INK, lw=0.8, label=f"observed\n> {args.threshold:g} mm/h"),
                        Line2D([], [], color="none", marker="s", ms=8, mfc=NO_DATA(0), mec="none", label="no radar data")],
               loc="upper left", bbox_to_anchor=(box.x0 - box.width * 1.02, box.y0 + 0.45 * box.height),
               frameon=False, fontsize=8, handlelength=1.6)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
