#!/usr/bin/env python
"""
Temporal visual gate: animated member-1 sequences, analysis to +120 min, next to the observation and STEPS.

    python scripts/member_animation.py --run lag_v7_disp_smooth "label" --run lag_v6_rigid "label" \
        [--cascade lag_v7_disp_smooth "label"] [--cases 2022-06-20T07 ...] --out DIR

Still panels (member_panel.py) cannot show whether structure persists and moves, whether halos grow,
or how fast the observed texture is lost. `--cascade RUN` evaluates RUN's weights with the texture
cascade of `--texture 64` and `--padding reflection` switched on at inference (the network was not
trained with them): the same member and source, new rain redistributed into cells, mirrored inflow. STEPS sequences (one member, 24 leads) are computed
once per case on the CPU and cached in runs/panel_cache/steps24_<case>.npy.
"""

import argparse
import contextlib
import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nowcast import baselines  # noqa: E402
from nowcast.config import RUNS_DIR, time_index  # noqa: E402
from nowcast.evaluate import LEADS, load_case  # noqa: E402
from nowcast.model import from_checkpoint  # noqa: E402
from nowcast.transforms import y_to_rate  # noqa: E402

CASES = ("2022-06-20T07", "2021-07-14T14", "2021-07-24T05")
LEVELS = [0.1, 0.5, 1, 2, 5, 10, 20, 50]  # mm/h, as member_panel.py
COLORS = ["#edf8b1", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#225ea8", "#253494", "#081d58"]
WINDOW, SCALE = 384, 1.5  # km, downsampling factor of the tiles (256 px)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def colorize(rate: np.ndarray) -> Image.Image:
    lut = np.array([[255, 255, 255]] + [[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in COLORS], dtype=np.uint8)
    k = np.digitize(np.nan_to_num(rate, nan=0.0), LEVELS)  # 0 = below 0.1 mm/h
    img = Image.fromarray(lut[k][::-1])  # origin lower, as in the panels
    size = round(rate.shape[1] / SCALE)
    return img.resize((size, size), Image.NEAREST)


def network(checkpoint: Path, cascade: bool = False, detail: bool = False):
    state = torch.load(checkpoint, map_location="cuda", weights_only=False)
    args = dict(state["args"])
    if cascade:  # the operator of the session-9 queue: cascade, and mirrored inflow at the domain edge
        args["texture"], args["padding"] = 64.0, "reflection"
    if detail:  # the flow detail of session 11, fading out until +60 min
        args["motion_detail"] = 60.0
    model = from_checkpoint(args).cuda().eval()
    model.load_state_dict(state["model"])

    @torch.no_grad()
    def run(history, context, seed):
        valid = torch.from_numpy(np.isfinite(history).all(0))[None, None].cuda()
        y = torch.log1p(torch.from_numpy(np.nan_to_num(history, nan=0.0))).cuda()[None]
        generator = torch.Generator(device="cuda").manual_seed(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = model(y, valid, torch.from_numpy(context).cuda()[None], members=1, generator=generator)["forecast"]
        return y_to_rate(f[0, 0].float()).cpu().numpy()  # (K, H, W)
    return run


def steps_sequence(case, history, seed):
    cache = RUNS_DIR / "panel_cache" / f"steps24_{case}.npy"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            out = baselines.steps(history, LEADS, members=1, seed=seed, workers=4)
        np.save(cache, out[0].astype(np.float16))
    return np.load(cache).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", nargs=2, action="append", default=[], metavar=("RUN", "LABEL"))
    p.add_argument("--cascade", nargs=2, action="append", default=[], metavar=("RUN", "LABEL"))
    p.add_argument("--detail", nargs=2, action="append", default=[], metavar=("RUN", "LABEL"),
                   help="RUN's weights with the fading flow detail (motion_detail 60) switched on at inference")
    p.add_argument("--cases", nargs="+", default=CASES)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ms", type=int, default=260, help="frame duration")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    models = [(label, network(RUNS_DIR / run / "best.pt", False)) for run, label in args.run]
    models += [(label, network(RUNS_DIR / run / "best.pt", True)) for run, label in args.cascade]
    models += [(label, network(RUNS_DIR / run / "best.pt", detail=True)) for run, label in args.detail]
    font, small = ImageFont.truetype(FONT, 15), ImageFont.truetype(FONT, 12)

    for case in args.cases:
        year, t0 = time_index(datetime.fromisoformat(case + ":00"))
        history, obs, context = load_case(year, t0, 6)
        # the window with the most observed rain at +60 min
        tiles = F.avg_pool2d(torch.from_numpy(np.nan_to_num(obs[11]))[None, None], WINDOW, stride=32)[0, 0]
        r, c = (32 * int(i) for i in np.unravel_index(int(tiles.argmax()), tiles.shape))
        cut = np.s_[..., r:r + WINDOW, c:c + WINDOW]
        columns = [("observed", obs)] + [(label, run(history, context, seed=t0)) for label, run in models]
        columns.append(("STEPS (pysteps)", steps_sequence(case, history, seed=t0)))
        analysis = history[-1]

        size, gap, head, foot = round(WINDOW / SCALE), 6, 44, 34
        width = len(columns) * size + (len(columns) - 1) * gap
        frames = []
        for k in range(-1, LEADS):  # -1: the analysis (identical in all columns)
            frame = Image.new("RGB", (width, head + size + foot), "white")
            draw = ImageDraw.Draw(frame)
            for j, (label, seq) in enumerate(columns):
                field = analysis if k < 0 else seq[k]
                frame.paste(colorize(field[cut]), (j * (size + gap), head))
                for line, text in enumerate(label.split("|")):
                    draw.text((j * (size + gap) + 2, 4 + 18 * line), text, fill=(18, 24, 28) if line == 0 else (90, 100, 108),
                              font=font if line == 0 else small)
            when = "analysis" if k < 0 else f"+{5 * (k + 1):3d} min"
            draw.text((2, head + size + 8), f"{case}:00 UTC  {when}   {WINDOW} km window, member 1", fill=(18, 24, 28), font=font)
            x = width
            for lev, col in reversed(list(zip(LEVELS, COLORS))):  # colour key, right-aligned
                tw = draw.textlength(f"{lev:g}", font=small)
                x -= tw + 18
                draw.rectangle((x, head + size + 10, x + 12, head + size + 22), fill=col)
                draw.text((x + 15, head + size + 9), f"{lev:g}", fill=(90, 100, 108), font=small)
            draw.text((x - 44, head + size + 9), "mm/h", fill=(90, 100, 108), font=small)
            frames.append(frame.quantize(colors=32, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
        durations = [900] + [args.ms] * (LEADS - 1) + [1600]
        path = args.out / f"animation_{case}.gif"
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
        print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
