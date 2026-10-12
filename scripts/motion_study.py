#!/usr/bin/env python
"""
How good is a motion field? Advect the analysis field with it and compare with the future.

    python scripts/motion_study.py [--checkpoint runs/<run>/best.pt] [--samples 48]

On validation crops (512 km, drawn like training samples, at least some rain),
the latest observation is advected with stationary velocity fields from:

    zero         Eulerian persistence
    lk           pysteps Lucas-Kanade
    flow         the variational optical flow of advection.estimate_velocity
    network      the velocity of a trained model (first-guess flow + learned correction), member 0
    oracle       a velocity fitted to the *future* (an optimistic bound on what any
                 stationary motion field could achieve)

and the mean absolute error in y = log(1 + R) is reported per lead time.
Pure advection, no growth or decay: this isolates the quality of the motion.
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nowcast import baselines
from nowcast.advection import departure_points, estimate_velocity, sample
from nowcast.config import RUNS_DIR, VAL_YEARS
from nowcast.data import FixedWindows, Windows, to_model_space
from nowcast.model import Nowcaster

LEADS = (6, 12, 24)  # 30, 60, 120 min
GRID = 16


def advection_error(latest, targets, valid, velocity) -> list[float]:
    points = departure_points(velocity, max(LEADS), upsample=GRID)
    return [float(((sample(latest, points[k - 1])[:, 0] - targets[:, k - 1]).abs() * valid[:, k - 1]).sum()
                  / valid[:, k - 1].sum()) for k in LEADS]


def oracle_velocity(latest, targets, valid, iterations=300) -> torch.Tensor:
    velocity = torch.zeros(1, 2, latest.shape[-2] // GRID, latest.shape[-1] // GRID, device=latest.device, requires_grad=True)
    optimizer = torch.optim.Adam([velocity], lr=0.1)
    k_max = max(LEADS)
    for _ in range(iterations):
        points = departure_points(velocity, k_max, upsample=GRID)
        loss = sum(((sample(latest, p)[:, 0] - targets[:, k]).abs() * valid[:, k]).sum() / valid[:, k].sum()
                   for k, p in enumerate(points))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return velocity.detach()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--samples", type=int, default=48)
    p.add_argument("--out", type=Path, default=RUNS_DIR / "motion_study.json")
    args = p.parse_args()

    model = None
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
        model = Nowcaster(history=state["args"]["history"], leads=state["args"]["leads"],
                          optical_flow=not state["args"].get("no_optical_flow", False),
                          noise=state["args"].get("noise", "concat")).cuda().eval()
        model.load_state_dict(state["model"])

    windows = Windows(VAL_YEARS, 6, 24, (512, 512))
    dataset = FixedWindows(windows, 4 * args.samples, seed=99)
    errors = {}
    used = 0
    for i in range(len(dataset)):
        if used == args.samples:
            break
        b = to_model_space({k: v[None] for k, v in dataset[i].items()}, 6, "cuda")
        latest, targets, valid = b["inputs"][:, -1:], b["targets"], b["target_valid"].float()
        if float(F.avg_pool2d((latest > 0.1).float(), GRID).sum()) < 5:
            continue  # too little rain to define motion
        used += 1
        with contextlib.redirect_stdout(io.StringIO()):
            lk = torch.from_numpy(baselines.velocity(np.expm1(b["inputs"][0].cpu().numpy()))).float().cuda()
        fields = {
            "zero": torch.zeros(1, 2, 32, 32, device="cuda"),
            "lk": F.avg_pool2d(lk[None], GRID),
            "flow": estimate_velocity(b["inputs"], b["input_valid"], grid=GRID),
            "oracle": oracle_velocity(latest, targets, valid),
        }
        if model is not None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                fields["network"] = model(b["inputs"], b["input_valid"], b["context"], members=1)["velocity"][:, 0].float()
        for name, velocity in fields.items():
            errors.setdefault(name, []).append(advection_error(latest, targets, valid, velocity))

    summary = {name: np.mean(e, axis=0).tolist() for name, e in errors.items()}
    print(f"advection MAE in y over {used} rainy validation crops (lower is better)")
    print("            " + "".join(f"+{5 * k:>3d} min  " for k in LEADS))
    for name, values in summary.items():
        print(f"  {name:8s}  " + "".join(f"{v:.4f}    " for v in values))
    args.out.write_text(json.dumps({"samples": used, "leads_min": [5 * k for k in LEADS], "mae_y": summary}, indent=2))


if __name__ == "__main__":
    main()
