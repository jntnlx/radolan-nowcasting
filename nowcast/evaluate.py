"""
Full-domain evaluation of the network and the classical baselines.

    python -m nowcast.evaluate --method steps --cases 120              # baselines (CPU)
    python -m nowcast.evaluate --method runs/<run>/best.pt --cases 120  # the network (GPU)

Evaluation population: analysis times on the full hour in the validation
seasons at which the domain-mean rain rate is at least 0.1 mm/h (about a
third of all times: precipitation somewhere over Germany), drawn uniformly
with a fixed seed. Membership depends on the analysis time only, never on
the future, so no method is favoured by the selection. Every method is scored
on the identical cases and pixels over the whole 1024 x 832 km domain, so
the network sees inflow from upstream radar data rather than from crop edges.
The baseline "flow" is extrapolation with the optical flow of advection.py;
"displacement" is the ensemble of such extrapolations in which every member
moves with its own smooth random velocity perturbation: no training, no
growth or decay, and members that are radar fields by construction.

Results go to runs/eval/<population>/<method>.json: pooled scores per lead
time (verify.py) and per-case CRPS for paired comparisons. An interrupted
evaluation resumes from <method>.partial.pt, which is only accepted if it was
written for the same method, checkpoint step and ensemble size.
"""

import argparse
import contextlib
import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from . import baselines
from .config import RUNS_DIR, STEPS_PER_HOUR, TEST_YEARS, VAL_YEARS
from .data import time_context
from .advection import departure_points, estimate_velocity, sample, smooth_noise
from .model import from_checkpoint
from .radklim import depth_to_rate
from .store import Season
from .transforms import y_to_rate
from .verify import Scores

HISTORY, LEADS = 6, 24
MIN_DOMAIN_RAIN = 0.1  # mm/h


def analysis_times(years, cases: int, seed: int = 2024) -> list[tuple[int, int]]:
    """(year, t0) pairs of the evaluation population, sorted in time."""
    pool = []
    for year in years:
        s = Season(year)
        t0 = s.complete_windows(HISTORY, LEADS)
        t0 = t0[t0 % STEPS_PER_HOUR == 0]  # analysis time (last observed frame) on the full hour
        rain, valid = s.tile_rain[t0].astype(np.float32), s.tile_valid[t0].astype(np.float32)
        domain = (rain * valid).sum((1, 2)) / valid.sum((1, 2)).clip(min=1e-6)
        pool += [(year, int(t)) for t in t0[domain >= MIN_DOMAIN_RAIN]]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(pool), size=min(cases, len(pool)), replace=False)
    return sorted(pool[i] for i in chosen)


def load_case(year: int, t0: int, history: int = HISTORY) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rain rates of `history` past frames (ending at t0) and LEADS future frames, and the time context."""
    s = Season(year)
    rate = depth_to_rate(s.frames(t0 - history + 1, t0 + LEADS + 1))
    return rate[:history], rate[history:], time_context(s.time(t0))


def _baseline(method: str, year: int, t0: int, members: int) -> np.ndarray:
    history, _, _ = load_case(year, t0)
    with contextlib.redirect_stdout(io.StringIO()):  # pysteps reports progress on stdout
        if method == "steps":
            return baselines.steps(history, LEADS, members=members, seed=t0)
        return getattr(baselines, method)(history, LEADS)


class FlowForecaster:
    """Deterministic extrapolation with the variational optical flow of advection.py (GPU)."""

    @torch.no_grad()
    def __call__(self, history: np.ndarray, context: np.ndarray, seed: int) -> torch.Tensor:
        valid = torch.from_numpy(np.isfinite(history).all(0))[None, None].cuda()
        y = torch.log1p(torch.from_numpy(np.nan_to_num(history, nan=0.0))).cuda()[None]
        points = departure_points(estimate_velocity(y, valid), LEADS, upsample=16)
        return y_to_rate(torch.cat([sample(y[:, -1:], p) for p in points], dim=1))


class DisplacementForecaster:
    """
    Displacement ensemble of the analysis: flow extrapolation with a smooth (512 km) random velocity
    perturbation per member. `std` (px per step, of the underlying white noise) was chosen on 12
    validation cases; the CRPS is flat between 1.5 and 2 and hardly depends on the scale (64-1024 km).
    """

    def __init__(self, members: int, std: float = 1.5, cells: int = 32, smoothness: float = 0.1):
        self.members, self.std, self.cells, self.smoothness = members, std, cells, smoothness
        self.provenance = {"method": "displacement", "members": members, "std": std, "cells": cells}
        if smoothness != 0.1:  # 0.1: the strongest displacement baseline; 1.0: the untrained network (training flow)
            self.provenance["smoothness"] = smoothness

    @torch.no_grad()
    def __call__(self, history: np.ndarray, context: np.ndarray, seed: int) -> torch.Tensor:
        valid = torch.from_numpy(np.isfinite(history).all(0))[None, None].cuda()
        y = torch.log1p(torch.from_numpy(np.nan_to_num(history, nan=0.0))).cuda()[None]
        velocity = estimate_velocity(y, valid, smoothness=self.smoothness)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        velocity = velocity + self.std * smooth_noise(self.members, 2, *velocity.shape[-2:], self.cells, generator, "cuda")
        latest = y[:, -1:].expand(self.members, -1, -1, -1)
        return y_to_rate(torch.cat([sample(latest, p) for p in departure_points(velocity, LEADS, upsample=16)], dim=1))


class NetworkForecaster:
    def __init__(self, checkpoint: Path, members: int, chunk: int = 5, motion_detail: float | None = None):
        state = torch.load(checkpoint, map_location="cuda", weights_only=False)
        args = state["args"]
        self.model = from_checkpoint(args).cuda().eval()
        self.provenance = {"checkpoint": str(checkpoint), "step": state.get("step"), "members": members}
        if motion_detail is not None:  # an option of the forecast operator, not of the weights: any checkpoint takes it
            self.model.motion_detail = motion_detail
            self.provenance["motion_detail"] = motion_detail
        self.model.load_state_dict(state["model"])
        self.members, self.chunk, self.history = members, chunk, args["history"]
        self.generator = torch.Generator(device="cuda")

    @torch.no_grad()
    def __call__(self, history: np.ndarray, context: np.ndarray, seed: int) -> torch.Tensor:
        valid = torch.from_numpy(np.isfinite(history).all(0))[None, None].cuda()
        y = torch.log1p(torch.from_numpy(np.nan_to_num(history, nan=0.0))).cuda()[None]
        ctx = torch.from_numpy(context).cuda()[None]
        self.generator.manual_seed(seed)
        prior = self.model.first_guess(y, valid)
        detail = self.model.flow_detail(y, valid, prior) if self.model.motion_detail > 0 else None
        out = []
        for _ in range(0, self.members, self.chunk):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = self.model(y, valid, ctx, members=self.chunk, generator=self.generator, prior=prior, detail=detail)["forecast"]
            out.append(y_to_rate(f[0].float()))
        return torch.cat(out)[:self.members]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, help="persistence | extrapolation | flow | displacement | steps | path to a checkpoint")
    p.add_argument("--cases", type=int, default=120)
    p.add_argument("--members", type=int, default=20)
    p.add_argument("--test", action="store_true", help="evaluate the locked test seasons")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--name", default=None, help="result name (default: method or run name)")
    p.add_argument("--motion-detail", type=float, default=None,
                   help="network only: lead (min) until which the detail of a less smooth optical flow acts on the motion "
                        "(default: as trained; 0 switches it off)")
    args = p.parse_args(argv)

    years = TEST_YEARS if args.test else VAL_YEARS
    population = f"{'test' if args.test else 'val'}_{args.cases}"
    cases = analysis_times(years, args.cases)
    is_network = args.method.endswith(".pt")
    name = args.name or (Path(args.method).parent.name if is_network else args.method)
    members = args.members if (is_network or args.method in ("steps", "displacement")) else 1
    out_dir = RUNS_DIR / "eval" / population
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accumulators are checkpointed after every case, so an interrupted evaluation resumes where it stopped.
    scores = Scores(LEADS, members)
    per_case = []
    partial = out_dir / f"{name}.partial.pt"
    forecaster = None
    if is_network:
        forecaster = NetworkForecaster(Path(args.method), members, motion_detail=args.motion_detail)
    elif args.method in ("flow", "displacement"):
        forecaster = FlowForecaster() if args.method == "flow" else DisplacementForecaster(members)
    provenance = getattr(forecaster, "provenance", {"method": args.method, "members": members})
    if partial.exists():
        state = torch.load(partial, map_location="cuda", weights_only=False)
        if state.get("provenance", provenance) != provenance:
            raise SystemExit(f"{partial} was written for {state['provenance']}, not {provenance}: delete it to start over")
        scores.__dict__.update(state["scores"])
        per_case = state["cases"]
    done = {(c["year"], c["t0"]) for c in per_case}
    todo = [case for case in cases if case not in done]
    print(f"{name}: {len(done)} of {len(cases)} cases already scored", flush=True)

    def record(forecast, obs, year, t0):
        before = {s: (scores.crps[s].clone(), scores.crps_n[s].clone()) for s in (1, 16)}
        scores.add(torch.as_tensor(forecast), torch.from_numpy(obs))
        per_case.append({
            "year": year, "t0": t0, "time": Season(year).time(t0).isoformat(),
            **{f"crps_y_{s}km": ((scores.crps[s] - b[0]) / (scores.crps_n[s] - b[1])).cpu().tolist()
               for s, b in before.items()},
        })
        torch.save({"scores": scores.__dict__, "cases": per_case, "provenance": provenance}, partial)
        print(f"{len(per_case)}/{len(cases)} {per_case[-1]['time']}  crps_1km@60min={per_case[-1]['crps_y_1km'][11]:.4f}", flush=True)

    if forecaster is not None:
        for year, t0 in todo:
            history, obs, context = load_case(year, t0, getattr(forecaster, "history", HISTORY))
            record(forecaster(history, context, seed=t0), obs, year, t0)
    else:
        # At most 2 x workers forecasts in flight: fast baselines would otherwise
        # pile up all results in memory while scoring catches up.
        with ProcessPoolExecutor(args.workers) as pool:
            queue, pending = iter(todo), {}

            def submit():
                case = next(queue, None)
                if case is not None:
                    pending[case] = pool.submit(_baseline, args.method, *case, members)

            for _ in range(2 * args.workers):
                submit()
            for year, t0 in todo:
                forecast = pending.pop((year, t0)).result()
                submit()
                record(forecast, load_case(year, t0)[1], year, t0)
                del forecast

    per_case.sort(key=lambda c: (c["year"], c["t0"]))
    result = {"method": args.method, "population": population, "provenance": provenance, "summary": scores.summary(), "cases": per_case}
    (out_dir / f"{name}.json").write_text(json.dumps(result))
    partial.unlink(missing_ok=True)
    print("crps_y_1km by lead:", np.round(result["summary"]["crps_y_1km"], 4).tolist())


if __name__ == "__main__":
    main()
