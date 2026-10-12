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

Results go to runs/eval/<population>/<method>.json: pooled scores per lead
time (verify.py) and per-case CRPS for paired comparisons. An interrupted
evaluation resumes from <method>.partial.pt.
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
from .advection import departure_points, estimate_velocity, sample
from .model import Nowcaster
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


def load_case(year: int, t0: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = Season(year)
    rate = depth_to_rate(s.frames(t0 - HISTORY + 1, t0 + LEADS + 1))
    return rate[:HISTORY], rate[HISTORY:], time_context(s.time(t0))


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


class NetworkForecaster:
    def __init__(self, checkpoint: Path, members: int, chunk: int = 5):
        state = torch.load(checkpoint, map_location="cuda", weights_only=False)
        args = state["args"]
        self.model = Nowcaster(history=args["history"], leads=args["leads"], eulerian=args.get("eulerian", False),
                               optical_flow=not args.get("no_optical_flow", True),
                               noise=args.get("noise", "concat")).cuda().eval()
        self.model.load_state_dict(state["model"])
        self.members, self.chunk = members, chunk
        self.generator = torch.Generator(device="cuda")

    @torch.no_grad()
    def __call__(self, history: np.ndarray, context: np.ndarray, seed: int) -> torch.Tensor:
        valid = torch.from_numpy(np.isfinite(history).all(0))[None, None].cuda()
        y = torch.log1p(torch.from_numpy(np.nan_to_num(history, nan=0.0))).cuda()[None]
        ctx = torch.from_numpy(context).cuda()[None]
        self.generator.manual_seed(seed)
        prior = self.model.first_guess(y, valid)
        out = []
        for _ in range(0, self.members, self.chunk):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = self.model(y, valid, ctx, members=self.chunk, generator=self.generator, prior=prior)["forecast"]
            out.append(y_to_rate(f[0].float()))
        return torch.cat(out)[:self.members]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, help="persistence | extrapolation | flow | steps | path to a checkpoint")
    p.add_argument("--cases", type=int, default=120)
    p.add_argument("--members", type=int, default=20)
    p.add_argument("--test", action="store_true", help="evaluate the locked test seasons")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--name", default=None, help="result name (default: method or run name)")
    args = p.parse_args(argv)

    years = TEST_YEARS if args.test else VAL_YEARS
    population = f"{'test' if args.test else 'val'}_{args.cases}"
    cases = analysis_times(years, args.cases)
    is_network = args.method.endswith(".pt")
    name = args.name or (Path(args.method).parent.name if is_network else args.method)
    members = args.members if (is_network or args.method == "steps") else 1
    out_dir = RUNS_DIR / "eval" / population
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accumulators are checkpointed after every case, so an interrupted evaluation resumes where it stopped.
    scores = Scores(LEADS, members)
    per_case = []
    partial = out_dir / f"{name}.partial.pt"
    if partial.exists():
        state = torch.load(partial, map_location="cuda", weights_only=False)
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
        torch.save({"scores": scores.__dict__, "cases": per_case}, partial)
        print(f"{len(per_case)}/{len(cases)} {per_case[-1]['time']}  crps_1km@60min={per_case[-1]['crps_y_1km'][11]:.4f}", flush=True)

    if is_network or args.method == "flow":
        forecaster = NetworkForecaster(Path(args.method), members) if is_network else FlowForecaster()
        for year, t0 in todo:
            history, obs, context = load_case(year, t0)
            record(forecaster(history, context, seed=t0), obs, year, t0)
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            futures = [pool.submit(_baseline, args.method, y, t, members) for y, t in todo]
            for i, (year, t0) in enumerate(todo):
                forecast, futures[i] = futures[i].result(), None  # release each ~1 GB ensemble once scored
                record(forecast, load_case(year, t0)[1], year, t0)
                del forecast

    per_case.sort(key=lambda c: (c["year"], c["t0"]))
    result = {"method": args.method, "population": population, "summary": scores.summary(), "cases": per_case}
    (out_dir / f"{name}.json").write_text(json.dumps(result))
    partial.unlink(missing_ok=True)
    print("crps_y_1km by lead:", np.round(result["summary"]["crps_y_1km"], 4).tolist())


if __name__ == "__main__":
    main()
