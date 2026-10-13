"""
Train the stochastic Lagrangian nowcaster.

    python -m nowcast.train --run lagrangian_v1 --hours 8
    python -m nowcast.train --run lagrangian_v1 --resume   # continue after an interruption

Objective per sample: multiscale almost-fair CRPS of the forecast ensemble,
plus a small motion-consistency term that ties the velocity field to the
observed past (losses.py). Validation tracks the fair CRPS (alpha = 1) of an
exponential moving average of the weights on a fixed sample of validation
windows, next to the CRPS of persistence on the same windows.
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import RUNS_DIR, TRAIN_YEARS, VAL_YEARS
from .data import FixedWindows, RandomWindows, Windows, to_model_space
from .losses import motion_consistency, multiscale_crps, smoothness
from .model import Nowcaster


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--resume", action="store_true",
                   help="continue an interrupted run from its last state (settings are taken from the run)")
    p.add_argument("--hours", type=float, default=1.0, help="wall-clock training budget")
    p.add_argument("--steps", type=int, default=0, help="optimizer steps (0: as many as fit the budget)")
    p.add_argument("--crop", type=int, nargs=2, default=(512, 512))
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--members", type=int, default=2)
    p.add_argument("--history", type=int, default=6)
    p.add_argument("--leads", type=int, default=24)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--alpha", type=float, default=0.95)
    p.add_argument("--scales", type=int, nargs="+", default=(1, 4, 16, 64))
    p.add_argument("--motion-weight", type=float, default=0.1)
    p.add_argument("--smooth-weight", type=float, default=0.01)
    p.add_argument("--heavy-weight", type=float, default=0.0,
                   help="weight of an extra CRPS of R**heavy_power (threshold weighting towards heavy rain)")
    p.add_argument("--heavy-power", type=float, default=0.5)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--val-samples", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--eulerian", action="store_true", help="ablation: disable advection")
    p.add_argument("--no-optical-flow", action="store_true", help="ablation: motion from the network only")
    p.add_argument("--noise", choices=("film", "concat"), default="film")
    p.add_argument("--film-levels", type=int, nargs="+", default=(5, 2),
                   help="noise fields with correlation scales 2^level km (FiLM noise)")
    p.add_argument("--train-years", type=int, nargs="+", default=TRAIN_YEARS)
    p.add_argument("--val-years", type=int, nargs="+", default=VAL_YEARS)
    return p.parse_args(argv)


def lr_factor(step: int, progress: float, warmup: int = 1000) -> float:
    """Linear warmup over `warmup` steps, then cosine decay to 10% over training progress in [0, 1]."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def objective(model, batch, args, members=None, alpha=None):
    out = model(batch["inputs"], batch["input_valid"], batch["context"], members=members or args.members)
    scores = multiscale_crps(out["forecast"], batch["targets"], batch["target_valid"],
                             scales=tuple(args.scales), alpha=args.alpha if alpha is None else alpha)
    m = out["forecast"].shape[1]
    scores["motion"] = motion_consistency(
        batch["inputs"].float().repeat_interleave(m, dim=0),
        batch["input_valid"][:, 0].repeat_interleave(m, dim=0),
        out["points"],
    )
    scores["smoothness"] = smoothness(out["velocity"])
    scores["loss"] = scores["crps"] + args.motion_weight * scores["motion"] + args.smooth_weight * scores["smoothness"]
    if args.heavy_weight > 0:
        scores["crps_heavy"] = multiscale_crps(out["forecast"], batch["targets"], batch["target_valid"],
                                               scales=tuple(args.scales), alpha=args.alpha if alpha is None else alpha,
                                               power=args.heavy_power)["crps"]
        scores["loss"] = scores["loss"] + args.heavy_weight * scores["crps_heavy"]
    scores["speed_kmh"] = 12 * out["velocity"].norm(dim=2).mean()
    return scores


@torch.no_grad()
def validate(model, loader, args, device, members=4) -> dict[str, float]:
    model.eval()
    totals, persistence, n = {}, {}, 0
    for raw in loader:
        batch = to_model_space(raw, args.history, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = objective(model, batch, args, members=members, alpha=1.0)
        latest = batch["inputs"][:, -1:].expand_as(batch["targets"])[:, None]
        base = multiscale_crps(latest, batch["targets"], batch["target_valid"], scales=tuple(args.scales))
        for k, v in scores.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        for k, v in base.items():
            persistence[k] = persistence.get(k, 0.0) + float(v)
        n += 1
    model.train()
    result = {f"val_{k}": v / n for k, v in totals.items()}
    result.update({f"val_persistence_{k}": v / n for k, v in persistence.items()})
    result["val_skill"] = 1 - result["val_crps"] / result["val_persistence_crps"]
    return result


def main(argv=None):
    args = parse_args(argv)
    run_dir = RUNS_DIR / args.run
    state = None
    if args.resume:
        defaults = vars(parse_args(["--run", args.run]))  # options added since the run started
        args = argparse.Namespace(**{**defaults, **json.loads((run_dir / "args.json").read_text()), "resume": True})
        state = torch.load(run_dir / "state.pt", map_location="cpu", weights_only=False)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if state is None:
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    step = state["step"] if state else 0

    train_windows = Windows(args.train_years, args.history, args.leads, tuple(args.crop))
    val_windows = Windows(args.val_years, args.history, args.leads, tuple(args.crop))
    # A resumed run draws a fresh sample stream instead of replaying the first one.
    loader = DataLoader(RandomWindows(train_windows, args.seed + step), batch_size=args.batch,
                        num_workers=args.workers, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    val_loader = DataLoader(FixedWindows(val_windows, args.val_samples, seed=12345), batch_size=args.batch,
                            num_workers=args.workers)

    model = Nowcaster(history=args.history, leads=args.leads, eulerian=args.eulerian,
                      optical_flow=not args.no_optical_flow, noise=args.noise,
                      film_levels=tuple(args.film_levels)).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    best, elapsed_before = math.inf, 0.0
    if state:
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        best, elapsed_before = state["best"], state["elapsed"]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.run}: {n_params / 1e6:.1f} M parameters, {len(train_windows.times)} training analysis times"
          + (f", resuming at step {step} after {elapsed_before / 3600:.2f} h" if state else ""), flush=True)
    step_fn = torch.compile(model) if args.compile else model

    budget = args.hours * 3600
    columns = ["step", "hours", "lr", "grad_norm", "loss", "crps", "crps_1km", "crps_4km", "crps_16km", "crps_64km", "motion", "smoothness", "speed_kmh"]
    log = open(run_dir / "log.tsv", "a" if state else "w")
    if not state:
        log.write("\t".join(columns) + "\n")
    started = time.time() - elapsed_before  # the time budget counts training time across sessions
    skipped = 0
    for raw in loader:
        elapsed = time.time() - started
        progress = max(elapsed / budget, step / args.steps if args.steps else 0.0)
        if progress >= 1:
            break
        for group in optimizer.param_groups:
            group["lr"] = args.lr * lr_factor(step, progress)

        batch = to_model_space(raw, args.history, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = objective(step_fn, batch, args)
        if not torch.isfinite(scores["loss"]):
            raise FloatingPointError(f"non-finite loss at step {step}: { {k: float(v) for k, v in scores.items()} }")
        optimizer.zero_grad(set_to_none=True)
        scores["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):  # never let one bad batch corrupt the weights
            skipped += 1
            print(f"  skipped step {step}: non-finite gradient ({skipped} so far)", flush=True)
            if skipped > 10:
                raise FloatingPointError("more than 10 non-finite gradients")
            continue
        optimizer.step()
        with torch.no_grad():
            decay = min(args.ema, (1 + step) / (10 + step))
            for p_ema, p in zip(ema.parameters(), model.parameters()):
                p_ema.lerp_(p, 1 - decay)
        step += 1

        if step % 50 == 0 or step == 1:
            row = {"step": step, "hours": elapsed / 3600, "lr": optimizer.param_groups[0]["lr"],
                   "grad_norm": float(grad_norm), **{k: v.item() for k, v in scores.items()}}
            log.write("\t".join(f"{row.get(c, float('nan')):.6g}" for c in columns) + "\n")
            log.flush()
            print(" ".join(f"{k}={row[k]:.4g}" for k in ("step", "hours", "loss", "crps", "motion", "speed_kmh")), flush=True)

        if step % args.val_every == 0:
            val = {"step": step, "hours": elapsed / 3600, **validate(ema, val_loader, args, device)}
            with open(run_dir / "val.jsonl", "a") as f:
                f.write(json.dumps(val) + "\n")
            print(f"  validation: crps={val['val_crps']:.4f} persistence={val['val_persistence_crps']:.4f} "
                  f"skill={val['val_skill']:.3f}", flush=True)
            if val["val_crps"] < best:
                best = val["val_crps"]
                torch.save({"model": ema.state_dict(), "args": vars(args), "step": step, "val": val}, run_dir / "best.pt")
            # Full training state for --resume, written atomically.
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step, "elapsed": time.time() - started, "best": best}, run_dir / "state.tmp")
            (run_dir / "state.tmp").replace(run_dir / "state.pt")

    final = validate(ema, val_loader, args, device)
    torch.save({"model": ema.state_dict(), "args": vars(args), "step": step, "val": final}, run_dir / "last.pt")
    (run_dir / "final.json").write_text(json.dumps({"step": step, **final}, indent=2))
    print("final", json.dumps(final, indent=1), flush=True)


if __name__ == "__main__":
    main()
