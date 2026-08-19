"""Generate the NPE training dataset.

Sobol-samples the structural parameters, solves the lifecycle model for each
draw, simulates one household trajectory, aggregates to biennial observation
waves, drops households whose lineage was replaced by a newborn inside the
window, and saves ``(theta, x)`` to a ``.pt`` file.

Two simulators are available:

``twoasset`` (Phase 3, default)
    Port of Laibson et al.: credit cards, illiquid asset, naive quasi-hyperbolic
    discounting. Estimates ``(beta, delta, crra)``. Slow -- one solve is tens of
    seconds to ~10 minutes depending on ``--grid``.

``hark`` (Phases 1-2)
    Single liquid asset, no borrowing, ``beta`` locked at 1. Estimates
    ``(delta, crra)``. Kept so the Phase 2 results stay reproducible.

Usage::

    uv run python scripts/generate_dataset.py --n_samples 32 --n_jobs 4
    uv run python scripts/generate_dataset.py --simulator hark
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from math import ceil
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

from hh_npe.data.dataset import save_dataset
from hh_npe.npe.prior import PHASE3, PriorBox, sample_sobol
from hh_npe.simulator.dispatch import SIMULATORS
from hh_npe.utils.seeding import seed_all

log = logging.getLogger("generate_dataset")


def generate_block_gpu(
    thetas: np.ndarray, seed_base: int, start_age: int, n_waves: int,
    wave_years: int, grid: str, theta_batch: int, chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a block of draws on the GPU, then forward-simulate on the CPU.

    The backward induction is the only expensive part and is batched over draws;
    the forward pass and wave aggregation are milliseconds and stay on the CPU,
    reusing exactly the code the CPU path uses.
    """
    from hh_npe.data.waves import FEATURES_TWOASSET, aggregate_waves
    from hh_npe.simulator.dispatch import AGE_START_SIM
    from hh_npe.simulator.twoasset import GRIDS, simulate
    from hh_npe.simulator.twoasset_gpu import solve_batch

    # Consume each sub-batch before solving the next: a Solution holds ~58 MB of
    # policy arrays, so accumulating a whole large block would exhaust host RAM.
    xs, alives = [], []
    for s0 in range(0, len(thetas), theta_batch):
        s1 = min(s0 + theta_batch, len(thetas))
        sols = solve_batch(thetas[s0:s1], GRIDS[grid], theta_batch=theta_batch,
                           chunk=chunk)
        for i, sol in enumerate(sols):
            panel = simulate(sol, n_households=1, seed=seed_base + s0 + i)
            x, alive = aggregate_waves(
                panel, age_start_sim=AGE_START_SIM, start_age=start_age,
                n_waves=n_waves, wave_years=wave_years, features=FEATURES_TWOASSET,
            )
            xs.append(x[0])
            alives.append(alive[0])
        del sols
    return np.stack(xs), np.stack(alives)


def _shard_dir(out: Path) -> Path:
    return out.parent / (out.stem + "_shards")


def _solver_config(args) -> dict:
    """The settings a shard's contents actually depend on.

    The GPU solve is reproducible at a fixed (device, theta_batch, chunk) but
    not across them, and not across GPU models: the expectation step contracts
    through cuBLAS, whose summation order varies with problem size and
    architecture, and the grids are built from round dollar amounts so ~99% of
    states hold two exactly-tied choices for those last bits to decide between.
    Mixing configurations silently produces a dataset from two different
    simulators -- which is precisely what an A100-to-V100 swap did mid-run,
    moving 6 of 16 draws by up to $48,000.
    """
    cfg = {"simulator": args.simulator, "grid": args.grid, "device": args.device,
           "start_age": args.start_age, "n_waves": args.n_waves,
           "wave_years": args.wave_years, "seed": args.seed}
    if args.device == "cuda":
        import torch
        cfg |= {"theta_batch": args.theta_batch, "chunk": args.chunk,
                "gpu": torch.cuda.get_device_name(0)}
    return cfg


def _check_config(shard_dir: Path, cfg: dict, log_fn) -> None:
    """Refuse to add shards to a directory built under different settings."""
    marker = shard_dir / "solver_config.json"
    if not marker.exists():
        marker.write_text(json.dumps(cfg, indent=2, sort_keys=True))
        return
    old = json.loads(marker.read_text())
    if old == cfg:
        return
    diff = {k: (old.get(k), cfg.get(k)) for k in set(old) | set(cfg)
            if old.get(k) != cfg.get(k)}
    raise SystemExit(
        f"Existing shards in {shard_dir} were generated under a different "
        f"configuration, so resuming would mix two simulators:\n"
        + "\n".join(f"  {k}: {was!r} -> {now!r}" for k, (was, now) in sorted(diff.items()))
        + "\nMove the directory aside to start a fresh dataset, or restore the "
          "original settings to continue this one."
    )


def assemble(out: Path, theta_np: np.ndarray, log_fn) -> bool:
    """Build the ``.pt`` dataset from whatever contiguous shards exist.

    Only a contiguous prefix starting at shard 0 is used. That is deliberate:
    scrambled Sobol keeps its low-discrepancy balance on power-of-2 *prefixes*,
    so a prefix is a valid smaller dataset while an arbitrary subset is not.
    """
    shards = sorted(_shard_dir(out).glob("shard_*.npz"))
    if not shards:
        log_fn("No shards found.")
        return False

    xs, alives, expected = [], [], 0
    for sf in shards:
        d = np.load(sf)
        if int(d["lo"]) != expected:  # gap - stop at the contiguous prefix
            break
        xs.append(d["x"])
        alives.append(d["alive"])
        expected = int(d["hi"])

    x_np = np.concatenate(xs)
    alive_np = np.concatenate(alives)
    n = x_np.shape[0]
    fully_alive = alive_np.all(axis=1)
    n_alive = int(fully_alive.sum())
    log_fn(
        f"Assembling {n} contiguous samples from {len(xs)} shards; "
        f"survival filter keeps {n_alive} ({100 * n_alive / n:.1f}%)."
    )

    theta = torch.from_numpy(theta_np[:n][fully_alive]).float()
    x = torch.from_numpy(x_np[fully_alive]).float()
    save_dataset(theta, x, out)
    log_fn(f"Saved (theta {tuple(theta.shape)}, x {tuple(x.shape)}) to {out}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator", choices=sorted(SIMULATORS), default="twoasset")
    parser.add_argument("--grid", choices=["coarse", "mid", "full"], default="mid",
                        help="'mid' (107x56) carries 0.94 se of discretization "
                             "error and is the Phase 3 working choice; 'full' is "
                             "Laibson et al.'s exact 190x84 and is ~7x slower.")
    parser.add_argument("--n_samples", type=int, default=32768,
                        help="Total Sobol draws. Powers of 2 preferred - Sobol "
                             "keeps its balance on power-of-2 prefixes, so "
                             "partial runs assemble into valid datasets.")
    parser.add_argument("--block", type=int, default=512,
                        help="Samples per checkpoint shard. A multi-day run must "
                             "survive interruption; shards let it resume.")
    parser.add_argument("--start_age", type=int, default=30)
    parser.add_argument("--n_waves", type=int, default=5)
    parser.add_argument("--wave_years", type=int, default=2,
                        help="Years per wave. 2 = biennial (PSID); 1 = annual.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="joblib parallelism; -1 uses all cores.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Defaults to data/processed/<simulator>_dataset.pt")
    parser.add_argument("--assemble_only", action="store_true",
                        help="Build the .pt from existing shards and exit. Use to "
                             "train on a partial run without stopping it.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="'cuda' batches the backward induction on GPU "
                             "(~4.8 s/solve at the full grid vs ~1678 s on one "
                             "CPU core).")
    parser.add_argument("--theta_batch", type=int, default=16,
                        help="cuda only: draws solved simultaneously. Capped to "
                             "what device memory allows; the solve is "
                             "batch-invariant, so this changes speed only.")
    parser.add_argument("--chunk", type=int, default=48,
                        help="cuda only: liquid-grid rows per inner block. "
                             "Smaller chunks shrink each slab and so allow a "
                             "larger theta_batch; the product is what fits.")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed_all(args.seed)

    box = PHASE3 if args.simulator == "twoasset" else PriorBox()
    out = args.out or Path(f"data/processed/{args.simulator}_dataset.pt")
    fn = SIMULATORS[args.simulator]
    theta_np = sample_sobol(args.n_samples, box, seed=args.seed)

    if args.assemble_only:
        assemble(out, theta_np, log.info)
        return

    shard_dir = _shard_dir(out)
    shard_dir.mkdir(parents=True, exist_ok=True)
    cfg = _solver_config(args)
    _check_config(shard_dir, cfg, log.info)
    if args.device == "cuda":
        # The planner silently caps theta_batch to what fits. Silent is wrong
        # here: the recorded config would no longer describe the shards, and a
        # card with less free memory would quietly start a second regime.
        from hh_npe.simulator import grids
        from hh_npe.simulator.twoasset import GRIDS
        from hh_npe.simulator.twoasset_gpu import _plan_batching

        spec = GRIDS[args.grid]
        age = grids.ages(spec.age_start, spec.age_end)
        nX = len(grids.liquid_grid(age, spec.xjump, spec.xmax, spec.x_cells_per_step)[0])
        nZ = len(grids.illiquid_grid(spec.zjump, spec.zmax, spec.z_cells_per_step))
        fits, _ = _plan_batching(nX, nZ, args.theta_batch, args.chunk, "cuda")
        if fits < args.theta_batch:
            raise SystemExit(
                f"theta_batch {args.theta_batch} with chunk {args.chunk} does not "
                f"fit in this GPU's free memory; only {fits} would. Pass "
                f"--theta_batch {fits} (or a smaller --chunk) explicitly, so the "
                f"configuration recorded with the shards is the one actually used."
            )
    if args.device == "cuda" and args.block % args.theta_batch:
        log.warning(
            f"block {args.block} is not a multiple of theta_batch "
            f"{args.theta_batch}: the last {args.block % args.theta_batch} draws "
            f"of every shard are solved in a smaller batch, which is a different "
            f"reduction regime. Prefer a theta_batch that divides the block."
        )
    n_blocks = ceil(args.n_samples / args.block)
    done = sum(1 for b in range(n_blocks) if (shard_dir / f"shard_{b:05d}.npz").exists())
    log.info(
        f"simulator={args.simulator} grid={args.grid} params={box.names} "
        f"waves={args.n_waves}x{args.wave_years}y from age {args.start_age}"
    )
    log.info(
        f"{args.n_samples} draws in {n_blocks} shards of {args.block}; "
        f"{done} already complete, {n_blocks - done} to go. Shards: {shard_dir}"
    )

    t_start = time.time()
    completed_now = 0
    for b in range(n_blocks):
        sf = shard_dir / f"shard_{b:05d}.npz"
        if sf.exists():
            continue
        lo, hi = b * args.block, min((b + 1) * args.block, args.n_samples)
        t0 = time.time()
        if args.device == "cuda":
            xb, ab = generate_block_gpu(
                theta_np[lo:hi], args.seed + lo + 1, args.start_age,
                args.n_waves, args.wave_years, args.grid, args.theta_batch,
                args.chunk,
            )
        else:
            results = Parallel(n_jobs=args.n_jobs, verbose=args.verbose)(
                delayed(fn)(
                    theta_np[i], args.seed + i + 1,
                    args.start_age, args.n_waves, args.wave_years, args.grid,
                )
                for i in range(lo, hi)
            )
            xb = np.stack([r[0] for r in results])
            ab = np.stack([r[1] for r in results])
        # Write to a temp name then rename, so an interrupted write cannot leave
        # a half-formed shard that a resume would trust. The temp name must
        # already end in .npz (np.savez appends the suffix otherwise, which
        # breaks the rename) and must not match the shard_*.npz glob.
        tmp = shard_dir / f".tmp_shard_{b:05d}.npz"
        np.savez(tmp, x=xb, alive=ab, lo=lo, hi=hi)
        tmp.rename(sf)

        completed_now += 1
        dt = time.time() - t0
        rate = (time.time() - t_start) / completed_now
        remaining = (n_blocks - done - completed_now) * rate
        log.info(
            f"shard {b + 1}/{n_blocks} ({lo}-{hi}) in {dt / 60:.1f} min "
            f"[{dt / (hi - lo):.1f} s/sample]  "
            f"ETA {remaining / 3600:.1f} h ({remaining / 86400:.2f} d)"
        )

    log.info(f"All shards complete in {(time.time() - t_start) / 3600:.2f} h")
    assemble(out, theta_np, log.info)


if __name__ == "__main__":
    main()
