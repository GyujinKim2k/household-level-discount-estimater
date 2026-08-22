"""Compare observation windows on both estimation *and* calibration.

Shards store the annual panel, so the observation window is a post-hoc
aggregation choice: 5, 10 and 15 biennial waves are three views of one dataset,
not three datasets. This trains one NPE per window on identical draws and then
asks two different questions of each:

**Estimation** -- posterior contraction, correlation between truth and
posterior mean, mean absolute error, and held-out ``log q(theta_true | x)``,
all on draws the training prefix never saw.

**Calibration** -- SBC rank uniformity (Talts et al. 2018) and empirical
coverage of the 90% credible interval. A posterior can contract beautifully and
still be wrong; contraction without calibration is confidence, not accuracy,
and only the second question catches a posterior that is confidently mistaken.

The SBC simulations are run **once** and shared across windows -- the same
stored-panel trick that makes the comparison cheap in the first place. They use
i.i.d. prior draws rather than the dataset's Sobol points, because SBC's rank
argument assumes i.i.d. sampling from the prior and a low-discrepancy sequence
is deliberately not that.

Usage::

    uv run python scripts/compare_windows.py --windows 5 10 15
    uv run python scripts/compare_windows.py --n_sbc 200 --train_n 8192
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from hh_npe.data.dataset import read_solver_config
from hh_npe.data.windows import build_windowed, max_start_age, window_panel
from hh_npe.evaluation.scoring import calibration_scores, estimation_scores
from hh_npe.npe.embedder import TrajectoryTransformer
from hh_npe.npe.prior import PHASE3, make_sbi_prior, sample_sobol
from hh_npe.npe.train import save_posterior, train_npe
from hh_npe.simulator.dispatch import simulate_batch_twoasset_gpu
from hh_npe.utils.seeding import seed_all

log = logging.getLogger("compare_windows")

WAVE_YEARS = 2
AGE_RETIRE = 64  # laibson_calibration.AGE_RETIRE, for the end-age warning
EMBEDDER = dict(d_model=64, n_heads=4, n_layers=2, output_dim=32)
TRAINING = dict(flow="nsf", max_num_epochs=200, stop_after_epochs=20,
                learning_rate=5e-4, batch_size=256, validation_fraction=0.1)


def split_shards(shard_files: list[Path], train_n: int):
    """Shards below the panel cutoff train; the rest are held out."""
    train, held = [], []
    for sf in shard_files:
        (train if int(np.load(sf)["lo"]) < train_n else held).append(sf)
    if not train or not held:
        raise SystemExit(
            f"need shards on both sides of panel {train_n}; got {len(train)} "
            f"train and {len(held)} held out"
        )
    return train, held


def simulate_sbc_once(n_sbc: int, seed: int, solver_config: dict):
    """One GPU pass; the panels are re-windowed per window afterwards."""
    prior = make_sbi_prior(PHASE3)
    torch.manual_seed(seed)
    thetas = prior.sample((n_sbc,))
    t0 = time.time()
    # n_waves here only sizes the throwaway `x`; the panels are what we keep,
    # and they get windowed per arm afterwards.
    _x, _alive, panels = simulate_batch_twoasset_gpu(
        thetas.numpy(), seed, 30, n_waves=5, wave_years=WAVE_YEARS,
        grid=solver_config.get("grid", "full"),
        theta_batch=solver_config["theta_batch"],
        chunk=solver_config["chunk"],
        return_panels=True,
    )
    log.info(f"SBC simulations done in {(time.time() - t0) / 3600:.2f} h")
    return thetas, panels


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--windows", type=int, nargs="+", default=[5, 10, 15])
    p.add_argument("--shards", type=Path,
                   default=Path("data/processed/phase3_dataset_shards"))
    p.add_argument("--dataset", type=Path,
                   default=Path("data/processed/phase3_dataset.pt"),
                   help="Only used to locate the recorded solver config.")
    p.add_argument("--train_n", type=int, default=65536,
                   help="Draws used for training; the rest are held out. Sobol "
                        "keeps its balance on power-of-2 prefixes.")
    p.add_argument("--n_total", type=int, default=65536)
    p.add_argument("--n_sbc", type=int, default=1000,
                   help="SBC simulations. These are fresh GPU solves and the "
                        "dominant cost: ~12.4 s each, shared across windows.")
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--n_heldout_eval", type=int, default=2048,
                   help="Held-out draws scored for estimation metrics.")
    p.add_argument("--start_low", type=int, default=25)
    p.add_argument("--start_high", type=int, default=40,
                   help="Start ages are drawn from [low, high] for every arm, "
                        "so the age distribution is identical and only the "
                        "window length differs. Note this makes the longer "
                        "arms reach further past retirement -- reported.")
    p.add_argument("--k", type=int, default=8,
                   help="Windows per panel (augmentation).")
    p.add_argument("--no_age", action="store_true",
                   help="Drop the per-wave age channel.")
    p.add_argument("--out", type=Path, default=Path("outputs/window_comparison"))
    p.add_argument("--skip_sbc", action="store_true",
                   help="Estimation metrics only; skips the GPU simulations.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    # Frozen once: generation may still be running and writing new shards, and
    # every window must see the same data for the comparison to mean anything.
    shard_files = sorted(args.shards.glob("shard_*.npz"))
    if not shard_files:
        raise SystemExit(f"no shards in {args.shards}")
    log.info(f"Shard list frozen at {len(shard_files)} shards "
             f"({shard_files[0].name}..{shard_files[-1].name})")

    sbc_thetas = sbc_panels = None
    if not args.skip_sbc:
        cfg = read_solver_config(args.dataset)
        if not cfg or cfg.get("device") != "cuda":
            raise SystemExit(
                "No CUDA solver config beside the dataset. SBC must reproduce "
                "the training set's simulator exactly; refusing to guess."
            )
        log.info(f"Simulating {args.n_sbc} SBC draws once, shared across "
                 f"windows (solver config: {cfg})")
        sbc_thetas, sbc_panels = simulate_sbc_once(args.n_sbc, 20260822, cfg)

    train_sh, held_sh = split_shards(shard_files, args.train_n)
    theta_all = sample_sobol(args.n_total, PHASE3, seed=0)
    win = dict(start_low=args.start_low, start_high=args.start_high,
               wave_years=WAVE_YEARS, with_age=not args.no_age)

    results = {}
    for k in args.windows:
        limit = max_start_age(k, WAVE_YEARS)
        if args.start_high > limit:
            raise SystemExit(
                f"{k} waves cannot start as late as {args.start_high}; the "
                f"panel runs out at {limit}."
            )
        end_lo = args.start_low + WAVE_YEARS * k - 1
        end_hi = args.start_high + WAVE_YEARS * k - 1
        ages = f"{args.start_low}-{args.start_high} start, ends {end_lo}-{end_hi}"

        th_tr, x_tr, pid_tr = build_windowed(
            train_sh, theta_all, k=args.k, n_waves=k, seed=0, **win
        )
        th_ho, x_ho, _pid = build_windowed(
            held_sh, theta_all, k=1, n_waves=k, seed=999, **win
        )
        th_ho, x_ho = th_ho[: args.n_heldout_eval], x_ho[: args.n_heldout_eval]
        n_panels = len(pid_tr.unique())
        log.info(f"=== {k} waves ({ages}) | train {len(th_tr)} windows from "
                 f"{n_panels} panels | held-out scored {len(th_ho)} ===")
        if end_hi > AGE_RETIRE:
            # Stated per arm rather than assumed: the forward pass applies no
            # mortality, so windows reaching past retirement describe a cohort
            # in which everyone survives. That flatters the longer arms.
            log.warning(
                f"  {k}w windows reach age {end_hi}, past retirement at "
                f"{AGE_RETIRE}. The forward pass has no mortality, so this arm "
                f"gets a survivorship advantage the shorter arms do not."
            )

        seed_all(0)  # identical init and grouped split for every arm
        embedder = TrajectoryTransformer(
            n_features=x_tr.shape[-1], seq_len=k,
            feature_mean=x_tr.mean(dim=(0, 1)), feature_std=x_tr.std(dim=(0, 1)),
            **EMBEDDER,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        post, _de, _inf = train_npe(
            th_tr, x_tr, embedder=embedder, box=PHASE3, device=device,
            group_ids=pid_tr, **TRAINING
        )
        save_posterior(post, embedder, PHASE3, args.out / f"posterior_{k}w.pt")

        torch.manual_seed(0)
        per_param, log_q = estimation_scores(post, PHASE3, th_ho, x_ho, n_post=400)
        entry = {"ages": ages, "n_train": len(th_tr), "n_panels": n_panels,
                 "n_heldout": len(th_ho), "ends_past_retirement": end_hi > AGE_RETIRE,
                 "estimation": per_param, "held_out_log_q": log_q}

        if sbc_panels is not None:
            # One window per SBC draw, cut the way a training window was.
            th_sbc, x_sbc, ids = window_panel(
                sbc_panels, sbc_thetas.numpy(), k=1, n_waves=k, seed=4242, **win
            )
            entry["calibration"] = calibration_scores(
                post, PHASE3, th_sbc, x_sbc,
                n_post=args.n_post, out_dir=args.out, tag=f"{k}w",
            )
            entry["n_sbc"] = len(th_sbc)
        results[k] = entry
        log.info(f"  {k}w done: log q = {log_q:.3f}")

    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    _report(results, args.windows)


def _report(results: dict, windows: list[int]) -> None:
    hdr = "".join(f"{f'{k}w':>12s}" for k in windows)
    print(f"\n{'':10s}{hdr}")
    print(f"{'ends':10s}" + "".join(
        f"{results[k]['ages'].split('ends ')[-1]:>12s}" for k in windows))
    print(f"{'windows':10s}" + "".join(f"{results[k]['n_train']:12d}"
                                       for k in windows))
    print(f"{'panels':10s}" + "".join(f"{results[k]['n_panels']:12d}"
                                      for k in windows))

    for metric, fmt in (("contraction", "12.3f"), ("corr", "12.3f"),
                        ("mae", "12.4f")):
        print(f"\n=== {metric} ===")
        for n in PHASE3.names:
            row = "".join(
                f"{results[k]['estimation'][n][metric]:{fmt}}" for k in windows
            )
            print(f"{n:10s}{row}")

    print("\n=== held-out log q(theta_true | x) ===")
    print(f"{'':10s}" + "".join(f"{results[k]['held_out_log_q']:12.3f}"
                                for k in windows))

    flagged = [k for k in windows if results[k]["ends_past_retirement"]]
    if flagged:
        print(f"\nNOTE: arms {flagged} reach past retirement at {AGE_RETIRE}. "
              f"The forward pass applies no mortality, so those windows train\n"
              f"      on a cohort where everyone survives -- an advantage the "
              f"shorter arms do not get. Discount accordingly.")
    print("\nEffective independent sample is the panel count, not the window "
          "count: augmentation teaches\nthe age mapping, it adds no information "
          "about theta.")

    if "calibration" not in results[windows[0]]:
        return
    for metric, fmt, target in (("coverage_90", "12.3f", " (target 0.900)"),
                                ("ks_p", "12.4f", " (>0.05 = uniform)")):
        print(f"\n=== SBC {metric}{target} ===")
        for n in PHASE3.names:
            row = "".join(
                f"{results[k]['calibration'][n][metric]:{fmt}}" for k in windows
            )
            print(f"{n:10s}{row}")


if __name__ == "__main__":
    main()
