"""Does a random observation window beat a fixed one, and does augmentation help?

Three models on the same panels, differing only in how windows are cut:

    A  fixed start age 30, one window per panel, no age channel  (today's design)
    B  start ~ U{25..40}, one window per panel, age channel
    C  start ~ U{25..40}, k windows per panel, age channel

A -> B isolates randomization; B -> C isolates augmentation.

All three are scored on one shared held-out set of **random-age** windows,
because that is the distribution a PSID extract actually presents. A is also
scored on fixed-age-30 windows -- its home turf -- so the comparison shows both
what randomization buys and what, if anything, it costs at the original design
point.

Held-out panels are disjoint from training panels, and training splits by panel
too (see hh_npe.npe.train._use_grouped_split): with k windows per household, a
row-wise split would validate on households the model trained on.

CPU only; the GPU stays with dataset generation.

Usage::

    uv run python scripts/pilot_random_age.py
    uv run python scripts/pilot_random_age.py --k 8 --train_panels 8192
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from hh_npe.data.windows import build_windowed
from hh_npe.evaluation.scoring import estimation_scores
from hh_npe.npe.embedder import TrajectoryTransformer
from hh_npe.npe.prior import PHASE3, sample_sobol
from hh_npe.npe.train import train_npe
from hh_npe.utils.seeding import seed_all

log = logging.getLogger("pilot_random_age")

EMBEDDER = dict(d_model=64, n_heads=4, n_layers=2, output_dim=32)
TRAINING = dict(flow="nsf", max_num_epochs=200, stop_after_epochs=20,
                learning_rate=5e-4, batch_size=256, validation_fraction=0.1)


def _split_shards(shard_files: list[Path], train_panels: int):
    """Shards below the panel cutoff train; the rest are held out."""
    train, held = [], []
    for sf in shard_files:
        lo = int(np.load(sf)["lo"])
        (train if lo < train_panels else held).append(sf)
    if not train or not held:
        raise SystemExit(
            f"need shards on both sides of panel {train_panels}; got "
            f"{len(train)} train and {len(held)} held-out"
        )
    return train, held


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", type=Path,
                   default=Path("data/processed/phase3_dataset_shards"))
    p.add_argument("--n_total", type=int, default=65536)
    p.add_argument("--train_panels", type=int, default=8192,
                   help="Sobol power-of-2 prefix used for training.")
    p.add_argument("--n_waves", type=int, default=10)
    p.add_argument("--start_low", type=int, default=25)
    p.add_argument("--start_high", type=int, default=40)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--fixed_start", type=int, default=30)
    p.add_argument("--n_eval", type=int, default=1024)
    p.add_argument("--n_post", type=int, default=400)
    p.add_argument("--out", type=Path, default=Path("outputs/pilot_random_age"))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    # Frozen once: generation is still writing shards, and every model must see
    # the same data or the comparison measures the wrong thing.
    shard_files = sorted(args.shards.glob("shard_*.npz"))
    train_sh, held_sh = _split_shards(shard_files, args.train_panels)
    log.info(f"{len(shard_files)} shards frozen: {len(train_sh)} train, "
             f"{len(held_sh)} held out")

    theta_all = sample_sobol(args.n_total, PHASE3, seed=0)
    common = dict(n_waves=args.n_waves, wave_years=2)
    rand = dict(start_low=args.start_low, start_high=args.start_high, **common)

    # Two evaluation sets, both from held-out panels. Seeds differ from the
    # training seed so held-out windows are not the same cuts by coincidence.
    ev_random = build_windowed(held_sh, theta_all, k=1, seed=999,
                               with_age=True, **rand)
    ev_fixed = build_windowed(held_sh, theta_all, fixed_start=args.fixed_start,
                              with_age=True, **common)
    ev_random_noage = build_windowed(held_sh, theta_all, k=1, seed=999,
                                     with_age=False, **rand)
    ev_fixed_noage = build_windowed(held_sh, theta_all,
                                    fixed_start=args.fixed_start,
                                    with_age=False, **common)

    def cut(triple, n):
        return triple[0][:n], triple[1][:n]

    models = {
        "A_fixed": dict(
            train=build_windowed(train_sh, theta_all,
                                 fixed_start=args.fixed_start, with_age=False,
                                 **common),
            evals={"random_age": cut(ev_random_noage, args.n_eval),
                   "fixed_30": cut(ev_fixed_noage, args.n_eval)},
        ),
        "B_random": dict(
            train=build_windowed(train_sh, theta_all, k=1, seed=0,
                                 with_age=True, **rand),
            evals={"random_age": cut(ev_random, args.n_eval),
                   "fixed_30": cut(ev_fixed, args.n_eval)},
        ),
        "C_random_aug": dict(
            train=build_windowed(train_sh, theta_all, k=args.k, seed=0,
                                 with_age=True, **rand),
            evals={"random_age": cut(ev_random, args.n_eval),
                   "fixed_30": cut(ev_fixed, args.n_eval)},
        ),
    }

    results = {}
    for name, spec in models.items():
        th, x, pid = spec["train"]
        n_panels = len(pid.unique())
        log.info(f"=== {name}: {len(th)} windows from {n_panels} panels, "
                 f"x {tuple(x.shape)} ===")

        seed_all(0)
        embedder = TrajectoryTransformer(
            n_features=x.shape[-1], seq_len=args.n_waves,
            feature_mean=x.mean(dim=(0, 1)), feature_std=x.std(dim=(0, 1)),
            **EMBEDDER,
        )
        post, _de, _inf = train_npe(
            th, x, embedder=embedder, box=PHASE3, device="cpu",
            group_ids=pid, **TRAINING,
        )

        entry = {"n_windows": len(th), "n_panels": n_panels,
                 "n_features": x.shape[-1], "evals": {}}
        for ev_name, (th_ev, x_ev) in spec["evals"].items():
            torch.manual_seed(0)
            per_param, log_q = estimation_scores(post, PHASE3, th_ev, x_ev,
                                                 n_post=args.n_post)
            entry["evals"][ev_name] = {"per_param": per_param,
                                       "log_q": log_q, "n": len(th_ev)}
            log.info(f"  {name} on {ev_name}: log q = {log_q:.3f}")
        results[name] = entry

    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    _report(results)


def _report(results: dict) -> None:
    names = list(results)
    hdr = "".join(f"{n:>16s}" for n in names)
    print(f"\n{'':12s}{hdr}")
    print(f"{'windows':12s}" + "".join(f"{results[n]['n_windows']:16d}"
                                       for n in names))
    print(f"{'panels':12s}" + "".join(f"{results[n]['n_panels']:16d}"
                                      for n in names))

    for ev in ("random_age", "fixed_30"):
        print(f"\n########## evaluated on {ev} ##########")
        for metric in ("contraction", "corr", "mae"):
            print(f"\n=== {metric} ===")
            for pname in PHASE3.names:
                row = "".join(
                    f"{results[n]['evals'][ev]['per_param'][pname][metric]:16.4f}"
                    for n in names
                )
                print(f"{pname:12s}{row}")
        print(f"\n=== held-out log q ===")
        print(f"{'':12s}" + "".join(f"{results[n]['evals'][ev]['log_q']:16.3f}"
                                    for n in names))


if __name__ == "__main__":
    main()
