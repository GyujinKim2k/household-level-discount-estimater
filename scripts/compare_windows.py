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
from scipy import stats

from hh_npe.data.dataset import read_solver_config
from hh_npe.data.waves import FEATURES_TWOASSET, aggregate_waves
from hh_npe.evaluation.sbc import compute_ranks, coverage_at_level, plot_sbc_ranks
from hh_npe.npe.embedder import TrajectoryTransformer
from hh_npe.npe.prior import PHASE3, make_sbi_prior, sample_sobol
from hh_npe.npe.train import save_posterior, train_npe
from hh_npe.simulator.dispatch import AGE_START_SIM, simulate_batch_twoasset_gpu
from hh_npe.utils.seeding import seed_all

log = logging.getLogger("compare_windows")

START_AGE, WAVE_YEARS = 30, 2
EMBEDDER = dict(d_model=64, n_heads=4, n_layers=2, output_dim=32)
TRAINING = dict(flow="nsf", max_num_epochs=200, stop_after_epochs=20,
                learning_rate=5e-4, batch_size=256, validation_fraction=0.1)


def _windowed(panel: dict, n_waves: int):
    return aggregate_waves(
        panel, age_start_sim=AGE_START_SIM, start_age=START_AGE,
        n_waves=n_waves, wave_years=WAVE_YEARS, features=FEATURES_TWOASSET,
    )


def load_split(shard_files: list[Path], n_waves: int, train_n: int, n_total: int):
    """(train, held-out) at this window, re-aggregated from the stored panels."""
    theta_all = sample_sobol(n_total, PHASE3, seed=0)
    split: dict[str, tuple[list, list]] = {"train": ([], []), "heldout": ([], [])}
    for sf in shard_files:
        d = np.load(sf)
        lo, hi = int(d["lo"]), int(d["hi"])
        panel = {k[6:]: d[k] for k in d.files if k.startswith("panel_")}
        if not panel:
            raise SystemExit(
                f"{sf.name} stores no annual panel, so the window cannot be "
                f"changed without re-solving. Only runs generated after the "
                f"panel change support this comparison."
            )
        x, alive = _windowed(panel, n_waves)
        keep = alive.all(axis=1)
        bucket = "train" if lo < train_n else "heldout"
        split[bucket][0].append(theta_all[lo:hi][keep])
        split[bucket][1].append(x[keep])

    out = {}
    for k, (ths, xs) in split.items():
        if not xs:
            raise SystemExit(f"no {k} shards available")
        out[k] = (torch.from_numpy(np.concatenate(ths)).float(),
                  torch.from_numpy(np.concatenate(xs)).float())
    return out["train"], out["heldout"]


def estimation_scores(post, theta, x, n_post: int):
    prior_var = ((PHASE3.high - PHASE3.low) ** 2) / 12.0
    means, sds, lps = [], [], []
    for i in range(len(theta)):
        s = post.sample((n_post,), x=x[i], show_progress_bars=False)
        means.append(s.mean(0))
        sds.append(s.std(0))
        lps.append(post.log_prob(theta[i][None], x=x[i]).item())
    means, sds = torch.stack(means).numpy(), torch.stack(sds).numpy()
    truth = theta.numpy()
    per_param = {
        n: {
            "contraction": float(1 - (sds[:, j] ** 2).mean() / prior_var[j]),
            "corr": float(np.corrcoef(truth[:, j], means[:, j])[0, 1]),
            "mae": float(np.abs(means[:, j] - truth[:, j]).mean()),
        }
        for j, n in enumerate(PHASE3.names)
    }
    return per_param, float(np.mean(lps))


def calibration_scores(post, theta, x, n_post: int, out_dir: Path, tag: str):
    """SBC ranks -> uniformity p-value and 90% interval coverage."""
    ranks = compute_ranks(post, theta, x, n_posterior_samples=n_post)
    cov = coverage_at_level(ranks, n_post, level=0.9)
    per_param = {}
    for j, n in enumerate(PHASE3.names):
        # Ranks are uniform on {0..n_post} under calibration; compare the
        # normalized ranks to Uniform(0,1). A small p-value means miscalibrated.
        u = (ranks[:, j] + 0.5) / (n_post + 1)
        ks = stats.kstest(u, "uniform")
        per_param[n] = {"coverage_90": float(cov[j]),
                        "ks_p": float(ks.pvalue),
                        "ks_stat": float(ks.statistic)}
    plot_sbc_ranks(ranks, n_post, list(PHASE3.names), out_dir / f"sbc_ranks_{tag}.png")
    np.savez(out_dir / f"sbc_ranks_{tag}.npz", ranks=ranks, coverage_90=cov)
    return per_param


def simulate_sbc_once(n_sbc: int, seed: int, solver_config: dict):
    """One GPU pass; the panels are re-windowed per window afterwards."""
    prior = make_sbi_prior(PHASE3)
    torch.manual_seed(seed)
    thetas = prior.sample((n_sbc,))
    t0 = time.time()
    _x, _alive, panels = simulate_batch_twoasset_gpu(
        thetas.numpy(), seed, START_AGE, n_waves=5, wave_years=WAVE_YEARS,
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

    results = {}
    for k in args.windows:
        ages = f"{START_AGE}-{START_AGE + WAVE_YEARS * k - 1}"
        (th_tr, x_tr), (th_ho, x_ho) = load_split(
            shard_files, k, args.train_n, args.n_total
        )
        th_ho, x_ho = th_ho[: args.n_heldout_eval], x_ho[: args.n_heldout_eval]
        log.info(f"=== {k} waves (ages {ages}) | train {len(th_tr)} | "
                 f"held-out scored {len(th_ho)} ===")

        seed_all(0)  # identical init and validation split for every window
        embedder = TrajectoryTransformer(
            n_features=x_tr.shape[-1], seq_len=k,
            feature_mean=x_tr.mean(dim=(0, 1)), feature_std=x_tr.std(dim=(0, 1)),
            **EMBEDDER,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        post, _de, _inf = train_npe(
            th_tr, x_tr, embedder=embedder, box=PHASE3, device=device, **TRAINING
        )
        save_posterior(post, embedder, PHASE3, args.out / f"posterior_{k}w.pt")

        torch.manual_seed(0)
        per_param, log_q = estimation_scores(post, th_ho, x_ho, 400)
        entry = {"ages": ages, "n_train": len(th_tr), "n_heldout": len(th_ho),
                 "estimation": per_param, "held_out_log_q": log_q}

        if sbc_panels is not None:
            x_sbc, alive_sbc = _windowed(sbc_panels, k)
            keep = alive_sbc.all(axis=1)
            entry["calibration"] = calibration_scores(
                post, sbc_thetas[keep], torch.from_numpy(x_sbc[keep]).float(),
                args.n_post, args.out, f"{k}w",
            )
            entry["n_sbc"] = int(keep.sum())
        results[k] = entry
        log.info(f"  {k}w done: log q = {log_q:.3f}")

    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    _report(results, args.windows)


def _report(results: dict, windows: list[int]) -> None:
    hdr = "".join(f"{f'{k}w':>12s}" for k in windows)
    ages = "".join(f"{results[k]['ages']:>12s}" for k in windows)
    print(f"\n{'':10s}{hdr}\n{'ages':10s}{ages}")

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
