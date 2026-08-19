"""SBC must simulate through the simulator that made the training set.

For the two-asset model that is not a formality. Its grids are built from round
dollar amounts, so ~99% of states hold two ``(X', Z')`` choices leaving
bitwise-identical cash on hand; where their continuation values tie as well, the
CPU and GPU solvers pick different -- equally optimal -- portfolios. Both
reproduce Laibson et al.'s table 3 (fidelity 0.0102 and 0.0104), yet across six
draws the difference moved mean liquid assets by 25%. SBC run on the CPU against
a GPU-generated training set would report miscalibration caused by that
mismatch rather than by the posterior, which is exactly the kind of false signal
that costs days to chase.

These tests need no GPU: they check the routing decision, not the solve.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hh_npe.data.dataset import read_solver_config, shard_dir  # noqa: E402


def test_shard_dir_matches_generator_layout():
    assert shard_dir("data/processed/phase3_dataset.pt") == Path(
        "data/processed/phase3_dataset_shards"
    )


def test_read_solver_config_absent_is_none(tmp_path):
    assert read_solver_config(tmp_path / "nothing.pt") is None


def test_read_solver_config_roundtrip(tmp_path):
    ds = tmp_path / "d.pt"
    sd = shard_dir(ds)
    sd.mkdir()
    cfg = {"device": "cuda", "theta_batch": 16, "chunk": 16, "grid": "full"}
    (sd / "solver_config.json").write_text(json.dumps(cfg))
    assert read_solver_config(ds) == cfg


def test_cuda_config_routes_to_gpu_path(monkeypatch):
    """A recorded CUDA config must take the batched GPU path, not joblib."""
    import run_sbc

    seen = {}

    def fake_gpu(thetas, seed_base, start_age, n_waves, wave_years, **kw):
        seen.update(kw | {"n": len(thetas), "seed_base": seed_base})
        return np.zeros((len(thetas), n_waves, 4)), np.ones((len(thetas), n_waves), bool)

    def fail_cpu(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("CPU path used for a CUDA-generated training set")

    monkeypatch.setattr(run_sbc, "simulate_batch_twoasset_gpu", fake_gpu)
    monkeypatch.setattr(run_sbc, "Parallel", fail_cpu)

    x = run_sbc.simulate_for_sbc(
        torch.zeros(4, 3), start_age=30, n_waves=5, wave_years=2, seed_base=7,
        solver_config={"device": "cuda", "theta_batch": 16, "chunk": 16,
                       "grid": "full"},
    )
    assert x.shape == (4, 5, 4)
    # The training set's settings, not SBC's own defaults.
    assert seen["theta_batch"] == 16 and seen["chunk"] == 16 and seen["grid"] == "full"


def test_no_config_falls_back_to_cpu(monkeypatch):
    """Without a recorded config the CPU path still runs (Phases 1-2 datasets)."""
    import run_sbc

    def fail_gpu(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("GPU path used without a CUDA config")

    monkeypatch.setattr(run_sbc, "simulate_batch_twoasset_gpu", fail_gpu)
    monkeypatch.setattr(
        run_sbc, "Parallel",
        lambda **k: (lambda jobs: [(np.zeros((5, 3)), np.ones(5, bool)) for _ in jobs]),
    )
    x = run_sbc.simulate_for_sbc(
        torch.zeros(2, 2), start_age=30, n_waves=5, wave_years=2, seed_base=0,
        simulator="hark", solver_config=None,
    )
    assert x.shape == (2, 5, 3)


@pytest.mark.parametrize("device", ["cpu", None])
def test_non_cuda_config_uses_cpu_path(device, monkeypatch):
    import run_sbc

    monkeypatch.setattr(
        run_sbc, "simulate_batch_twoasset_gpu",
        lambda *a, **k: pytest.fail("GPU path used for a non-CUDA config"),
    )
    monkeypatch.setattr(
        run_sbc, "Parallel",
        lambda **k: (lambda jobs: [(np.zeros((5, 4)), np.ones(5, bool)) for _ in jobs]),
    )
    cfg = None if device is None else {"device": device}
    x = run_sbc.simulate_for_sbc(
        torch.zeros(3, 3), start_age=30, n_waves=5, wave_years=2, seed_base=0,
        solver_config=cfg,
    )
    assert x.shape == (3, 5, 4)
