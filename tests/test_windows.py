"""Random-age windows must be exact cuts of the panel, and grouped by household.

Two properties carry the design. A window at start age ``s`` has to be
*identical* to what ``aggregate_waves`` produces at that age -- the builder is
plumbing, not a second implementation. And every row must carry the panel it
came from, because several rows share one household and a split that separates
them silently inflates every number downstream.

No GPU and no solve: panels are synthetic, as in test_panel_shards.py.
"""

import numpy as np
import pytest
import torch

from hh_npe.data.waves import (
    FEATURES_TWOASSET,
    FEATURES_TWOASSET_AGE,
    aggregate_waves,
)
from hh_npe.data.windows import (
    AGE_START_SIM,
    PANEL_PREFIX,
    build_windowed,
    load_windowed,
    max_start_age,
    sample_start_ages,
    save_windowed,
)

T = 71  # ages 20..90
N = 16


def _panel(seed=0):
    rng = np.random.default_rng(seed)
    p = {k: rng.normal(1e5, 3e4, (N, T)) for k in FEATURES_TWOASSET}
    p["t_age"] = np.tile(np.arange(T), (N, 1))
    return p


def _shard(tmp_path, panel, lo=0, idx=0):
    d = tmp_path / "s_shards"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"shard_{idx:05d}.npz"
    np.savez(f, x=np.zeros((N, 5, 4), np.float32), alive=np.ones((N, 5), bool),
             lo=lo, hi=lo + N,
             **{PANEL_PREFIX + k: v for k, v in panel.items()})
    return f


def test_age_start_sim_matches_dispatch():
    """The two constants must agree or every window is shifted."""
    from hh_npe.simulator.dispatch import AGE_START_SIM as dispatch_age
    assert AGE_START_SIM == dispatch_age


def test_max_start_age_is_the_last_window_that_fits():
    assert max_start_age(n_waves=10, wave_years=2) == 71
    assert max_start_age(n_waves=15, wave_years=2) == 61
    panel = _panel()
    # The advertised limit must actually work, and one past it must not.
    aggregate_waves(panel, AGE_START_SIM, max_start_age(10, 2), 10, 2,
                    FEATURES_TWOASSET)
    with pytest.raises(ValueError, match="exceeds simulator range"):
        aggregate_waves(panel, AGE_START_SIM, max_start_age(10, 2) + 1, 10, 2,
                        FEATURES_TWOASSET)


def test_start_ages_are_in_range_and_distinct_within_a_panel():
    s = sample_start_ages(200, k=8, low=25, high=40, seed=0)
    assert s.shape == (200, 8)
    assert s.min() >= 25 and s.max() <= 40
    for row in s:
        assert len(set(row.tolist())) == 8, "a panel drew the same window twice"


def test_start_ages_refuse_impossible_k():
    with pytest.raises(ValueError, match="cannot draw"):
        sample_start_ages(4, k=20, low=25, high=40, seed=0)


def test_window_equals_aggregate_waves_at_that_start(tmp_path):
    """The builder must be plumbing, not a reimplementation."""
    panel = _panel()
    f = _shard(tmp_path, panel)
    theta_all = np.arange(N * 3, dtype=np.float64).reshape(N, 3)

    theta, x, pid = build_windowed([f], theta_all, fixed_start=33, n_waves=10,
                                   with_age=False)
    expected, _alive = aggregate_waves(panel, AGE_START_SIM, 33, 10, 2,
                                       FEATURES_TWOASSET)
    assert x.shape == (N, 10, 4)
    np.testing.assert_array_equal(x.numpy(), expected)
    np.testing.assert_array_equal(theta.numpy(), theta_all.astype(np.float32))
    np.testing.assert_array_equal(pid.numpy(), np.arange(N))


def test_age_channel_is_the_wave_age_and_leaves_others_untouched(tmp_path):
    panel = _panel()
    f = _shard(tmp_path, panel)
    theta_all = np.zeros((N, 3))

    _th, x_age, _ = build_windowed([f], theta_all, fixed_start=30, n_waves=10,
                                   with_age=True)
    _th, x_plain, _ = build_windowed([f], theta_all, fixed_start=30, n_waves=10,
                                     with_age=False)
    assert x_age.shape == (N, 10, 5) and x_plain.shape == (N, 10, 4)
    # Adding the channel must not disturb the dollar series.
    np.testing.assert_array_equal(x_age.numpy()[..., :4], x_plain.numpy())
    # age is a stock: read at the end of each 2-year wave, so 31, 33, ... 49.
    np.testing.assert_array_equal(
        x_age.numpy()[0, :, 4], np.arange(31, 51, 2, dtype=np.float32)
    )


def test_random_windows_produce_k_rows_per_panel(tmp_path):
    panel = _panel()
    f = _shard(tmp_path, panel)
    theta_all = np.zeros((N, 3))
    theta, x, pid = build_windowed([f], theta_all, start_low=25, start_high=40,
                                   k=8, n_waves=10, seed=0)
    assert len(theta) == N * 8
    counts = np.bincount(pid.numpy(), minlength=N)
    assert (counts == 8).all(), f"uneven windows per panel: {counts}"
    # Each panel's rows must differ -- they are different ages of one household.
    for p in range(N):
        rows = x[pid == p].numpy()
        assert len({r.tobytes() for r in rows}) == 8


def test_same_seed_same_dataset(tmp_path):
    panel = _panel()
    f = _shard(tmp_path, panel)
    theta_all = np.zeros((N, 3))
    a = build_windowed([f], theta_all, k=4, n_waves=10, seed=7)
    b = build_windowed([f], theta_all, k=4, n_waves=10, seed=7)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u.numpy(), v.numpy())


def test_panel_ids_are_global_across_shards(tmp_path):
    """A household's id must not restart at each shard."""
    panel = _panel()
    f0 = _shard(tmp_path, panel, lo=0, idx=0)
    f1 = _shard(tmp_path, _panel(1), lo=N, idx=1)
    theta_all = np.zeros((2 * N, 3))
    _th, _x, pid = build_windowed([f0, f1], theta_all, k=2, n_waves=10, seed=0)
    assert set(pid.tolist()) == set(range(2 * N))


def test_start_age_past_the_panel_is_refused(tmp_path):
    panel = _panel()
    f = _shard(tmp_path, panel)
    with pytest.raises(ValueError, match="runs past the simulated panel"):
        build_windowed([f], np.zeros((N, 3)), start_low=60, start_high=80,
                       k=2, n_waves=10)


def test_shard_without_panel_is_refused(tmp_path):
    d = tmp_path / "s_shards"
    d.mkdir(parents=True)
    f = d / "shard_00000.npz"
    np.savez(f, x=np.zeros((N, 5, 4), np.float32),
             alive=np.ones((N, 5), bool), lo=0, hi=N)
    with pytest.raises(SystemExit, match="no annual panel"):
        build_windowed([f], np.zeros((N, 3)), k=2, n_waves=10)


def test_windowed_dataset_round_trips(tmp_path):
    theta, x = torch.randn(6, 3), torch.randn(6, 10, 5)
    pid = torch.tensor([0, 0, 1, 1, 2, 2])
    meta = {"start_low": 25, "start_high": 40, "k": 2}
    save_windowed(theta, x, pid, meta, tmp_path / "w.pt")
    d = load_windowed(tmp_path / "w.pt")
    torch.testing.assert_close(d["theta"], theta)
    torch.testing.assert_close(d["x"], x)
    torch.testing.assert_close(d["panel_id"], pid)
    assert d["meta"] == meta


def test_window_panel_matches_the_shard_path(tmp_path):
    """SBC windows must be cut exactly the way training windows are.

    They come from an in-memory panel rather than a shard, so the two code
    paths could drift; a calibration set cut differently from the training set
    would report miscalibration that belongs to the mismatch.
    """
    from hh_npe.data.windows import window_panel

    panel = _panel()
    f = _shard(tmp_path, panel)
    theta_all = np.arange(N * 3, dtype=np.float64).reshape(N, 3)

    _th_a, x_a, _ = build_windowed([f], theta_all, k=3, n_waves=10, seed=5,
                                   start_low=25, start_high=40)
    _th_b, x_b, _ = window_panel(panel, theta_all, k=3, n_waves=10, seed=5,
                                 start_low=25, start_high=40)
    np.testing.assert_array_equal(x_a.numpy(), x_b.numpy())


def test_add_age_is_idempotent_and_derived():
    from hh_npe.data.windows import add_age

    panel = {"t_age": np.tile(np.arange(5), (2, 1))}
    once = add_age(panel)
    np.testing.assert_array_equal(once["age"], AGE_START_SIM + panel["t_age"])
    assert add_age(once)["age"] is once["age"]
    assert "age" not in panel, "add_age must not mutate its input"
