"""Shards keep the annual panel so the observation window stays changeable.

The backward induction is ~99% of generation cost and never sees ``start_age``,
``n_waves`` or ``wave_years`` -- those are applied afterwards by
``aggregate_waves``. Storing only the aggregated tensor therefore bakes a
research-design choice into a 9-day run: extending the window from 10 to 20
years would mean re-solving every draw. Storing the panel costs ~1 MB per 256
draws and makes any window re-derivable.

These tests build shards directly, so they need neither a GPU nor a solve.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_dataset import PANEL_PREFIX, _panel_of, assemble  # noqa: E402
from hh_npe.data.dataset import load_dataset  # noqa: E402
from hh_npe.data.waves import FEATURES_TWOASSET, aggregate_waves  # noqa: E402
from hh_npe.simulator.dispatch import AGE_START_SIM  # noqa: E402

T = 71  # ages 20..90, the simulator's full range
N = 8


def _panel(seed=0):
    rng = np.random.default_rng(seed)
    p = {k: rng.normal(1e5, 3e4, (N, T)) for k in FEATURES_TWOASSET}
    p["t_age"] = np.tile(np.arange(T), (N, 1))
    return p


def _write_shard(shard_dir: Path, panel, window, with_panel=True, idx=0, lo=0):
    shard_dir.mkdir(parents=True, exist_ok=True)
    x, alive = aggregate_waves(
        panel, age_start_sim=AGE_START_SIM, features=FEATURES_TWOASSET, **window
    )
    extra = ({PANEL_PREFIX + k: v for k, v in panel.items()} if with_panel else {})
    np.savez(shard_dir / f"shard_{idx:05d}.npz",
             x=x, alive=alive, lo=lo, hi=lo + N, **extra)
    return x, alive


BASE = {"start_age": 30, "n_waves": 5, "wave_years": 2}


def test_panel_round_trips_through_the_npz(tmp_path):
    panel = _panel()
    _write_shard(tmp_path / "d_shards", panel, BASE)
    loaded = _panel_of(np.load(tmp_path / "d_shards" / "shard_00000.npz"))
    assert set(loaded) == set(panel)
    for k, v in panel.items():
        np.testing.assert_array_equal(loaded[k], v)


def test_reaggregating_the_same_window_reproduces_stored_x(tmp_path):
    """The re-derived tensor must be the stored one, not merely close to it."""
    out = tmp_path / "d.pt"
    panel = _panel()
    x_stored, _ = _write_shard(tmp_path / "d_shards", panel, BASE)

    assert assemble(out, np.zeros((N, 3), dtype=np.float32), lambda *_: None, BASE)
    _theta, x = load_dataset(out)
    np.testing.assert_array_equal(x.numpy(), x_stored)


def test_window_can_be_extended_without_resolving(tmp_path):
    """20 years from a run generated at 10, and the first 5 waves are unchanged."""
    out = tmp_path / "d.pt"
    panel = _panel()
    x_stored, _ = _write_shard(tmp_path / "d_shards", panel, BASE)

    wide = BASE | {"n_waves": 10}
    assert assemble(out, np.zeros((N, 3), dtype=np.float32), lambda *_: None, wide)
    _theta, x = load_dataset(out)
    assert x.shape == (N, 10, 4)
    np.testing.assert_array_equal(x.numpy()[:, :5], x_stored)


@pytest.mark.parametrize("window", [
    {"start_age": 35, "n_waves": 5, "wave_years": 2},   # later start
    {"start_age": 30, "n_waves": 10, "wave_years": 1},  # annual, not biennial
])
def test_start_age_and_frequency_are_also_re_derivable(tmp_path, window):
    out = tmp_path / "d.pt"
    panel = _panel()
    _write_shard(tmp_path / "d_shards", panel, BASE)
    expected, _ = aggregate_waves(
        panel, age_start_sim=AGE_START_SIM, features=FEATURES_TWOASSET, **window
    )

    assert assemble(out, np.zeros((N, 3), dtype=np.float32), lambda *_: None, window)
    _theta, x = load_dataset(out)
    np.testing.assert_array_equal(x.numpy(), expected)


def test_panelless_shard_still_assembles_at_its_own_window(tmp_path):
    """Shards from before this change must keep working."""
    out = tmp_path / "d.pt"
    x_stored, _ = _write_shard(tmp_path / "d_shards", _panel(), BASE, with_panel=False)
    assert assemble(out, np.zeros((N, 3), dtype=np.float32), lambda *_: None, BASE)
    _theta, x = load_dataset(out)
    np.testing.assert_array_equal(x.numpy(), x_stored)


def test_panelless_shard_refuses_a_different_window(tmp_path):
    """Better to stop than to silently return the window that happens to be stored."""
    out = tmp_path / "d.pt"
    _write_shard(tmp_path / "d_shards", _panel(), BASE, with_panel=False)
    with pytest.raises(SystemExit, match="cannot be re-aggregated"):
        assemble(out, np.zeros((N, 3), dtype=np.float32), lambda *_: None,
                 BASE | {"n_waves": 10})


def test_survival_filter_uses_the_widened_window(tmp_path):
    """A rebirth outside the old window but inside the new one must be caught.

    The two-asset forward pass never replaces a household, so this is dead code
    for Phase 3 today -- but ``assemble`` applies the filter to whatever window
    it built, and that has to be the new one, not the stored one.
    """
    out = tmp_path / "d.pt"
    panel = _panel()
    panel["t_age"] = np.tile(np.arange(T), (N, 1))
    panel["t_age"][0, 25:] = np.arange(T - 25)  # rebirth at age 45, wave 7
    _write_shard(tmp_path / "d_shards", panel, BASE)

    assert assemble(out, np.zeros((N, 3), dtype=np.float32), lambda *_: None,
                    BASE | {"n_waves": 10})
    _theta, x = load_dataset(out)
    assert x.shape[0] == N - 1, "household reborn inside the wider window not dropped"
