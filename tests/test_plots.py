"""Contour figures: the enclosed-mass levels have to actually enclose that mass.

The rest of the plotting is cosmetic and not worth testing, but `_hpd_levels`
is real logic -- it decides what "68%" means on the figure. If it drifts, every
posterior in the writeup is drawn with mislabelled credible regions, and
nothing else would catch it.
"""

import numpy as np
import pytest

from hh_npe.evaluation.plots import _hpd_levels, contour_corner
from hh_npe.npe.prior import PHASE3


def test_levels_enclose_the_requested_mass():
    """On a Gaussian grid, the level's enclosed mass must match its label."""
    g = np.linspace(-4, 4, 401)
    X, Y = np.meshgrid(g, g)
    dens = np.exp(-(X**2 + Y**2) / 2)
    lo, hi = _hpd_levels(dens, probs=(0.68, 0.95))
    total = dens.sum()
    assert dens[dens >= hi].sum() / total == pytest.approx(0.68, abs=0.01)
    assert dens[dens >= lo].sum() / total == pytest.approx(0.95, abs=0.01)


def test_levels_are_ordered_outer_first():
    """contour_corner relies on [95%, 68%] ordering to draw fill under line."""
    g = np.linspace(-4, 4, 201)
    X, Y = np.meshgrid(g, g)
    lv = _hpd_levels(np.exp(-(X**2 + Y**2) / 2), probs=(0.68, 0.95))
    assert lv[0] < lv[1], "wider region must have the lower density level"


def test_corner_writes_a_file_for_three_parameters(tmp_path):
    rng = np.random.default_rng(0)
    n = 800
    s = np.column_stack([
        rng.uniform(0.4, 0.9, n), rng.uniform(0.9, 0.99, n),
        rng.uniform(1.0, 3.0, n),
    ])
    out = tmp_path / "c.png"
    contour_corner({"a": s, "b": s + 0.01}, PHASE3,
                   truth={"a": s.mean(0), "b": s.mean(0)}, path=out)
    assert out.exists() and out.stat().st_size > 5000


def test_degenerate_series_does_not_abort_the_figure(tmp_path):
    """A collapsed posterior has singular covariance; skip it, don't crash."""
    rng = np.random.default_rng(0)
    good = np.column_stack([rng.uniform(0.4, 0.9, 400),
                            rng.uniform(0.9, 0.99, 400),
                            rng.uniform(1.0, 3.0, 400)])
    point = np.tile([0.5, 0.95, 2.0], (400, 1))
    out = tmp_path / "d.png"
    contour_corner({"ok": good, "collapsed": point}, PHASE3, path=out)
    assert out.exists()
