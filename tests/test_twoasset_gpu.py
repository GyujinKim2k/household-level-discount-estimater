"""The CUDA backend must reproduce the CPU reference solver.

The CPU solver is the one validated against Laibson et al.'s published
simulation, so it is the definition of correct here. These tests skip entirely
when no CUDA device is present, so the suite still passes on CPU-only machines.

One known, accepted difference: the GPU computes ``c**(1-rho)`` as
``exp((1-rho) * log c)`` so that a single logarithm serves a whole batch of
parameter draws. That differs from ``pow`` in the last bits, which can flip the
argmax where two choices are economically indistinguishable. At the Laibson
benchmark the policies agree exactly; at large rho a fraction of a percent of
states differ without changing any simulated panel. The panel-level tests below
are therefore the binding ones.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device"
)

from hh_npe.simulator import laibson_calibration as cal  # noqa: E402
from hh_npe.simulator.twoasset import ModelSpec, simulate, solve  # noqa: E402

TINY = ModelSpec(xjump=20000.0, x_cells_per_step=4, zjump=200000.0, z_cells_per_step=3)
BENCHMARK = cal.BENCHMARK_PREFS


@pytest.fixture(scope="module")
def gpu_solutions():
    from hh_npe.simulator.twoasset_gpu import solve_batch

    thetas = np.array([BENCHMARK, [1.0, 0.96, 1.4663]])
    return thetas, solve_batch(thetas, TINY, theta_batch=2)


def test_policies_match_cpu_at_benchmark(gpu_solutions):
    """At Laibson's own estimates the GPU must reproduce the CPU exactly."""
    thetas, sols = gpu_solutions
    cpu = solve(*thetas[0], TINY)
    g = sols[0]
    ok = cpu.solvable & cpu.feasible[:, :, None, None]
    np.testing.assert_array_equal(cpu.next_x[ok], g.next_x[ok])
    np.testing.assert_array_equal(cpu.next_z[ok], g.next_z[ok])
    np.testing.assert_allclose(cpu.cons[ok], g.cons[ok], rtol=0, atol=1e-6)


def test_exponential_branch_matches_cpu(gpu_solutions):
    """beta = 1 takes the non-naive path; it must agree too."""
    thetas, sols = gpu_solutions
    cpu = solve(*thetas[1], TINY)
    g = sols[1]
    ok = cpu.solvable & cpu.feasible[:, :, None, None]
    np.testing.assert_array_equal(cpu.next_x[ok], g.next_x[ok])


def test_simulated_panels_identical(gpu_solutions):
    """The binding check: identical trajectories, hence identical training data."""
    thetas, sols = gpu_solutions
    for th, g in zip(thetas, sols):
        pc = simulate(solve(*th, TINY), n_households=64, seed=0)
        pg = simulate(g, n_households=64, seed=0)
        for key in pc:
            np.testing.assert_array_equal(pc[key], pg[key], err_msg=f"{key} @ {th}")


def test_batching_does_not_change_results():
    """Splitting a batch must not alter any draw's solution."""
    from hh_npe.simulator.twoasset_gpu import solve_batch

    thetas = np.array([BENCHMARK, [0.6, 0.95, 2.5], [0.9, 0.97, 1.2]])
    one = solve_batch(thetas, TINY, theta_batch=3)
    split = solve_batch(thetas, TINY, theta_batch=1)
    for a, b in zip(one, split):
        np.testing.assert_array_equal(a.next_x, b.next_x)
        np.testing.assert_array_equal(a.next_z, b.next_z)
        np.testing.assert_array_equal(a.cons, b.cons)


def test_solution_is_float64():
    """float64 is a modeling requirement, not a preference (SIMULATOR_SPEC section 4)."""
    from hh_npe.simulator.twoasset_gpu import solve_batch

    sol = solve_batch(np.array([BENCHMARK]), TINY)[0]
    assert sol.cons.dtype == np.float64


def test_rejects_bad_parameters():
    from hh_npe.simulator.twoasset_gpu import solve_batch

    with pytest.raises(ValueError, match="delta"):
        solve_batch(np.array([[0.5, 1.0, 2.0]]), TINY)
    with pytest.raises(ValueError, match="positive"):
        solve_batch(np.array([[-0.1, 0.95, 2.0]]), TINY)
    with pytest.raises(ValueError, match=r"\(B, 3\)"):
        solve_batch(np.array([0.5, 0.95, 2.0]), TINY)
