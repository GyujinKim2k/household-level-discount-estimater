"""Environment validation — confirms all deps import and HARK can solve a trivial model.

Run with: `uv run pytest tests/test_env.py -v`
"""

import importlib


def test_import_numpy_scipy_polars_numba():
    for mod in ("numpy", "scipy", "polars", "numba"):
        importlib.import_module(mod)


def test_import_torch_reports_cuda():
    import torch

    assert torch.__version__
    cuda = torch.cuda.is_available()
    print(f"\ntorch={torch.__version__} cuda_available={cuda}")
    if cuda:
        print(f"  device_count={torch.cuda.device_count()} "
              f"device_name={torch.cuda.get_device_name(0)}")


def test_import_sbi():
    import sbi

    assert sbi.__version__


def test_hark_solve_perf_foresight():
    """Cheapest non-trivial HARK call — verifies econ-ark install integrity."""
    from HARK.ConsumptionSaving.ConsIndShockModel import PerfForesightConsumerType

    agent = PerfForesightConsumerType()
    agent.solve()
    assert agent.solution is not None
    assert len(agent.solution) > 0


def test_import_hydra():
    import hydra
    from omegaconf import OmegaConf

    assert hydra.__version__
    cfg = OmegaConf.create({"x": 1})
    assert cfg.x == 1


def test_wandb_disabled_init():
    """Verify wandb can init in disabled mode (no network, no files written)."""
    import wandb

    run = wandb.init(mode="disabled")
    run.log({"smoke": 1.0})
    run.finish()


def test_hh_npe_package_importable():
    import hh_npe

    assert hh_npe.__version__ == "0.1.0"
