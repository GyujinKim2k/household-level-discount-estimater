"""The train/validation split must cut between panels, never through one.

Random-age augmentation gives one simulated household several rows. sbi splits
rows by random permutation, so by default windows of the same household land on
both sides of the split: the validation loss is then measured on panels the
model trained on, early stopping fires late, and every number that follows is
optimistic. ``group_ids`` exists to prevent exactly that, and these tests fail
if it silently stops working -- including if a future sbi changes the internals
it hooks.
"""

import pytest
import torch

from hh_npe.npe.embedder import TrajectoryTransformer
from hh_npe.npe.prior import PriorBox, sample_sobol
from hh_npe.npe.train import _use_grouped_split, train_npe

N_PANELS, PER_PANEL = 32, 4


@pytest.fixture(scope="module")
def grouped_dataset():
    """Each panel contributes PER_PANEL rows sharing one theta."""
    torch.manual_seed(0)
    box = PriorBox()
    theta_panel = torch.from_numpy(sample_sobol(N_PANELS, box, seed=0)).float()
    theta = theta_panel.repeat_interleave(PER_PANEL, dim=0)
    panel_id = torch.arange(N_PANELS).repeat_interleave(PER_PANEL)
    x = torch.randn(len(theta), 5, 3)
    x[:, :, 0] += theta[:, 0:1] * 5.0
    return theta, x, panel_id


def _train(theta, x, group_ids, **kw):
    return train_npe(
        theta, x,
        embedder=TrajectoryTransformer(d_model=16, n_heads=2, n_layers=1,
                                       output_dim=8),
        hidden_features=16, num_transforms=2, max_num_epochs=2,
        stop_after_epochs=2, batch_size=16, group_ids=group_ids, **kw,
    )


def test_no_panel_appears_on_both_sides(grouped_dataset):
    theta, x, panel_id = grouped_dataset
    _post, _de, inference = _train(theta, x, panel_id)

    train_panels = set(panel_id[inference.train_indices].tolist())
    val_panels = set(panel_id[inference.val_indices].tolist())
    assert train_panels & val_panels == set(), (
        f"{len(train_panels & val_panels)} panels leaked across the split"
    )
    assert train_panels | val_panels == set(range(N_PANELS))
    # Every row of a chosen panel goes with it -- no partial panels.
    assert len(inference.train_indices) + len(inference.val_indices) == len(theta)


def test_validation_fraction_is_respected_in_panels(grouped_dataset):
    theta, x, panel_id = grouped_dataset
    _post, _de, inference = _train(theta, x, panel_id, validation_fraction=0.25)
    val_panels = set(panel_id[inference.val_indices].tolist())
    assert len(val_panels) == int(0.25 * N_PANELS)


def test_ungrouped_split_does_leak(grouped_dataset):
    """The control: without group_ids the leak is real, not hypothetical.

    If this ever stops leaking, sbi changed its splitting and the wrapper may
    no longer be needed -- but that should be a deliberate discovery, not a
    silent one.
    """
    theta, x, panel_id = grouped_dataset
    _post, _de, inference = _train(theta, x, group_ids=None)
    train_panels = set(panel_id[inference.train_indices].tolist())
    val_panels = set(panel_id[inference.val_indices].tolist())
    assert train_panels & val_panels, "expected row-wise splitting to leak panels"


def test_mismatched_group_ids_raise():
    theta, x = torch.randn(8, 3), torch.randn(8, 5, 3)
    with pytest.raises(ValueError, match="group_ids has 4 entries"):
        _train(theta, x, torch.arange(4))


def test_grouped_split_is_deterministic(grouped_dataset):
    theta, x, panel_id = grouped_dataset
    splits = []
    for _ in range(2):
        _post, _de, inference = _train(theta, x, panel_id)
        splits.append(sorted(panel_id[inference.val_indices].tolist()))
    assert splits[0] == splits[1]


def test_wrapper_reports_the_split_it_made(grouped_dataset, caplog):
    """The log line is how a leak gets noticed in a long run."""
    import logging
    theta, x, panel_id = grouped_dataset
    with caplog.at_level(logging.INFO, logger="hh_npe.npe.train"):
        _train(theta, x, panel_id)
    assert any("Grouped split" in r.message for r in caplog.records)


def test_hook_targets_attributes_sbi_still_has(grouped_dataset):
    """Guard against a silent sbi upgrade breaking the wrapper.

    The wrapper stands in for ``get_dataloaders``, so its signature has to match
    what sbi's trainer passes positionally -- ``starting_round`` first, not the
    dataset size. Getting that wrong is silent: the wrapper would still run and
    still split, just on the wrong count.
    """
    from sbi.inference import SNPE_C

    from hh_npe.npe.prior import make_sbi_prior

    theta, x, panel_id = grouped_dataset
    inference = SNPE_C(prior=make_sbi_prior(PriorBox()), device="cpu")
    inference.append_simulations(theta, x)
    _use_grouped_split(inference, panel_id)
    inference.get_dataloaders(0, 16, 0.1)
    assert hasattr(inference, "train_indices") and hasattr(inference, "val_indices")
    train_panels = set(panel_id[inference.train_indices].tolist())
    val_panels = set(panel_id[inference.val_indices].tolist())
    assert train_panels & val_panels == set()
