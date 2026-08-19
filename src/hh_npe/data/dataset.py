"""Minimal save/load helpers for ``(theta, x)`` training pairs.

sbi consumes raw tensors via ``inferer.append_simulations(theta, x).train()``,
so we don't need a torch Dataset wrapper for Phase 2 — just persistence
helpers. Keep the on-disk format trivial and torch-native (``.pt``).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def shard_dir(dataset_path: str | Path) -> Path:
    """Where a dataset's checkpoint shards live, given its ``.pt`` path."""
    p = Path(dataset_path)
    return p.parent / (p.stem + "_shards")


def read_solver_config(dataset_path: str | Path) -> dict | None:
    """The solver settings a dataset's shards were generated under, if recorded.

    The GPU solve is reproducible at a fixed ``(device, theta_batch, chunk)``
    and not across them, so SBC has to reproduce the training set's settings
    exactly rather than pick its own. Returns ``None`` for datasets built before
    the configuration was recorded, or on the CPU path where it does not apply.
    """
    marker = shard_dir(dataset_path) / "solver_config.json"
    return json.loads(marker.read_text()) if marker.exists() else None


def save_dataset(theta: torch.Tensor, x: torch.Tensor, path: str | Path) -> None:
    """Save ``(theta, x)`` to ``path`` as a single ``.pt`` file.

    Both tensors are cast to ``float32``. Parent directories are created.
    """
    if theta.shape[0] != x.shape[0]:
        raise ValueError(
            f"theta and x must have the same n_samples; got {theta.shape[0]} and {x.shape[0]}"
        )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"theta": theta.float(), "x": x.float()}, p)


def load_dataset(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load ``(theta, x)`` from a ``.pt`` file saved by :func:`save_dataset`."""
    d = torch.load(Path(path), weights_only=True)
    return d["theta"], d["x"]
