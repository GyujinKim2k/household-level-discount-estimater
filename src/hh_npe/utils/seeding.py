"""Centralized seeding for numpy, torch, and python random."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed python random, numpy, torch (CPU + CUDA if present), and PYTHONHASHSEED.

    For bitwise determinism on CUDA, callers should additionally set
    ``torch.backends.cudnn.deterministic = True`` and
    ``torch.backends.cudnn.benchmark = False`` — those are not set here because
    they impose a global throughput penalty that not every caller wants.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
