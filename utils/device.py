"""
utils/device.py
================
CUDA / MPS / CPU device resolver.

Usage
-----
    from utils.device import resolve_device
    device = resolve_device("auto")   # returns torch.device("cuda") if available
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def resolve_device(device_str: str = "auto") -> "torch.device":
    """
    Resolve a device string to a torch.device.

    Parameters
    ----------
    device_str : str
        ``"auto"``, ``"cuda"``, ``"cuda:0"``, ``"mps"``, or ``"cpu"``.
        ``"auto"`` selects CUDA > MPS > CPU.

    Returns
    -------
    torch.device
    """
    import torch

    if device_str != "auto":
        device = torch.device(device_str)
        logger.debug("Device: %s (explicit)", device)
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info("Device auto-selected: %s", device)
    return device


def device_info() -> dict:
    """Return a dict of available compute device information."""
    try:
        import torch
        return {
            "cuda_available":   torch.cuda.is_available(),
            "cuda_device_count":torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "mps_available":    (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
            "selected":         str(resolve_device("auto")),
        }
    except ImportError:
        return {"error": "PyTorch not installed"}