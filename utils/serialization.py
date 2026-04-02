"""
utils/serialization.py
=======================
Safe JSON and pickle serialisation helpers.

Usage
-----
    from utils.serialization import safe_json_dumps, safe_pickle_save

    json_str = safe_json_dumps({"reward": 8.74, "device": torch.device("cuda")})
    safe_pickle_save(obj, "data/experiences/task_0.pkl")
    obj = safe_pickle_load("data/experiences/task_0.pkl")
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class _SafeEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays, torch tensors, and Path objects."""

    def default(self, obj: Any) -> Any:
        # numpy
        try:
            import numpy as np
            if isinstance(obj, np.integer):  return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray):  return obj.tolist()
        except ImportError:
            pass

        # torch
        try:
            import torch
            if isinstance(obj, torch.Tensor): return obj.detach().cpu().tolist()
            if isinstance(obj, torch.device): return str(obj)
        except ImportError:
            pass

        # Path
        if isinstance(obj, Path):
            return str(obj)

        # Fallback
        try:
            return str(obj)
        except Exception:
            return None


def safe_json_dumps(obj: Any, indent: int = None) -> str:
    """
    Serialise ``obj`` to JSON string, handling numpy/torch types.

    Parameters
    ----------
    obj : Any
        Object to serialise.
    indent : int, optional
        JSON indentation.

    Returns
    -------
    str
    """
    return json.dumps(obj, cls=_SafeEncoder, indent=indent)


def safe_json_loads(text: str) -> Any:
    """Deserialise JSON string, returning None on error."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("JSON parse error: %s", exc)
        return None


def safe_pickle_save(obj: Any, path: str, mkdir: bool = True) -> Path:
    """
    Save an object to a pickle file.

    Parameters
    ----------
    obj : Any
        Object to serialise.
    path : str
        Destination file path.
    mkdir : bool
        Create parent directories if they don't exist.

    Returns
    -------
    Path
    """
    p = Path(path)
    if mkdir:
        p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.debug("Pickled object saved to %s (%d bytes)", p, p.stat().st_size)
    return p


def safe_pickle_load(path: str) -> Optional[Any]:
    """
    Load an object from a pickle file.

    Parameters
    ----------
    path : str
        Source file path.

    Returns
    -------
    Any or None
        Loaded object, or None if file not found / corrupt.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("Pickle file not found: %s", p)
        return None

    try:
        with p.open("rb") as f:
            obj = pickle.load(f)
        logger.debug("Loaded pickle from %s", p)
        return obj
    except Exception as exc:
        logger.error("Failed to load pickle %s: %s", p, exc)
        return None