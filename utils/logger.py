"""
utils/logger.py
================
Centralised logger factory.

Usage
-----
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Training started — step=%d", 0)
"""

from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Return a named logger.

    Parameters
    ----------
    name : str
        Logger name — typically ``__name__`` of the calling module.
    level : str, optional
        Override log level for this specific logger.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if level is not None:
        numeric = getattr(logging, level.upper(), None)
        if numeric is not None:
            logger.setLevel(numeric)

    return logger