from __future__ import annotations
from typing import Any
import numpy as np
from drift.base_detector import BaseDetector

class KSDetector(BaseDetector):
    def __init__(self, **kwargs):
        super().__init__(threshold=0.05, window_size=1000, cooldown_steps=500, **kwargs)

    @property
    def detector_type(self) -> str:
        return "KS"

    def compute_distance(self, ref_window: np.ndarray, cur_window: np.ndarray) -> float:
        return 0.0

    def get_per_dimension_distances(self) -> np.ndarray:
        return np.zeros(54, dtype=np.float32)