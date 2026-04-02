"""
drift/
======
Concept drift detection for the ACD Framework.

Monitors the observation distribution during training and triggers the
EWC continual learning pipeline when the attacker's behaviour distribution
shifts significantly — indicating the agent has entered a new attack regime.

Mathematical background
-----------------------
Let P_ref be the reference window distribution (collected observations
over the past W steps before the last detected drift).

Let P_cur be the current window distribution (most recent W steps).

Drift is declared when D(P_ref, P_cur) > threshold, where D is one of:

    Wasserstein-1 : W1(P, Q) = inf_γ E_{(x,y)~γ}[||x - y||]
                  : Approximated via sorted empirical CDFs

    KS test       : KS(P, Q) = sup_x |F_P(x) - F_Q(x)|
                  : Exact for 1D marginals

    MMD           : MMD²(P, Q) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
                  : Maximum Mean Discrepancy with RBF kernel

All three compute a scalar distance that is compared to a configurable
threshold.  Drift events are debounced by a cooldown period.

Public API
----------
    from drift import DriftDetector, WassersteinDetector
    from drift import KSDetector, MMDDetector
    from drift import WindowManager, DriftEvent
    from drift import DetectorFactory
"""

from drift.base_detector import BaseDetector, DriftEvent, DriftResult
from drift.wasserstein_detector import WassersteinDetector
from drift.ks_detector import KSDetector
from drift.mmd_detector import MMDDetector
from drift.window_manager import WindowManager, ObservationWindow
from drift.detector_factory import DetectorFactory, DriftDetector

__all__ = [
    "BaseDetector",
    "DriftEvent",
    "DriftResult",
    "WassersteinDetector",
    "KSDetector",
    "MMDDetector",
    "WindowManager",
    "ObservationWindow",
    "DetectorFactory",
    "DriftDetector",
]