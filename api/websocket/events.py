"""
api/websocket/events.py
========================
Typed WebSocket event payload definitions.

Every event broadcast through the WebSocket manager has a known
``event`` field that clients use to route the payload to the
correct handler.

Event catalogue
---------------
    connected           Connection confirmation
    training_update     Live training metrics (reward, loss, CVaR)
    training_complete   Training run finished
    alert               New security alert
    drift_detected      Concept drift event
    network_update      Network host state changed
    belief_update       Bayesian attacker belief changed
    action_taken        Agent took a defensive action
    override            Human override executed
    ewc_update          EWC task registered / forgetting metric
    episode_end         Episode completed with summary
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Optional


class WSEventType(str, Enum):
    CONNECTED         = "connected"
    TRAINING_UPDATE   = "training_update"
    TRAINING_COMPLETE = "training_complete"
    ALERT             = "alert"
    DRIFT_DETECTED    = "drift_detected"
    NETWORK_UPDATE    = "network_update"
    BELIEF_UPDATE     = "belief_update"
    ACTION_TAKEN      = "action_taken"
    OVERRIDE          = "override"
    EWC_UPDATE        = "ewc_update"
    EPISODE_END       = "episode_end"


@dataclass
class WSEvent:
    """Base WebSocket event."""
    event: str
    data:  Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"event": self.event, **self.data}


def make_training_update(
    step:        int,
    mean_reward: Optional[float],
    cvar_005:    Optional[float],
    loss_policy: Optional[float] = None,
    loss_ewc:    Optional[float] = None,
    episode:     int = 0,
) -> Dict[str, Any]:
    return {
        "event":       WSEventType.TRAINING_UPDATE,
        "step":        step,
        "episode":     episode,
        "mean_reward": mean_reward,
        "cvar_005":    cvar_005,
        "loss_policy": loss_policy,
        "loss_ewc":    loss_ewc,
    }


def make_alert_event(
    title:    str,
    severity: str,
    desc:     str = "",
    **kwargs,
) -> Dict[str, Any]:
    return {
        "event":    WSEventType.ALERT,
        "title":    title,
        "severity": severity,
        "desc":     desc,
        **kwargs,
    }


def make_drift_event(
    step:     int,
    distance: float,
    threshold:float,
    event_id: int = 0,
) -> Dict[str, Any]:
    return {
        "event":     WSEventType.DRIFT_DETECTED,
        "step":      step,
        "distance":  distance,
        "threshold": threshold,
        "event_id":  event_id,
    }


def make_belief_update(
    dominant_type: str,
    probabilities: Dict[str, float],
    entropy:       float,
    step:          int = 0,
) -> Dict[str, Any]:
    return {
        "event":         WSEventType.BELIEF_UPDATE,
        "dominant_type": dominant_type,
        "probabilities": probabilities,
        "entropy":       entropy,
        "step":          step,
    }


def make_action_event(
    action:      str,
    action_idx:  int,
    step:        int,
    reward:      float,
    risk_score:  float = 0.0,
    success:     bool  = True,
) -> Dict[str, Any]:
    return {
        "event":      WSEventType.ACTION_TAKEN,
        "action":     action,
        "action_idx": action_idx,
        "step":       step,
        "reward":     reward,
        "risk_score": risk_score,
        "success":    success,
    }


def make_ewc_update(
    tasks_registered: int,
    forgetting_metric:float,
    lambda_ewc:       float = 0.4,
) -> Dict[str, Any]:
    return {
        "event":              WSEventType.EWC_UPDATE,
        "tasks_registered":   tasks_registered,
        "forgetting_metric":  forgetting_metric,
        "lambda_ewc":         lambda_ewc,
    }


def make_episode_end(
    episode:     int,
    reward:      float,
    steps:       int,
    terminated:  bool,
    n_compromised:int = 0,
) -> Dict[str, Any]:
    return {
        "event":        WSEventType.EPISODE_END,
        "episode":      episode,
        "reward":       reward,
        "steps":        steps,
        "terminated":   terminated,
        "n_compromised":n_compromised,
    }