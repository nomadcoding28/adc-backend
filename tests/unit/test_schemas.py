"""
tests/unit/test_schemas.py
===========================
Unit tests for all Pydantic v2 API schemas.

Validates that all schema models:
    - Instantiate with defaults
    - Accept valid data
    - Reject invalid data (where constraints are defined)
    - Serialise via model_dump()
"""

from __future__ import annotations

import pytest


# ── Training schemas ────────────────────────────────────────────────────────

class TestTrainingSchemas:
    def test_training_start_request_defaults(self) -> None:
        from api.schemas.training import TrainingStartRequest
        r = TrainingStartRequest()
        assert r.agent_type is None
        assert r.total_timesteps is None

    def test_training_start_request_with_values(self) -> None:
        from api.schemas.training import TrainingStartRequest
        r = TrainingStartRequest(
            agent_type="cvar_ppo",
            total_timesteps=500_000,
            cvar_alpha=0.05,
        )
        d = r.model_dump(exclude_none=True)
        assert d["agent_type"] == "cvar_ppo"
        assert d["total_timesteps"] == 500_000

    def test_training_status_defaults(self) -> None:
        from api.schemas.training import TrainingStatus
        s = TrainingStatus()
        assert s.is_training is False

    def test_training_status_with_metrics(self) -> None:
        from api.schemas.training import TrainingStatus
        s = TrainingStatus(
            is_training=True, mean_reward=42.0, cvar_005=-15.0, agent_type="cvar_ppo",
        )
        assert s.mean_reward == 42.0

    def test_training_config_has_defaults(self) -> None:
        from api.schemas.training import TrainingConfig
        c = TrainingConfig()
        assert c.agent_type == "cvar_ppo"
        assert c.cvar_alpha == 0.05

    def test_training_result_serialisable(self) -> None:
        from api.schemas.training import TrainingResult
        r = TrainingResult(status="completed", mean_reward=42.0)
        d = r.model_dump()
        assert isinstance(d, dict)
        assert d["status"] == "completed"


# ── Evaluation schemas ──────────────────────────────────────────────────────

class TestEvaluationSchemas:
    def test_eval_result_defaults(self) -> None:
        from api.schemas.evaluation import EvalResult
        r = EvalResult()
        assert r.n_episodes == 0

    def test_eval_result_with_metrics(self) -> None:
        from api.schemas.evaluation import EvalResult
        r = EvalResult(
            n_episodes=50, mean_reward=40.0, cvar_005=-15.0,
            success_rate=0.85, catastrophic_rate=0.05,
        )
        assert r.success_rate == 0.85

    def test_benchmark_table(self) -> None:
        from api.schemas.evaluation import BenchmarkTable, BenchmarkRow
        table = BenchmarkTable(rows=[
            BenchmarkRow(agent="CVaR-PPO + EWC", mean_reward=42.0, is_ours=True),
            BenchmarkRow(agent="Standard PPO", mean_reward=38.0),
        ])
        assert len(table.rows) == 2
        assert table.rows[0].is_ours is True


# ── CVaR schemas ────────────────────────────────────────────────────────────

class TestCVaRSchemas:
    def test_cvar_metrics_defaults(self) -> None:
        from api.schemas.cvar import CVaRMetrics
        m = CVaRMetrics()
        assert m.alpha == 0.05

    def test_alpha_sensitivity(self) -> None:
        from api.schemas.cvar import AlphaSensitivity
        a = AlphaSensitivity(alpha=0.05, cvar_value=-15.0, interpretation="Worst 5%")
        assert a.alpha == 0.05


# ── Drift schemas ───────────────────────────────────────────────────────────

class TestDriftSchemas:
    def test_drift_score_defaults(self) -> None:
        from api.schemas.drift import DriftScore
        s = DriftScore()
        assert s.is_drifting is False
        assert s.threshold == 0.15

    def test_drift_event(self) -> None:
        from api.schemas.drift import DriftEvent
        e = DriftEvent(step=1000, distance=0.25, threshold=0.15)
        assert e.distance > e.threshold


# ── Explain schemas ─────────────────────────────────────────────────────────

class TestExplainSchemas:
    def test_explanation_request(self) -> None:
        from api.schemas.explain import ExplanationRequest
        r = ExplanationRequest(
            action="Isolate Host-3", step=100, threat="CVE-2021-44228",
            risk_score=0.87, technique_ids=["T1190"],
        )
        assert r.action == "Isolate Host-3"
        assert len(r.technique_ids) == 1

    def test_explanation_card(self) -> None:
        from api.schemas.explain import ExplanationCard
        c = ExplanationCard(
            action="Isolate Host-3",
            threat_detected="Malicious process on Host-3",
            why_action="Host-3 is compromised with active C2",
        )
        d = c.model_dump()
        assert "threat_detected" in d


# ── Game schemas ────────────────────────────────────────────────────────────

class TestGameSchemas:
    def test_belief_state_defaults(self) -> None:
        from api.schemas.game import BeliefState
        b = BeliefState()
        assert sum(b.probabilities.values()) == pytest.approx(1.0, abs=0.01)

    def test_attacker_prediction(self) -> None:
        from api.schemas.game import AttackerPrediction
        p = AttackerPrediction(dominant_type="APT", dominant_probability=0.71)
        assert p.dominant_type == "APT"

    def test_game_state(self) -> None:
        from api.schemas.game import GameState
        s = GameState(n_compromised=3, n_clean=4, step=500)
        assert s.n_compromised == 3


# ── KG schemas ──────────────────────────────────────────────────────────────

class TestKGSchemas:
    def test_kg_graph_empty(self) -> None:
        from api.schemas.kg import KGGraph
        g = KGGraph()
        assert g.nodes == []
        assert g.edges == []

    def test_cve_node(self) -> None:
        from api.schemas.kg import CVENode
        c = CVENode(cve_id="CVE-2021-44228", cvss_score=10.0, severity="CRITICAL")
        assert c.cvss_score == 10.0

    def test_attack_chain(self) -> None:
        from api.schemas.kg import AttackChain, AttackStep
        chain = AttackChain(
            cve_id="CVE-2021-44228",
            steps=[AttackStep(order=1, technique_id="T1190", tactic="Initial Access")],
        )
        assert len(chain.steps) == 1


# ── Alert schemas ───────────────────────────────────────────────────────────

class TestAlertSchemas:
    def test_alert_defaults(self) -> None:
        from api.schemas.alerts import Alert
        a = Alert()
        assert a.severity == "LOW"
        assert a.acknowledged is False

    def test_alert_severity_enum(self) -> None:
        from api.schemas.alerts import AlertSeverity
        assert AlertSeverity.CRITICAL.value == "CRITICAL"

    def test_alert_update(self) -> None:
        from api.schemas.alerts import AlertUpdate
        u = AlertUpdate(acknowledged=True)
        d = u.model_dump(exclude_none=True)
        assert d["acknowledged"] is True


# ── Incident schemas ────────────────────────────────────────────────────────

class TestIncidentSchemas:
    def test_incident_create(self) -> None:
        from api.schemas.incidents import IncidentCreate
        c = IncidentCreate(
            severity="HIGH",
            threat_description="Active C2 beacon on Host-5",
            affected_hosts=["Host-5"],
        )
        assert c.severity == "HIGH"

    def test_incident(self) -> None:
        from api.schemas.incidents import Incident
        i = Incident(title="Test Incident", severity="MEDIUM")
        d = i.model_dump()
        assert d["severity"] == "MEDIUM"


# ── Network schemas ─────────────────────────────────────────────────────────

class TestNetworkSchemas:
    def test_host_state(self) -> None:
        from api.schemas.network import HostState
        h = HostState(name="Enterprise0", compromised=True, malicious_process=True)
        assert h.compromised is True

    def test_topology_graph(self) -> None:
        from api.schemas.network import TopologyGraph, HostState
        topo = TopologyGraph(
            hosts=[HostState(name="Enterprise0"), HostState(name="Op_Server0")],
            compromised_ratio=0.15,
        )
        assert len(topo.hosts) == 2
        assert topo.compromised_ratio == 0.15
