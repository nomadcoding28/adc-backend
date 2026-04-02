"""
explainability/explanation_builder.py
=======================================
Assembles a structured ``ExplanationCard`` from ReAct step outputs
and a final LLM completion that fills in all four sections.

The ExplanationCard is the primary data structure for the dashboard's
explainability panel — it maps directly to the UI card shown for each
defender action.

Usage
-----
    builder = ExplanationBuilder(llm=llm_client, retriever=rag_retriever)

    card = builder.build(
        context     = action_context_dict,
        observe     = observe_text,
        think       = think_text,
        act         = act_text,
        result      = result_text,
        retrieved   = list_of_retrieval_results,
        context_str = formatted_docs_string,
        obs_summary = obs_summary_string,
        trace       = react_trace_object,
    )

    # Access structured fields
    print(card.threat_detected)
    print(card.why_action)
    print(card.risk_mitigated)
    print(card.recommended_followup)

    # Serialise for API / dashboard
    api_response = card.to_dict()
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from explainability.llm.client import LLMClient
from explainability.llm.prompts import PromptLibrary
from explainability.llm.token_counter import TokenCounter
from explainability.rag.retriever import RAGRetriever, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class ExplanationCard:
    """
    Structured explanation for a single defender action.

    This is the canonical output of the explainability pipeline.
    All fields are plain strings, safe for JSON serialisation.

    Attributes
    ----------
    card_id : str
        Unique identifier (e.g. ``"exp_4821_Isolate_Host3"``).
    step : int
        Episode step at which the action occurred.
    action : str
        Human-readable action string.
    action_idx : int
        Integer action index.
    severity : str
        Severity level: ``"CRITICAL"``, ``"HIGH"``, ``"MEDIUM"``, ``"LOW"``.
    threat_detected : str
        LLM-generated threat description (OBSERVE section).
    why_action : str
        LLM-generated action justification (ACT section).
    risk_mitigated : str
        LLM-generated risk mitigation description (RESULT section).
    recommended_followup : str
        LLM-generated follow-up recommendations.
    cve_ids : list[str]
        CVE IDs involved in this incident.
    technique_ids : list[str]
        ATT&CK technique IDs detected.
    risk_score : float
        CVaR risk score (0–1).
    attacker_type : str
        Attacker type belief string.
    obs_summary : str
        Compact observation state summary.
    retrieved_docs : list[dict]
        Top retrieved RAG documents (for the context panel in the UI).
    react_trace : dict
        Full ReAct reasoning trace for the detail view.
    latency_s : float
        Wall-clock time to generate this explanation.
    tokens_used : int
        Total LLM tokens consumed.
    generated_at : str
        ISO 8601 timestamp.
    action_success : bool
        Whether the action succeeded in the environment.
    reward_breakdown : dict
        Shaped reward component breakdown.
    """
    card_id:              str
    step:                 int
    action:               str
    action_idx:           int                  = 0
    severity:             str                  = "MEDIUM"
    threat_detected:      str                  = ""
    why_action:           str                  = ""
    risk_mitigated:       str                  = ""
    recommended_followup: str                  = ""
    cve_ids:              List[str]            = field(default_factory=list)
    technique_ids:        List[str]            = field(default_factory=list)
    risk_score:           float                = 0.0
    attacker_type:        str                  = ""
    obs_summary:          str                  = ""
    retrieved_docs:       List[Dict[str, Any]] = field(default_factory=list)
    react_trace:          Dict[str, Any]       = field(default_factory=dict)
    latency_s:            float                = 0.0
    tokens_used:          int                  = 0
    generated_at:         str                  = ""
    action_success:       bool                 = True
    reward_breakdown:     Dict[str, float]     = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict for the API response."""
        d = asdict(self)
        return d

    @property
    def is_critical(self) -> bool:
        """True if this explanation involves a critical-severity action."""
        return self.severity in ("CRITICAL", "HIGH") or self.risk_score > 0.7

    def __str__(self) -> str:
        return (
            f"ExplanationCard(step={self.step}, action={self.action!r}, "
            f"severity={self.severity}, risk={self.risk_score:.2f})"
        )


class ExplanationBuilder:
    """
    Assembles ``ExplanationCard`` objects from ReAct outputs.

    Two assembly modes:
      1. **Section-aware** (default): Makes one final LLM call that takes
         all four ReAct outputs as context and produces a clean, well-formatted
         explanation filling in all four card sections.
      2. **Direct assembly**: If ``use_final_pass=False``, uses the ReAct
         outputs directly without a final LLM pass (faster, less polished).

    Parameters
    ----------
    llm : LLMClient
        LLM client for the optional final assembly pass.
    retriever : RAGRetriever
        Used to retrieve docs for the final assembly prompt (if needed).
    use_final_pass : bool
        If True (default), run one final LLM call to produce clean
        four-section card content.
    """

    def __init__(
        self,
        llm:            LLMClient,
        retriever:      RAGRetriever,
        use_final_pass: bool = True,
    ) -> None:
        self.llm            = llm
        self.retriever      = retriever
        self.use_final_pass = use_final_pass
        self._counter       = TokenCounter(model=llm.model)

    # ------------------------------------------------------------------ #
    # Primary interface
    # ------------------------------------------------------------------ #

    def build(
        self,
        context:     Dict[str, Any],
        observe:     str,
        think:       str,
        act:         str,
        result:      str,
        retrieved:   List[RetrievalResult],
        context_str: str,
        obs_summary: str,
        trace:       Any,   # ReActTrace
    ) -> ExplanationCard:
        """
        Build an ExplanationCard from ReAct outputs.

        Parameters
        ----------
        context : dict
            Original action context dict.
        observe, think, act, result : str
            Text outputs from each ReAct step.
        retrieved : list[RetrievalResult]
            Documents retrieved during the THINK step.
        context_str : str
            Pre-formatted context documents string.
        obs_summary : str
            Observation state summary.
        trace : ReActTrace
            Full reasoning trace.

        Returns
        -------
        ExplanationCard
        """
        action = context.get("action", "Unknown")
        step   = context.get("step", 0)

        # Determine severity from risk score and attacker type
        risk_score = context.get("risk_score", 0.0)
        severity   = self._infer_severity(risk_score, context.get("cve_ids", []))

        # Generate or assemble section content
        if self.use_final_pass:
            sections = self._final_assembly_pass(
                context      = context,
                observe      = observe,
                think        = think,
                act          = act,
                result       = result,
                context_str  = context_str,
                obs_summary  = obs_summary,
            )
        else:
            sections = {
                "threat_detected":      think,
                "why_action":           act,
                "risk_mitigated":       result,
                "recommended_followup": self._extract_followup(result),
            }

        # Build the card
        card = ExplanationCard(
            card_id              = self._make_card_id(step, action),
            step                 = step,
            action               = action,
            action_idx           = context.get("action_idx", 0),
            severity             = severity,
            threat_detected      = sections.get("threat_detected", think),
            why_action           = sections.get("why_action", act),
            risk_mitigated       = sections.get("risk_mitigated", result),
            recommended_followup = sections.get("recommended_followup", ""),
            cve_ids              = context.get("cve_ids", []),
            technique_ids        = context.get("technique_ids", []),
            risk_score           = risk_score,
            attacker_type        = context.get("attacker_type", ""),
            obs_summary          = obs_summary,
            retrieved_docs       = [r.to_dict() for r in retrieved],
            react_trace          = trace.to_dict() if hasattr(trace, "to_dict") else {},
            tokens_used          = getattr(trace, "total_tokens", 0),
            generated_at         = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            action_success       = context.get("action_success", True),
            reward_breakdown     = context.get("reward_breakdown", {}),
        )
        return card

    # ------------------------------------------------------------------ #
    # Final assembly pass
    # ------------------------------------------------------------------ #

    def _final_assembly_pass(
        self,
        context:     Dict[str, Any],
        observe:     str,
        think:       str,
        act:         str,
        result:      str,
        context_str: str,
        obs_summary: str,
    ) -> Dict[str, str]:
        """
        One final LLM call that takes all ReAct outputs and produces
        clean, structured four-section card content.

        Returns a dict with keys:
            threat_detected, why_action, risk_mitigated, recommended_followup
        """
        prompt = PromptLibrary.action_explanation(
            action        = context.get("action", ""),
            threat        = context.get("threat", ""),
            context_docs  = context_str,
            obs_summary   = obs_summary,
            risk_score    = context.get("risk_score", 0.0),
            attacker_type = context.get("attacker_type", "Unknown"),
            step          = context.get("step", 0),
            technique_ids = context.get("technique_ids"),
        )

        # Append ReAct reasoning as additional context
        react_context = (
            f"\n\n## REASONING CHAIN\n"
            f"OBSERVE: {observe}\n\n"
            f"THINK: {think}\n\n"
            f"ACT: {act}\n\n"
            f"RESULT: {result}"
        )

        messages = self._counter.build_prompt_within_budget(
            system       = PromptLibrary.system_explanation(),
            query        = prompt + react_context,
            context_docs = "",    # Already included in prompt
            max_context_tokens   = 3000,
            reserved_for_completion = 800,
        )

        resp = self.llm.chat(messages, max_tokens=800, temperature=0.2)
        return self._parse_sections(resp.content)

    # ------------------------------------------------------------------ #
    # Parsing helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_sections(text: str) -> Dict[str, str]:
        """
        Parse the four-section LLM output into a dict.

        Handles both ``### SECTION NAME`` and ``**SECTION NAME**`` formats.
        Falls back to splitting the text if headers are not found.
        """
        sections: Dict[str, str] = {
            "threat_detected":      "",
            "why_action":           "",
            "risk_mitigated":       "",
            "recommended_followup": "",
        }

        # Pattern: ### THREAT DETECTED or **THREAT DETECTED**
        header_pattern = re.compile(
            r"(?:#+\s*|[*]{2})?"
            r"(THREAT DETECTED|WHY THIS ACTION|RISK MITIGATED|RECOMMENDED FOLLOW.?UP)"
            r"(?:[*]{2})?\s*\n",
            re.IGNORECASE,
        )

        parts = header_pattern.split(text)

        # parts layout after split: [before, header1, content1, header2, content2, ...]
        i = 1
        while i < len(parts) - 1:
            header  = parts[i].strip().upper()
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""

            if "THREAT" in header:
                sections["threat_detected"] = content
            elif "WHY" in header:
                sections["why_action"] = content
            elif "RISK" in header:
                sections["risk_mitigated"] = content
            elif "FOLLOW" in header or "RECOMMEND" in header:
                sections["recommended_followup"] = content

            i += 2

        # Fallback: if no headers found, use full text for threat_detected
        if not any(sections.values()):
            sections["threat_detected"] = text.strip()

        return sections

    @staticmethod
    def _extract_followup(result_text: str) -> str:
        """
        Extract the follow-up recommendation from the RESULT step output.

        Looks for sentences containing "monitor", "watch", "next", or "follow".
        Returns the full result text if no match found.
        """
        sentences = re.split(r"(?<=[.!?])\s+", result_text)
        followup_keywords = {"monitor", "watch", "next", "follow", "check", "alert"}

        followup = [
            s for s in sentences
            if any(kw in s.lower() for kw in followup_keywords)
        ]
        return " ".join(followup) if followup else result_text

    @staticmethod
    def _infer_severity(
        risk_score: float,
        cve_ids:    List[str],
    ) -> str:
        """
        Infer the severity label from the CVaR risk score.

        Parameters
        ----------
        risk_score : float
            CVaR risk score (0–1).
        cve_ids : list[str]
            CVE IDs involved (used for heuristic CVSS estimation).

        Returns
        -------
        str
            One of ``"CRITICAL"``, ``"HIGH"``, ``"MEDIUM"``, ``"LOW"``.
        """
        if risk_score >= 0.8:
            return "CRITICAL"
        elif risk_score >= 0.5:
            return "HIGH"
        elif risk_score >= 0.3:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _make_card_id(step: int, action: str) -> str:
        """Generate a unique card ID from step and action."""
        safe_action = action.replace(" ", "_").replace("(", "").replace(")", "")
        return f"exp_{step}_{safe_action}"

    def __repr__(self) -> str:
        return (
            f"ExplanationBuilder("
            f"model={self.llm.model!r}, "
            f"final_pass={self.use_final_pass})"
        )