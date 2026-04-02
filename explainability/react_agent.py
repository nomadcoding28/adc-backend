"""
explainability/react_agent.py
==============================
ReAct (Reason + Act) agent for structured explanation generation.

The ReAct framework (Yao et al., 2022) interleaves reasoning traces
with action calls.  For ACD explanation, the loop is:

    OBSERVE  →  What is happening in the network right now?
    THINK    →  What does the retrieved threat intel say about this?
    ACT      →  Why did the agent choose this specific action?
    RESULT   →  What outcome is expected?

Each step produces a short LLM-generated text that is passed as context
to the next step, creating a coherent reasoning chain that grounds the
final explanation in real CVE/ATT&CK knowledge.

Usage
-----
    react = ReActAgent(llm=llm_client, retriever=rag_retriever, builder=builder)

    context = {
        "action":         "Isolate Host-3",
        "action_idx":     8,
        "step":           4821,
        "threat":         "CVE-2021-44228 detected on Host-3",
        "obs_decoded":    obs_processor.decode(obs_vec),
        "risk_score":     0.87,
        "attacker_type":  "Targeted APT (71%)",
        "technique_ids":  ["T1190", "T1059"],
        "cve_ids":        ["CVE-2021-44228"],
        "action_success": True,
    }

    card = react.explain(context)
    print(card.threat_detected)
    print(card.why_action)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from explainability.llm.client import LLMClient
from explainability.llm.prompts import PromptLibrary
from explainability.llm.token_counter import TokenCounter
from explainability.rag.retriever import RAGRetriever, RetrievalResult
from explainability.explanation_builder import ExplanationBuilder, ExplanationCard

logger = logging.getLogger(__name__)

# ReAct step labels (used in trace logging)
_STEP_OBSERVE = "OBSERVE"
_STEP_THINK   = "THINK"
_STEP_ACT     = "ACT"
_STEP_RESULT  = "RESULT"


class ReActTrace:
    """
    Captures the full ReAct reasoning trace for a single explanation.

    Attributes
    ----------
    steps : list[dict]
        Ordered list of ReAct steps, each with:
        ``{step_type, prompt_tokens, output, latency_s}``.
    retrieved_docs : list[RetrievalResult]
        Documents retrieved during the OBSERVE step.
    total_tokens : int
        Cumulative token usage across all steps.
    total_latency_s : float
        Total wall-clock time for all LLM calls.
    """

    def __init__(self) -> None:
        self.steps:          List[Dict[str, Any]] = []
        self.retrieved_docs: List[RetrievalResult] = []
        self.total_tokens:   int   = 0
        self.total_latency_s: float = 0.0

    def add_step(
        self,
        step_type:     str,
        output:        str,
        prompt_tokens: int   = 0,
        latency_s:     float = 0.0,
    ) -> None:
        """Record a single ReAct step."""
        self.steps.append({
            "step_type":     step_type,
            "output":        output,
            "prompt_tokens": prompt_tokens,
            "latency_s":     round(latency_s, 3),
        })
        self.total_tokens    += prompt_tokens
        self.total_latency_s += latency_s

    def get_step_output(self, step_type: str) -> str:
        """Return the output text for a given step type."""
        return next(
            (s["output"] for s in self.steps if s["step_type"] == step_type),
            "",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps":           self.steps,
            "n_docs_retrieved": len(self.retrieved_docs),
            "total_tokens":    self.total_tokens,
            "total_latency_s": round(self.total_latency_s, 3),
        }


class ReActAgent:
    """
    Orchestrates the ReAct reasoning loop for defender action explanations.

    Parameters
    ----------
    llm : LLMClient
        LLM client for text generation.
    retriever : RAGRetriever
        RAG retriever for document lookup.
    builder : ExplanationBuilder
        Assembles the final ``ExplanationCard`` from ReAct outputs.
    max_react_steps : int
        Maximum ReAct iterations.  Default 4 (one per OBSERVE/THINK/ACT/RESULT).
    """

    def __init__(
        self,
        llm:             LLMClient,
        retriever:       RAGRetriever,
        builder:         ExplanationBuilder,
        max_react_steps: int = 4,
    ) -> None:
        self.llm             = llm
        self.retriever       = retriever
        self.builder         = builder
        self.max_react_steps = max_react_steps
        self._token_counter  = TokenCounter(model=llm.model)

        # Explanation history (for the dashboard feed)
        self._history: List[ExplanationCard] = []

    # ------------------------------------------------------------------ #
    # Primary interface
    # ------------------------------------------------------------------ #

    def explain(self, context: Dict[str, Any]) -> ExplanationCard:
        """
        Generate a structured explanation for a single defender action.

        Parameters
        ----------
        context : dict
            Action context.  Required keys:
                action          : str — action description
                step            : int — episode step
                threat          : str — brief threat description
                obs_decoded     : dict — decoded observation
                risk_score      : float — CVaR risk score
                attacker_type   : str — attacker belief string
            Optional keys:
                technique_ids   : list[str]
                cve_ids         : list[str]
                action_success  : bool
                action_idx      : int
                reward_breakdown : dict

        Returns
        -------
        ExplanationCard
        """
        start  = time.monotonic()
        trace  = ReActTrace()
        action = context.get("action", "Unknown action")

        logger.info(
            "ReAct explain — action=%r, step=%d",
            action, context.get("step", 0),
        )

        # ── Step 1: OBSERVE ────────────────────────────────────────────
        obs_summary  = PromptLibrary.format_obs_summary(
            context.get("obs_decoded", {})
        )
        observe_output = self._observe(context, obs_summary, trace)

        # ── Step 2: THINK (with RAG retrieval) ────────────────────────
        retrieved_docs = self._retrieve_docs(context)
        trace.retrieved_docs = retrieved_docs
        context_str    = PromptLibrary.format_context_docs(retrieved_docs)
        think_output   = self._think(context, observe_output, context_str, trace)

        # ── Step 3: ACT ────────────────────────────────────────────────
        act_output = self._act(context, think_output, trace)

        # ── Step 4: RESULT ────────────────────────────────────────────
        result_output = self._result(context, act_output, trace)

        # ── Assemble ExplanationCard ───────────────────────────────────
        card = self.builder.build(
            context      = context,
            observe      = observe_output,
            think        = think_output,
            act          = act_output,
            result       = result_output,
            retrieved    = retrieved_docs,
            context_str  = context_str,
            obs_summary  = obs_summary,
            trace        = trace,
        )

        card.latency_s = round(time.monotonic() - start, 3)
        self._history.append(card)

        logger.info(
            "ReAct complete — latency=%.2fs, tokens=%d",
            card.latency_s, trace.total_tokens,
        )
        return card

    # ------------------------------------------------------------------ #
    # ReAct steps
    # ------------------------------------------------------------------ #

    def _observe(
        self,
        context:     Dict[str, Any],
        obs_summary: str,
        trace:       ReActTrace,
    ) -> str:
        """Step 1: Observe the current network state."""
        prompt = PromptLibrary.react_observe(
            obs_summary  = obs_summary,
            action       = context.get("action", ""),
            threat       = context.get("threat", ""),
            step         = context.get("step", 0),
        )

        messages = [
            {"role": "system", "content": PromptLibrary.system_react()},
            {"role": "user",   "content": prompt},
        ]

        resp = self.llm.chat(messages, max_tokens=200, temperature=0.1)

        trace.add_step(
            _STEP_OBSERVE,
            output        = resp.content,
            prompt_tokens = resp.prompt_tokens,
            latency_s     = resp.latency_s,
        )
        logger.debug("OBSERVE: %s", resp.content[:100])
        return resp.content

    def _think(
        self,
        context:      Dict[str, Any],
        observation:  str,
        context_docs: str,
        trace:        ReActTrace,
    ) -> str:
        """Step 2: Reason about the threat using retrieved documents."""
        prompt = PromptLibrary.react_think(
            observation   = observation,
            context_docs  = context_docs,
            technique_ids = context.get("technique_ids"),
        )

        messages = self._token_counter.build_prompt_within_budget(
            system       = PromptLibrary.system_react(),
            query        = prompt,
            context_docs = context_docs,
            max_context_tokens = 1500,
        )

        resp = self.llm.chat(messages, max_tokens=300, temperature=0.2)

        trace.add_step(
            _STEP_THINK,
            output        = resp.content,
            prompt_tokens = resp.prompt_tokens,
            latency_s     = resp.latency_s,
        )
        logger.debug("THINK: %s", resp.content[:100])
        return resp.content

    def _act(
        self,
        context:   Dict[str, Any],
        reasoning: str,
        trace:     ReActTrace,
    ) -> str:
        """Step 3: Justify the chosen action based on the reasoning."""
        prompt = PromptLibrary.react_act(
            reasoning     = reasoning,
            action        = context.get("action", ""),
            risk_score    = context.get("risk_score", 0.0),
            attacker_type = context.get("attacker_type", "Unknown"),
        )

        messages = [
            {"role": "system", "content": PromptLibrary.system_react()},
            {"role": "user",   "content": prompt},
        ]

        resp = self.llm.chat(messages, max_tokens=300, temperature=0.2)

        trace.add_step(
            _STEP_ACT,
            output        = resp.content,
            prompt_tokens = resp.prompt_tokens,
            latency_s     = resp.latency_s,
        )
        logger.debug("ACT: %s", resp.content[:100])
        return resp.content

    def _result(
        self,
        context:       Dict[str, Any],
        justification: str,
        trace:         ReActTrace,
    ) -> str:
        """Step 4: State the expected outcome and follow-up actions."""
        prompt = PromptLibrary.react_result(
            action        = context.get("action", ""),
            justification = justification,
        )

        messages = [
            {"role": "system", "content": PromptLibrary.system_react()},
            {"role": "user",   "content": prompt},
        ]

        resp = self.llm.chat(messages, max_tokens=200, temperature=0.15)

        trace.add_step(
            _STEP_RESULT,
            output        = resp.content,
            prompt_tokens = resp.prompt_tokens,
            latency_s     = resp.latency_s,
        )
        logger.debug("RESULT: %s", resp.content[:100])
        return resp.content

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def _retrieve_docs(
        self, context: Dict[str, Any]
    ) -> List[RetrievalResult]:
        """
        Build retrieval queries from the context and fetch relevant documents.

        Uses multi-query retrieval: generates separate queries for the
        threat description, technique IDs, and policy context.
        """
        queries: List[str] = []

        # Query 1: threat description
        threat = context.get("threat", "")
        if threat:
            queries.append(threat)

        # Query 2: CVE IDs
        for cve_id in context.get("cve_ids", []):
            queries.append(f"{cve_id} vulnerability exploit technique")

        # Query 3: technique IDs
        for tech_id in context.get("technique_ids", []):
            queries.append(f"{tech_id} ATT&CK technique defender policy")

        # Query 4: action policy
        action = context.get("action", "")
        if action:
            queries.append(f"ACD policy {action} cyber defence strategy")

        if not queries:
            queries = [context.get("threat", "cybersecurity incident")]

        return self.retriever.retrieve_multi_query(
            queries   = queries,
            top_k     = 5,
            doc_types = ["cve", "technique", "tactic", "policy"],
        )

    # ------------------------------------------------------------------ #
    # History
    # ------------------------------------------------------------------ #

    @property
    def history(self) -> List[ExplanationCard]:
        """List of all generated explanation cards (most recent last)."""
        return list(self._history)

    def get_recent(self, n: int = 10) -> List[ExplanationCard]:
        """Return the N most recent explanation cards."""
        return self._history[-n:]

    def clear_history(self) -> None:
        """Clear the explanation history."""
        self._history.clear()

    def __repr__(self) -> str:
        return (
            f"ReActAgent("
            f"llm={self.llm.model!r}, "
            f"n_explanations={len(self._history)})"
        )