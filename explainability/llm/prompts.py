"""
explainability/llm/prompts.py
==============================
All LLM prompt templates for the ACD explanation pipeline.

Design principle: NO prompt strings outside this file.
Every template lives here so they can be versioned, tested, and
swapped without touching business logic.

Templates use Python str.format() with named placeholders:
    {action}         Defender action taken (e.g. "Isolate Host-3")
    {threat}         Threat description (e.g. "CVE-2021-44228 detected")
    {context_docs}   Retrieved RAG documents (formatted string)
    {obs_summary}    Decoded observation summary
    {risk_score}     CVaR risk score
    {attacker_type}  Attacker type belief (e.g. "Targeted APT 71%")
    {step}           Current episode step number
    {technique_ids}  Detected ATT&CK technique IDs

Usage
-----
    from explainability.llm.prompts import PromptLibrary

    system = PromptLibrary.system_explanation()
    user   = PromptLibrary.action_explanation(
        action        = "Isolate Host-3",
        threat        = "CVE-2021-44228 Log4Shell detected",
        context_docs  = formatted_docs,
        obs_summary   = "3 hosts compromised, attacker on subnet B",
        risk_score    = 0.87,
        attacker_type = "Targeted APT (71%)",
        step          = 4821,
        technique_ids = ["T1190", "T1059"],
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class PromptLibrary:
    """
    Static library of all LLM prompt templates.

    All methods are ``@staticmethod`` — no instantiation needed.
    """

    # ------------------------------------------------------------------ #
    # System prompts
    # ------------------------------------------------------------------ #

    @staticmethod
    def system_explanation() -> str:
        """
        System prompt for the action explanation task.

        Sets the LLM's persona as a cybersecurity analyst who explains
        AI defender decisions in structured, human-readable format.
        """
        return (
            "You are a senior cybersecurity analyst explaining the decisions of an "
            "autonomous AI cyber defence system to a Security Operations Centre (SOC) "
            "team.\n\n"
            "Your explanations must be:\n"
            "- Factual: grounded in the provided CVE and ATT&CK documentation\n"
            "- Concise: 2-4 sentences per section, no padding\n"
            "- Structured: always use the exact section headers provided\n"
            "- Technical: use correct cybersecurity terminology\n"
            "- Honest: if the reason for an action is uncertain, say so\n\n"
            "Do not invent CVE IDs, technique IDs, or facts not present in the context."
        )

    @staticmethod
    def system_incident_report() -> str:
        """System prompt for incident report generation."""
        return (
            "You are a cybersecurity incident response analyst writing formal "
            "incident reports for a Security Operations Centre.\n\n"
            "Write in clear, professional language suitable for both technical "
            "and managerial audiences. Structure reports with the exact sections "
            "requested. Be specific about timeline, impact, and remediation. "
            "Ground all statements in the provided context data."
        )

    @staticmethod
    def system_react() -> str:
        """System prompt for the ReAct reasoning loop."""
        return (
            "You are a cybersecurity reasoning agent that explains AI defender "
            "decisions using a structured Observe-Think-Act-Observe cycle.\n\n"
            "For each step:\n"
            "  OBSERVE: State what you see in the network and threat context\n"
            "  THINK:   Reason about the threat using CVE/ATT&CK knowledge\n"
            "  ACT:     Explain the defender's chosen action and why\n"
            "  RESULT:  State the expected outcome of the action\n\n"
            "Use only information from the provided context documents. "
            "Be concise and technically precise."
        )

    # ------------------------------------------------------------------ #
    # Action explanation prompt
    # ------------------------------------------------------------------ #

    @staticmethod
    def action_explanation(
        action:        str,
        threat:        str,
        context_docs:  str,
        obs_summary:   str,
        risk_score:    float,
        attacker_type: str,
        step:          int,
        technique_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Full user prompt for explaining a single defender action.

        Parameters
        ----------
        action : str
            The action taken, e.g. ``"Isolate Host-3"``.
        threat : str
            Brief threat description, e.g. ``"CVE-2021-44228 Log4Shell detected"``.
        context_docs : str
            Formatted retrieved documents (from ``format_context_docs()``).
        obs_summary : str
            Human-readable observation summary.
        risk_score : float
            CVaR risk score for this step (0–1).
        attacker_type : str
            Attacker type belief string.
        step : int
            Episode step number.
        technique_ids : list[str], optional
            Detected ATT&CK technique IDs.

        Returns
        -------
        str
        """
        tech_str = (
            ", ".join(technique_ids) if technique_ids
            else "None detected"
        )

        return f"""
## DEFENDER ACTION EXPLANATION REQUEST

**Episode Step:** {step}
**Action Taken:** {action}
**Threat Detected:** {threat}
**CVaR Risk Score:** {risk_score:.3f}
**Attacker Type Belief:** {attacker_type}
**Detected ATT&CK Techniques:** {tech_str}

## CURRENT NETWORK STATE
{obs_summary}

## RETRIEVED CONTEXT DOCUMENTS
{context_docs}

---

Please explain the defender's decision using exactly these four sections:

### THREAT DETECTED
Describe what attack was detected, which CVE(s) or techniques are involved,
and the severity level.

### WHY THIS ACTION
Explain specifically why the AI agent chose {action} over alternatives.
Reference the attacker type, risk score, and relevant policy/technique context.

### RISK MITIGATED
What catastrophic outcome does this action prevent?
How does it reduce the tail-risk probability?

### RECOMMENDED FOLLOW-UP
What should the defender monitor or do next based on the current threat?
""".strip()

    # ------------------------------------------------------------------ #
    # ReAct loop prompts
    # ------------------------------------------------------------------ #

    @staticmethod
    def react_observe(
        obs_summary:   str,
        action:        str,
        threat:        str,
        step:          int,
    ) -> str:
        """Prompt for the OBSERVE step in the ReAct loop."""
        return (
            f"STEP {step} — OBSERVE\n\n"
            f"Network state: {obs_summary}\n"
            f"Threat detected: {threat}\n"
            f"Action under analysis: {action}\n\n"
            f"Summarise what is happening in 1-2 sentences."
        )

    @staticmethod
    def react_think(
        observation:  str,
        context_docs: str,
        technique_ids: Optional[List[str]] = None,
    ) -> str:
        """Prompt for the THINK step — reasoning about the threat."""
        tech_str = (
            f"Detected techniques: {', '.join(technique_ids)}\n"
            if technique_ids else ""
        )
        return (
            f"THINK — Reason about the threat:\n\n"
            f"{tech_str}"
            f"Observation: {observation}\n\n"
            f"Context documents:\n{context_docs}\n\n"
            f"Based on the context, what type of attack is this? "
            f"What kill-chain stage is the attacker at? "
            f"What are they likely to do next? "
            f"Answer in 2-3 sentences."
        )

    @staticmethod
    def react_act(
        reasoning:    str,
        action:       str,
        risk_score:   float,
        attacker_type: str,
    ) -> str:
        """Prompt for the ACT step — justifying the chosen action."""
        return (
            f"ACT — Explain the chosen action:\n\n"
            f"Reasoning: {reasoning}\n"
            f"Action chosen: {action}\n"
            f"CVaR risk score: {risk_score:.3f}\n"
            f"Attacker type: {attacker_type}\n\n"
            f"In 2-3 sentences, explain why {action} was selected over alternatives "
            f"given the threat analysis above."
        )

    @staticmethod
    def react_result(
        action:       str,
        justification: str,
    ) -> str:
        """Prompt for the RESULT step — expected outcome."""
        return (
            f"RESULT — Expected outcome:\n\n"
            f"Action: {action}\n"
            f"Justification: {justification}\n\n"
            f"In 1-2 sentences, state the expected outcome of {action} and "
            f"what the defender should monitor next."
        )

    # ------------------------------------------------------------------ #
    # Incident report prompts
    # ------------------------------------------------------------------ #

    @staticmethod
    def incident_report(
        incident_id:   str,
        title:         str,
        timeline:      List[Dict[str, Any]],
        cve_ids:       List[str],
        technique_ids: List[str],
        hosts_affected: List[str],
        actions_taken: List[str],
        forgetting_delta: Optional[float] = None,
        drift_detected:   bool = False,
    ) -> str:
        """
        Prompt for generating a full incident report.

        Parameters
        ----------
        incident_id : str
        title : str
        timeline : list[dict]
            List of ``{"timestamp": ..., "event": ...}`` dicts.
        cve_ids : list[str]
        technique_ids : list[str]
        hosts_affected : list[str]
        actions_taken : list[str]
        forgetting_delta : float, optional
            EWC forgetting metric after this incident's drift event.
        drift_detected : bool
            Whether a concept drift event was associated with this incident.

        Returns
        -------
        str
        """
        timeline_str = "\n".join(
            f"  {e.get('timestamp', '')}: {e.get('event', '')}"
            for e in timeline
        )

        drift_section = ""
        if drift_detected:
            forgetting_str = (
                f" (EWC forgetting delta: {forgetting_delta:.4f})"
                if forgetting_delta is not None else ""
            )
            drift_section = (
                f"\n**Concept Drift:** A distribution shift was detected during "
                f"this incident.  The EWC continual learning module registered a "
                f"new task and adapted the agent's policy{forgetting_str}."
            )

        return f"""
Generate a formal cybersecurity incident report with these exact sections:

**Incident ID:** {incident_id}
**Title:** {title}

**Data:**
- CVEs involved: {', '.join(cve_ids) or 'None identified'}
- ATT&CK techniques: {', '.join(technique_ids) or 'None identified'}
- Hosts affected: {', '.join(hosts_affected) or 'None'}
- Defender actions: {', '.join(actions_taken) or 'None'}{drift_section}

**Timeline:**
{timeline_str}

---

Write the report with these sections:
1. EXECUTIVE SUMMARY (2-3 sentences for non-technical management)
2. TECHNICAL DETAILS (CVEs, techniques, attack vector — for SOC analysts)
3. IMPACT ASSESSMENT (what was compromised or at risk)
4. DEFENDER RESPONSE (actions taken and why, referencing AI decision logic)
5. TIMELINE (structured list of events)
6. RECOMMENDATIONS (what to monitor, patch, or harden going forward)
""".strip()

    # ------------------------------------------------------------------ #
    # Context document formatter
    # ------------------------------------------------------------------ #

    @staticmethod
    def format_context_docs(
        results: List[Any],   # list[RetrievalResult]
        max_chars_per_doc: int = 400,
    ) -> str:
        """
        Format a list of ``RetrievalResult`` objects into a readable string
        for inclusion in prompts.

        Parameters
        ----------
        results : list[RetrievalResult]
        max_chars_per_doc : int
            Truncate each document's content to this many characters.

        Returns
        -------
        str
        """
        if not results:
            return "No relevant context documents retrieved."

        parts = []
        for r in results:
            content = r.content
            if len(content) > max_chars_per_doc:
                content = content[:max_chars_per_doc] + "..."

            parts.append(
                f"[{r.rank}] SOURCE: {r.doc_id} (type={r.doc_type}, "
                f"similarity={r.score:.3f})\n{content}"
            )

        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    # Observation summary formatter
    # ------------------------------------------------------------------ #

    @staticmethod
    def format_obs_summary(obs_decoded: Dict[str, Any]) -> str:
        """
        Convert a decoded observation dict into a compact prose summary.

        Parameters
        ----------
        obs_decoded : dict
            Output of ``ObservationProcessor.decode()``.

        Returns
        -------
        str
        """
        hosts      = obs_decoded.get("hosts", {})
        feedback   = obs_decoded.get("action_feedback", {})

        compromised = [h for h, s in hosts.items() if s.get("compromised")]
        decoys      = [h for h, s in hosts.items() if s.get("is_decoy")]
        malicious   = [h for h, s in hosts.items() if s.get("malicious_process")]
        privileged  = [h for h, s in hosts.items() if s.get("privileged_session")]

        parts = []
        if compromised:
            parts.append(f"Compromised hosts: {', '.join(compromised)}")
        if malicious:
            parts.append(f"Malicious processes: {', '.join(malicious)}")
        if decoys:
            parts.append(f"Active decoys: {', '.join(decoys)}")
        if privileged:
            parts.append(f"Privileged sessions: {', '.join(privileged)}")

        step_frac = feedback.get("step_fraction", 0.0)
        parts.append(
            f"Episode progress: {step_frac:.0%}. "
            f"Last action: {feedback.get('last_action_type', 'Unknown')} "
            f"({'succeeded' if feedback.get('last_action_success') else 'failed'})."
        )

        return " | ".join(parts) if parts else "Network state: all hosts clean."