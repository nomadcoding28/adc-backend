"""
explainability/
===============
LLM-powered explanation pipeline for the ACD Framework.

Every defensive action taken by the RL agent is explained in plain English
using a RAG (Retrieval-Augmented Generation) + ReAct (Reason + Act) pipeline.
Explanations are structured, human-readable, and grounded in real CVE and
ATT&CK documentation — not hallucinated.

Pipeline overview
-----------------
    Agent action
        ↓
    ReActAgent.explain()
        ├── retrieve()  →  RAGRetriever  → FAISS top-k docs
        ├── think()     →  LLMClient     → chain-of-thought reasoning
        ├── act()       →  ExplanationBuilder → structured card
        └── output      →  ExplanationCard (JSON-serialisable)

    ExplanationCard
        ↓
    ReportGenerator.generate_incident_report()
        └──  Markdown / PDF incident report

Sub-packages
------------
    rag/            Document store, embedder, retriever, indexer
    llm/            LLM client, prompt templates, token counter

Public API
----------
    from explainability import ExplainabilityPipeline
    from explainability import ExplanationCard, IncidentReport
    from explainability.rag import RAGRetriever, DocumentStore
    from explainability.llm import LLMClient
"""

from explainability.react_agent import ReActAgent
from explainability.explanation_builder import ExplanationBuilder, ExplanationCard
from explainability.report_generator import ReportGenerator, IncidentReport
from explainability.rag.retriever import RAGRetriever
from explainability.rag.document_store import DocumentStore
from explainability.rag.embedder import Embedder
from explainability.rag.indexer import FAISSIndexer
from explainability.llm.client import LLMClient
from explainability.llm.prompts import PromptLibrary
from explainability.llm.token_counter import TokenCounter


class ExplainabilityPipeline:
    """
    Convenience façade that wires all sub-components together.

    Instantiate this once and call ``explain()`` for every agent action.

    Parameters
    ----------
    config : dict
        Full explainability config.  Keys:
            llm.provider        : "openai", "anthropic", "ollama"
            llm.model           : model name string
            llm.api_key         : API key (or use env var)
            rag.index_path      : path to FAISS index
            rag.top_k           : documents to retrieve per query
            rag.embedder_model  : sentence-transformers model name
    """

    def __init__(self, config: dict) -> None:
        from explainability.rag.document_store import DocumentStore
        from explainability.rag.embedder import Embedder
        from explainability.rag.retriever import RAGRetriever
        from explainability.llm.client import LLMClient
        from explainability.explanation_builder import ExplanationBuilder

        llm_cfg = config.get("llm", {})
        rag_cfg = config.get("rag", {})

        self._llm      = LLMClient.from_config(llm_cfg)
        self._store    = DocumentStore()
        self._embedder = Embedder(model_name=rag_cfg.get("embedder_model", "all-MiniLM-L6-v2"))
        self._retriever = RAGRetriever(
            store   = self._store,
            embedder= self._embedder,
            top_k   = rag_cfg.get("top_k", 5),
        )
        self._builder = ExplanationBuilder(
            llm       = self._llm,
            retriever = self._retriever,
        )
        self._agent = ReActAgent(
            llm       = self._llm,
            retriever = self._retriever,
            builder   = self._builder,
        )
        self._reporter = ReportGenerator(llm=self._llm)

    def explain(self, context: dict) -> "ExplanationCard":
        """Generate an explanation for a single agent action."""
        return self._agent.explain(context)

    def generate_report(self, incident_context: dict) -> "IncidentReport":
        """Generate a full incident report."""
        return self._reporter.generate(incident_context)


__all__ = [
    "ExplainabilityPipeline",
    "ReActAgent",
    "ExplanationBuilder",
    "ExplanationCard",
    "ReportGenerator",
    "IncidentReport",
    "RAGRetriever",
    "DocumentStore",
    "Embedder",
    "FAISSIndexer",
    "LLMClient",
    "PromptLibrary",
    "TokenCounter",
]