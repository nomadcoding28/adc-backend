"""
knowledge/
==========
Cybersecurity Knowledge Graph pipeline for the ACD Framework.

Builds and queries a Neo4j graph that connects CVEs (Common Vulnerabilities
and Exposures) to MITRE ATT&CK techniques and tactics, enriching the
defender agent's observations with structured threat intelligence.

Pipeline overview
-----------------
    NVD API  ──►  nvd_fetcher.py   ──►  CVE objects
    ATT&CK   ──►  mitre_parser.py  ──►  Technique/Tactic objects
    Both     ──►  bert_mapper.py   ──►  CVE → Technique edges (BERT similarity)
    All      ──►  kg_builder.py    ──►  Neo4j graph (via neo4j_client.py)
    Graph    ──►  kg_features.py   ──►  16-dim feature vectors for agent obs
    Graph    ──►  kg_queries.py    ──►  Named Cypher queries for API routes

Public API
----------
    from knowledge import KGBuilder, KGQuerier, BERTMapper
    from knowledge import Neo4jClient, CVEFetcher, MITREParser
"""

from knowledge.neo4j_client import Neo4jClient
from knowledge.nvd_fetcher import CVEFetcher, CVERecord
from knowledge.mitre_parser import MITREParser, TechniqueRecord, TacticRecord
from knowledge.bert_mapper import BERTMapper, MappingResult
from knowledge.kg_builder import KGBuilder
from knowledge.kg_features import KGFeatureExtractor
from knowledge.kg_queries import KGQuerier
from knowledge.cache import DiskCache

__all__ = [
    "Neo4jClient",
    "CVEFetcher",
    "CVERecord",
    "MITREParser",
    "TechniqueRecord",
    "TacticRecord",
    "BERTMapper",
    "MappingResult",
    "KGBuilder",
    "KGFeatureExtractor",
    "KGQuerier",
    "DiskCache",
]