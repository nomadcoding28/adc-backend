"""
scripts/build_embeddings.py
============================
Build the FAISS vector index used by the RAG explainability pipeline.

Reads from:
    data/kg_cache/nvd_cves.json          (from download_nvd.py)
    data/kg_cache/attck_parsed.json      (from download_attck.py)

Writes to:
    data/embeddings/rag_index.faiss      (FAISS binary index)
    data/embeddings/rag_store.json       (document store JSON)
    data/embeddings/rag_index.meta.json  (build metadata)

Usage
-----
    # Build with default settings
    python scripts/build_embeddings.py

    # Specify custom paths
    python scripts/build_embeddings.py \\
        --nvd-cache   data/kg_cache/nvd_cves.json \\
        --attck-cache data/kg_cache/attck_parsed.json \\
        --index-path  data/embeddings/rag_index.faiss \\
        --store-path  data/embeddings/rag_store.json

    # Override embedding model and CVE filter
    python scripts/build_embeddings.py \\
        --model all-mpnet-base-v2 \\
        --max-cves 2000 \\
        --min-cvss 6.0

    # Force rebuild even if index exists
    python scripts/build_embeddings.py --force
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.structlog_config import configure_logging
from utils.timer import Timer

logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL      = "all-MiniLM-L6-v2"
DEFAULT_NVD_CACHE  = "data/kg_cache/nvd_cves.json"
DEFAULT_ATTCK_CACHE= "data/kg_cache/attck_parsed.json"
DEFAULT_INDEX_PATH = "data/embeddings/rag_index.faiss"
DEFAULT_STORE_PATH = "data/embeddings/rag_store.json"
DEFAULT_MAX_CVES   = 500
DEFAULT_MIN_CVSS   = 6.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build FAISS RAG index from KG cache files."
    )
    p.add_argument("--nvd-cache",   default=DEFAULT_NVD_CACHE)
    p.add_argument("--attck-cache", default=DEFAULT_ATTCK_CACHE)
    p.add_argument("--index-path",  default=DEFAULT_INDEX_PATH)
    p.add_argument("--store-path",  default=DEFAULT_STORE_PATH)
    p.add_argument("--model",       default=DEFAULT_MODEL,    help="Sentence-transformers model name")
    p.add_argument("--max-cves",    type=int,   default=DEFAULT_MAX_CVES)
    p.add_argument("--min-cvss",    type=float, default=DEFAULT_MIN_CVSS)
    p.add_argument("--force",       action="store_true", help="Rebuild even if index exists")
    p.add_argument("--verbose",     action="store_true")
    return p.parse_args()


def check_prerequisites(nvd_cache: str, attck_cache: str) -> bool:
    """
    Verify that the required cache files exist before building.

    Returns True if both files exist, False otherwise.
    """
    ok = True
    for path, name in [(nvd_cache, "NVD cache"), (attck_cache, "ATT&CK cache")]:
        p = Path(path)
        if p.exists():
            size_kb = p.stat().st_size / 1024
            logger.info("✓ %s found: %s (%.0f KB)", name, p, size_kb)
        else:
            logger.error(
                "✗ %s NOT found: %s\n"
                "  Run: python scripts/download_%s.py",
                name, p, "nvd" if "nvd" in path else "attck",
            )
            ok = False
    return ok


def main() -> None:
    args = parse_args()
    configure_logging(level="DEBUG" if args.verbose else "INFO")

    # Check if index already exists
    index_path = Path(args.index_path)
    store_path = Path(args.store_path)
    meta_path  = index_path.with_suffix(".meta.json")

    if store_path.exists() and not args.force:
        try:
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            n_docs = meta.get("n_docs", "?")
            logger.info(
                "RAG index already exists at %s (%s docs). "
                "Use --force to rebuild.",
                store_path, n_docs,
            )
            return
        except Exception:
            pass

    # Verify prerequisites
    logger.info("="*60)
    logger.info("BUILDING RAG FAISS INDEX")
    logger.info("="*60)
    logger.info("Model:       %s", args.model)
    logger.info("NVD cache:   %s", args.nvd_cache)
    logger.info("ATT&CK cache:%s", args.attck_cache)
    logger.info("Max CVEs:    %d (CVSS ≥ %.1f)", args.max_cves, args.min_cvss)
    logger.info("Index path:  %s", args.index_path)
    logger.info("="*60)

    if not check_prerequisites(args.nvd_cache, args.attck_cache):
        logger.error(
            "\nMissing cache files. Run these first:\n"
            "  python scripts/download_nvd.py\n"
            "  python scripts/download_attck.py\n"
            "Or run: make kg-build"
        )
        sys.exit(1)

    # Build the index
    with Timer("RAG index build") as total_timer:
        from explainability.rag.document_store import DocumentStore
        from explainability.rag.embedder import Embedder
        from explainability.rag.indexer import FAISSIndexer

        config = {
            "index_path":  args.index_path,
            "store_path":  args.store_path,
            "nvd_cache":   args.nvd_cache,
            "attck_cache": args.attck_cache,
            "max_cves":    args.max_cves,
            "min_cvss":    args.min_cvss,
        }

        store    = DocumentStore()
        embedder = Embedder(model_name=args.model)
        indexer  = FAISSIndexer(store=store, embedder=embedder, config=config)

        # Stage 1: Load documents
        logger.info("\n[1/3] Loading documents from cache files...")
        with Timer("document loading"):
            n_cves    = store.load_from_kg_cache(
                nvd_cache_path = args.nvd_cache,
                max_cves       = args.max_cves,
                min_cvss       = args.min_cvss,
            )
            n_attck   = store.load_attck_techniques(
                attck_cache_path = args.attck_cache,
            )
            n_policy  = store.load_acd_policies()

        logger.info(
            "  Loaded: %d CVEs, %d ATT&CK docs, %d policies → %d total",
            n_cves, n_attck, n_policy, store.n_docs,
        )

        # Stage 2: Embed documents
        logger.info("\n[2/3] Embedding %d documents with %s...", store.n_docs, args.model)
        with Timer("embedding") as embed_timer:
            n_embedded = embedder.embed_store(store)

        logger.info(
            "  Embedded %d documents in %.1fs (%.0f docs/s)",
            n_embedded, embed_timer.elapsed_s,
            n_embedded / max(embed_timer.elapsed_s, 0.001),
        )

        # Stage 3: Build FAISS index
        logger.info("\n[3/3] Building FAISS index...")
        with Timer("FAISS index build"):
            n_indexed = indexer._build_faiss_index()
            indexer._save()

        logger.info("  Indexed %d vectors", n_indexed)

    # Print summary
    print("\n" + "="*60)
    print("RAG INDEX BUILD COMPLETE")
    print("="*60)
    print(f"  Total documents:    {store.n_docs}")
    by_type = store.stats()["by_type"]
    for doc_type, count in sorted(by_type.items()):
        print(f"    {doc_type:<15}  {count}")
    print(f"  Embedding model:    {args.model}")
    print(f"  Embedding dim:      {embedder.dim}")
    print(f"  Total build time:   {total_timer.elapsed_s:.1f}s")
    print(f"  Index path:         {args.index_path}")
    print(f"  Store path:         {args.store_path}")

    # Read and show meta
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"  Built at:           {meta.get('built_at', 'unknown')}")

    print("="*60)
    print("\n✓ RAG index ready. The explainability pipeline can now use it.")
    print("  Start the API: make api")


if __name__ == "__main__":
    main()