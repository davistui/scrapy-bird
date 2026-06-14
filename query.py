#!/usr/bin/env python3
"""Query the RAG database for scraped documentation.

Usage:
    python query.py --name eptura "how does visitor management work?"
    python query.py --name eptura   # interactive mode
"""

import argparse
import math
import os
import re
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

# Bi-encoder for fast vector recall; cross-encoder for precise reranking.
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


class DocQuery:
    def __init__(self, name, output_base="..", quiet=False):
        project_dir = Path(output_base) / name
        db_path = project_dir / "chroma_db"

        if not db_path.is_dir():
            print(f"Error: No database found at {db_path}")
            print(f"Run: python process.py --name {name}")
            sys.exit(1)

        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False)
        )

        collection_name = f"{name}_docs"
        try:
            self.collection = self.client.get_collection(collection_name)
        except Exception:
            print(f"Error: Collection '{collection_name}' not found in {db_path}")
            sys.exit(1)

        if not quiet:
            print("Loading models...")
        self.embedder = SentenceTransformer(EMBED_MODEL)

        # The cross-encoder reranker is the precision win, but it's optional:
        # if its model can't load (offline / not yet downloaded) or is disabled
        # via QUERY_NO_RERANK=1, fall back to RRF fusion of vector+BM25 so search
        # still works — just a bit less precise.
        self.reranker = None
        if os.environ.get("QUERY_NO_RERANK") != "1":
            try:
                self.reranker = CrossEncoder(RERANK_MODEL)
            except Exception as e:
                if not quiet:
                    print(f"(reranker unavailable — using vector+BM25 fusion: {e})")

        # Pull the full corpus once to back BM25 keyword search (catches exact
        # product/feature names that the embedding blurs). Falls back to
        # vector-only if rank-bm25 isn't installed.
        data = self.collection.get(include=["documents", "metadatas"])
        self._ids = data["ids"]
        self._docs = data["documents"]
        self._metas = data["metadatas"]
        self._bm25 = BM25Okapi([_tokenize(d) for d in self._docs]) if BM25Okapi and self._docs else None

        if not quiet:
            parts = ["vector"] + (["BM25"] if self._bm25 else [])
            parts.append("reranker" if self.reranker else "RRF-fusion")
            print(f"Ready! {len(self._ids):,} chunks indexed for '{name}' ({' + '.join(parts)}).\n")

    def search(self, query, n_results=5, category=None, n_candidates=25):
        """Hybrid retrieval: vector + BM25 recall, fused, then reranked.

        Pull a candidate pool two ways (dense vectors and keyword BM25) and fuse
        them with Reciprocal Rank Fusion. If the cross-encoder is available, it
        reranks the merged pool for final precision (relevance = sigmoid of its
        score); otherwise the RRF order stands.
        """
        pool = min(n_candidates, max(len(self._ids), 1))
        candidates = {}  # id -> {text, metadata, rrf}
        K = 60  # RRF dampening constant (standard default)

        def add(cid, text, meta, rank):
            c = candidates.get(cid)
            if c is None:
                c = candidates[cid] = {"text": text, "metadata": meta, "rrf": 0.0}
            c["rrf"] += 1.0 / (K + rank)

        # Dense vector recall (already rank-ordered)
        embedding = self.embedder.encode([query]).tolist()
        where = {"category": category} if category else None
        vres = self.collection.query(query_embeddings=embedding, n_results=pool, where=where)
        for rank in range(len(vres["ids"][0])):
            add(vres["ids"][0][rank], vres["documents"][0][rank], vres["metadatas"][0][rank], rank)

        # Keyword BM25 recall (rank-ordered)
        if self._bm25 is not None:
            scores = self._bm25.get_scores(_tokenize(query))
            ranked = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
            rank = 0
            for j in ranked:
                if rank >= pool or scores[j] <= 0:
                    break
                meta = self._metas[j]
                if category and meta.get("category") != category:
                    continue
                add(self._ids[j], self._docs[j], meta, rank)
                rank += 1

        if not candidates:
            return []

        items = list(candidates.values())
        if self.reranker is not None:
            pairs = [[query, it["text"]] for it in items]
            for it, s in zip(items, self.reranker.predict(pairs)):
                it["relevance"] = 1.0 / (1.0 + math.exp(-float(s)))
        else:
            top = max(it["rrf"] for it in items)
            for it in items:
                it["relevance"] = it["rrf"] / top if top else 0.0
        items.sort(key=lambda it: it["relevance"], reverse=True)
        return items[:n_results]

    def format_results(self, query, results):
        """Format results for terminal display."""
        lines = [f"\nResults for: '{query}'", "=" * 50]

        if not results:
            lines.append("No results found.")
            return "\n".join(lines)

        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            title = meta.get("title", "Untitled")
            source = meta.get("source", "")
            category = meta.get("category", "")
            score = r["relevance"]

            lines.append(f"\n{i}. [{category}] {title} (relevance: {score:.3f})")
            if source:
                lines.append(f"   Source: {source}")
            preview = r["text"][:300]
            if len(r["text"]) > 300:
                preview += "..."
            lines.append(f"   {preview}")

        return "\n".join(lines)

    def format_for_agent(self, query, results):
        """Lean machine-readable output: full untruncated chunks, no decorative UI.

        Minified XML, no relevance scores or previews. Each chunk already carries a
        situational context line (baked in at index time by Contextual Retrieval),
        so path + source are enough to cite without a follow-up lookup.
        """
        def attr(s):
            return (str(s).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;"))

        lines = [f'<results query="{attr(query)}" count="{len(results)}">']
        for r in results:
            meta = r["metadata"]
            lines.append(
                f'<chunk path="{attr(meta.get("path", ""))}"'
                f' source="{attr(meta.get("source", ""))}"'
                f' index="{attr(meta.get("chunk_index", 0))}">'
            )
            lines.append(r["text"])
            lines.append("</chunk>")
        lines.append("</results>")
        return "\n".join(lines)

    def interactive(self):
        """Interactive query loop."""
        print("Interactive mode. Type a question, or /quit to exit.\n")
        while True:
            try:
                query = input(f"query> ").strip()
                if not query:
                    continue
                if query == "/quit":
                    break
                if query == "/stats":
                    print(f"Chunks indexed: {self.collection.count()}")
                    continue
                if query == "/help":
                    print("  <question>  Search the knowledge base")
                    print("  /stats      Show database stats")
                    print("  /quit       Exit")
                    continue

                results = self.search(query)
                print(self.format_results(query, results))
                print()

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Query RAG database for scraped docs")
    parser.add_argument("--name", required=True, help="Project name")
    parser.add_argument("--output", default="output", help="Base output directory (default: output)")
    parser.add_argument("-n", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--candidates", type=int, default=25,
                        help="Recall pool size per backend before reranking (default: 25)")
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--agent", action="store_true",
                        help="Lean machine-readable output (XML, full chunks, no decorative UI)")
    parser.add_argument("query", nargs="*", help="Search query (omit for interactive mode)")
    args = parser.parse_args()

    dq = DocQuery(args.name, args.output, quiet=args.agent)

    if args.query:
        query = " ".join(args.query)
        results = dq.search(query, n_results=args.n, category=args.category, n_candidates=args.candidates)
        if args.agent:
            print(dq.format_for_agent(query, results))
        else:
            print(dq.format_results(query, results))
    else:
        dq.interactive()


if __name__ == "__main__":
    main()
