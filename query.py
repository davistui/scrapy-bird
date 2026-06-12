#!/usr/bin/env python3
"""Query the RAG database for scraped documentation.

Usage:
    python query.py --name eptura "how does visitor management work?"
    python query.py --name eptura   # interactive mode
"""

import argparse
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


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
            print(f"Loading embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        count = self.collection.count()
        if not quiet:
            print(f"Ready! {count:,} chunks indexed for '{name}'.\n")

    def search(self, query, n_results=5, category=None):
        """Semantic search, returns list of result dicts."""
        embedding = self.embedder.encode([query]).tolist()

        where = {"category": category} if category else None
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=n_results,
            where=where
        )

        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "relevance": 1 - results["distances"][0][i],
            })
        return formatted

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
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--agent", action="store_true",
                        help="Lean machine-readable output (XML, full chunks, no decorative UI)")
    parser.add_argument("query", nargs="*", help="Search query (omit for interactive mode)")
    args = parser.parse_args()

    dq = DocQuery(args.name, args.output, quiet=args.agent)

    if args.query:
        query = " ".join(args.query)
        results = dq.search(query, n_results=args.n, category=args.category)
        if args.agent:
            print(dq.format_for_agent(query, results))
        else:
            print(dq.format_results(query, results))
    else:
        dq.interactive()


if __name__ == "__main__":
    main()
