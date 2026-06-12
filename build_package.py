#!/usr/bin/env python3
"""Package a scraped corpus into a self-contained, portable query kit.

Writes CLAUDE.md, a standalone self-locating query.py, and a trimmed
requirements.txt into output/<name>/ so the folder can be moved anywhere and
queried on its own, with no dependency on this repo.

Usage:
    python build_package.py --name couchbase
"""

import argparse
from pathlib import Path


# A self-contained query tool. Finds ./chroma_db next to itself, so it runs
# from wherever the folder is moved to. Written verbatim (raw string) so its
# own f-strings and escapes survive into the generated file.
STANDALONE_QUERY = r'''#!/usr/bin/env python3
"""Standalone semantic search over this folder's RAG database.

Self-contained: finds ./chroma_db next to this script. Run from anywhere:
    python query.py "your question"
    python query.py --agent "your question"   # lean XML for AI agents
    python query.py                            # interactive mode
"""

import argparse
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent


class DocQuery:
    def __init__(self, quiet=False):
        db_path = HERE / "chroma_db"
        if not db_path.is_dir():
            print(f"Error: no chroma_db found next to this script ({db_path}).")
            sys.exit(1)

        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        cols = self.client.list_collections()
        if not cols:
            print(f"Error: no collection found in {db_path}.")
            sys.exit(1)
        self.collection = self.client.get_collection(cols[0].name)

        if not quiet:
            print("Loading embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        if not quiet:
            print(f"Ready! {self.collection.count():,} chunks indexed.\n")

    def search(self, query, n_results=5, category=None):
        embedding = self.embedder.encode([query]).tolist()
        where = {"category": category} if category else None
        results = self.collection.query(
            query_embeddings=embedding, n_results=n_results, where=where
        )
        out = []
        for i in range(len(results["ids"][0])):
            out.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "relevance": 1 - results["distances"][0][i],
            })
        return out

    def format_human(self, query, results):
        lines = [f"\nResults for: '{query}'", "=" * 50]
        if not results:
            lines.append("No results found.")
            return "\n".join(lines)
        for i, r in enumerate(results, 1):
            m = r["metadata"]
            lines.append(f"\n{i}. [{m.get('category', '')}] {m.get('title', 'Untitled')} "
                         f"(relevance: {r['relevance']:.3f})")
            if m.get("source"):
                lines.append(f"   Source: {m['source']}")
            preview = r["text"][:300] + ("..." if len(r["text"]) > 300 else "")
            lines.append(f"   {preview}")
        return "\n".join(lines)

    def format_for_agent(self, query, results):
        def attr(s):
            return str(s).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
        lines = [f'<results query="{attr(query)}" count="{len(results)}">']
        for r in results:
            m = r["metadata"]
            lines.append(
                f'<chunk path="{attr(m.get("path", ""))}"'
                f' source="{attr(m.get("source", ""))}"'
                f' index="{attr(m.get("chunk_index", 0))}">'
            )
            lines.append(r["text"])
            lines.append("</chunk>")
        lines.append("</results>")
        return "\n".join(lines)

    def interactive(self):
        print("Interactive mode. Type a question, or /quit to exit.\n")
        while True:
            try:
                q = input("query> ").strip()
                if not q:
                    continue
                if q == "/quit":
                    break
                print(self.format_human(q, self.search(q)))
                print()
            except (KeyboardInterrupt, EOFError):
                break
        print("\nDone.")


def main():
    p = argparse.ArgumentParser(description="Query this folder's RAG database")
    p.add_argument("-n", type=int, default=5, help="Number of results (default: 5)")
    p.add_argument("--category", help="Filter by category")
    p.add_argument("--agent", action="store_true",
                   help="Lean machine-readable XML output (full chunks, no decoration)")
    p.add_argument("query", nargs="*", help="Search query (omit for interactive mode)")
    args = p.parse_args()

    dq = DocQuery(quiet=args.agent)
    if args.query:
        q = " ".join(args.query)
        results = dq.search(q, n_results=args.n, category=args.category)
        print(dq.format_for_agent(q, results) if args.agent else dq.format_human(q, results))
    else:
        dq.interactive()


if __name__ == "__main__":
    main()
'''


REQUIREMENTS = "chromadb\nsentence-transformers\n"


CLAUDE_MD_TEMPLATE = '''# CLAUDE.md — {display} Knowledge Base

This folder is a **self-contained, portable RAG knowledge base** of the {display}
documentation. Your job here is to **answer questions grounded in these docs** —
not to guess from memory.

## What's here
- `index.md`      — macro-map: every section, file, and its headings. **Scan this first** to locate a topic.
- `docs/`         — the source markdown files (full text), organized by section.
- `chroma_db/`    — vector database for semantic search.
- `query.py`      — standalone semantic search tool (self-locating; no flags needed).
- `_combined.md`  — every doc in one file. **Do NOT load this into context** (very large).

## How to answer questions (JIT retrieval)
1. Scan `index.md` to find the relevant section/file.
2. Retrieve with the lean agent output:
   ```bash
   python query.py --agent "the user's question"
   ```
   Returns `<chunk path="..." source="..." index="...">` blocks. Each chunk
   starts with a one-line note saying which doc it came from, so you can cite
   without a follow-up lookup.
3. Answer **from the retrieved chunks** and **cite the `source` URL**.
4. Need full detail? `Read` the specific file under `docs/` named by the chunk's `path`.

Rules:
- Never load `_combined.md` wholesale — it dwarfs the context window.
- Never answer version-specific or API details from memory — retrieve first.
- If retrieval is empty or conflicting, say so; don't fabricate.

## Setup (first run in a new environment)
Python 3 with two packages:
```bash
pip install -r requirements.txt
```
The embedding model (`all-MiniLM-L6-v2`) downloads automatically on first query.
'''


def display_name(name):
    return name.replace("-", " ").replace("_", " ").title()


def build_package(name, output_base):
    project_dir = Path(output_base) / name
    if not project_dir.is_dir():
        print(f"Error: project folder not found: {project_dir}")
        return

    (project_dir / "query.py").write_text(STANDALONE_QUERY, encoding="utf-8")
    (project_dir / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (project_dir / "CLAUDE.md").write_text(
        CLAUDE_MD_TEMPLATE.format(display=display_name(name)), encoding="utf-8"
    )
    print(f"Packaged: {project_dir}/  (CLAUDE.md, query.py, requirements.txt)")


def main():
    parser = argparse.ArgumentParser(description="Package a corpus into a portable query kit")
    parser.add_argument("--name", required=True, help="Project name")
    parser.add_argument("--output", default="output", help="Base output directory (default: output)")
    args = parser.parse_args()
    build_package(args.name, args.output)


if __name__ == "__main__":
    main()
