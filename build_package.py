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

HERE = Path(__file__).resolve().parent

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


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
            print("Loading models...")
        self.embedder = SentenceTransformer(EMBED_MODEL)

        # Reranker is optional: if its model can't load (offline / not yet
        # downloaded) or is disabled via QUERY_NO_RERANK=1, fall back to RRF
        # fusion of vector+BM25 so search still works.
        self.reranker = None
        if os.environ.get("QUERY_NO_RERANK") != "1":
            try:
                self.reranker = CrossEncoder(RERANK_MODEL)
            except Exception as e:
                if not quiet:
                    print(f"(reranker unavailable - using vector+BM25 fusion: {e})")

        data = self.collection.get(include=["documents", "metadatas"])
        self._ids = data["ids"]
        self._docs = data["documents"]
        self._metas = data["metadatas"]
        self._bm25 = BM25Okapi([_tokenize(d) for d in self._docs]) if BM25Okapi and self._docs else None

        if not quiet:
            parts = ["vector"] + (["BM25"] if self._bm25 else [])
            parts.append("reranker" if self.reranker else "RRF-fusion")
            print(f"Ready! {len(self._ids):,} chunks indexed ({' + '.join(parts)}).\n")

    def search(self, query, n_results=5, category=None, n_candidates=25):
        """Hybrid retrieval: vector + BM25 recall, RRF-fused, then reranked."""
        pool = min(n_candidates, max(len(self._ids), 1))
        candidates = {}  # id -> {text, metadata, rrf}
        K = 60

        def add(cid, text, meta, rank):
            c = candidates.get(cid)
            if c is None:
                c = candidates[cid] = {"text": text, "metadata": meta, "rrf": 0.0}
            c["rrf"] += 1.0 / (K + rank)

        embedding = self.embedder.encode([query]).tolist()
        where = {"category": category} if category else None
        vres = self.collection.query(query_embeddings=embedding, n_results=pool, where=where)
        for rank in range(len(vres["ids"][0])):
            add(vres["ids"][0][rank], vres["documents"][0][rank], vres["metadatas"][0][rank], rank)

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
    p.add_argument("--candidates", type=int, default=25,
                   help="Recall pool size per backend before reranking (default: 25)")
    p.add_argument("--category", help="Filter by category")
    p.add_argument("--agent", action="store_true",
                   help="Lean machine-readable XML output (full chunks, no decoration)")
    p.add_argument("query", nargs="*", help="Search query (omit for interactive mode)")
    args = p.parse_args()

    dq = DocQuery(quiet=args.agent)
    if args.query:
        q = " ".join(args.query)
        results = dq.search(q, n_results=args.n, category=args.category, n_candidates=args.candidates)
        print(dq.format_for_agent(q, results) if args.agent else dq.format_human(q, results))
    else:
        dq.interactive()


if __name__ == "__main__":
    main()
'''


REQUIREMENTS = "chromadb\nsentence-transformers\nrank-bm25\n"


CLAUDE_MD_TEMPLATE = '''# CLAUDE.md — {display} Knowledge Base

This folder is a **self-contained, portable RAG knowledge base** of the {display}
documentation. Your job here is to **answer questions grounded in these docs** —
not to guess from memory.

## What's here
- `index.md`      — macro-map: every section, file, and its headings. **Scan this first** to locate a topic.
- `docs/`         — the source markdown files (full text), organized by section.
- `chroma_db/`    — vector database for semantic search.
- `query.py`      — standalone hybrid search (vector + BM25 keyword, then cross-encoder rerank). Self-locating; no flags needed.
- `_combined.md`  — every doc in one file. **Do NOT load this into context** (very large).
- `insights/`     — where `/capture` saves session takeaways (GTM briefs, deep dives, notes).

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

## Synthesis commands (for sales / GTM / positioning work)
- `/overview`            — a who/what/when/where/how/why brief of this product, grounded in the docs.
- `/deep-dive <topic>`   — a deep, structured breakdown of one section or topic.
- `/capture [slug]`      — save the session's takeaways to `insights/` (asks: full or summary).

## Setup (first run in a new environment)
Python 3 with the listed packages:
```bash
pip install -r requirements.txt
```
The models (`all-MiniLM-L6-v2` embedder and the `ms-marco-MiniLM-L-6-v2` reranker) download automatically on first query.
'''


# Claude Code slash commands shipped inside each portable folder, under
# .claude/commands/. Available when you open Claude Code in the folder. Each is
# a prompt template that drives grounded retrieval via the local query.py.
COMMANDS = {
    "overview": '''---
description: High-level who/what/when/where/how/why brief of this product, grounded in the docs
allowed-tools: Bash(python query.py:*), Read
---

Build a high-level brief of **{display}** for sales / GTM / positioning synthesis.
Ground everything in this folder's docs — never guess from memory.

1. Read `index.md` to see the section map.
2. Retrieve each angle below with the lean agent output (adapt the wording to this product):
   - WHAT:  `python query.py --agent "what is {display} and what core problem does it solve"`
   - WHO:   `python query.py --agent "who is {display} for - target users, roles, personas, industries"`
   - WHY:   `python query.py --agent "why {display} - value proposition, benefits, differentiation"`
   - HOW:   `python query.py --agent "how {display} works - architecture, key capabilities, usage"`
   - WHERE: `python query.py --agent "where {display} fits - integrations, ecosystem, deployment, platforms"`
   - WHEN:  `python query.py --agent "when to use {display} - use cases, scenarios, prerequisites, limitations"`
3. Synthesize a tight brief with sections: **What it is**, **Who it's for**, **Why it matters**,
   **How it works**, **Where it fits**, **When to use it**. A few sentences each. Cite the `source` URL after claims.
4. End with **Sections to go deeper** — list the major areas from `index.md` and remind me I can run
   `/deep-dive <topic>` on any of them, and `/capture` to save the result.

Only state what the retrieved chunks support. If something isn't covered, say "not covered in the docs" — don't invent it.
''',
    "deep-dive": '''---
description: Deep, structured breakdown of one topic or section, grounded in the docs
argument-hint: <topic or section>
allowed-tools: Bash(python query.py:*), Read
---

Do a deep dive on: **$ARGUMENTS** — within the {display} docs. Ground everything in retrieval; don't guess.

1. Check `index.md` for sections related to "$ARGUMENTS".
2. Retrieve broadly with several angled queries, e.g.:
   - `python query.py --agent "$ARGUMENTS overview"`
   - `python query.py --agent "$ARGUMENTS how it works, configuration, details"`
   - `python query.py --agent "$ARGUMENTS limitations, requirements, gotchas"`
   Run more queries for sub-parts. For full detail, `Read` the specific files under `docs/` named by a chunk's `path`.
3. Produce a structured breakdown: **Summary**, **Key concepts**, **How it works / details**,
   **Practical notes (limits, requirements, gotchas)**, and **Angles worth using** (positioning hooks and
   buyer pains it addresses) drawn ONLY from what the docs actually say.
4. Cite `source` URLs throughout.

If retrieval is thin or conflicting, say so.
''',
    "capture": '''---
description: Save the current session's insights to the insights/ folder
argument-hint: [optional filename slug]
allowed-tools: Bash(mkdir:*), Write
---

Capture this session to a file in `insights/` so it isn't lost in the CLI.

1. Ask me: **full** (the verbatim Q&A and findings from this session) or **summary** (a tight synthesis of
   the key takeaways)? Wait for my answer before writing.
2. Choose `insights/<slug>.md` — use my $ARGUMENTS as the slug if given, else derive a short kebab-case slug
   from the topic. Run `mkdir -p insights` first.
3. Write the file with frontmatter (`title`, `date` = today's date, `source_kb: {display}`, `mode: full|summary`),
   then the content. Keep all source-URL citations intact.
4. Tell me the path you wrote.
''',
}


def display_name(name):
    return name.replace("-", " ").replace("_", " ").title()


def build_package(name, output_base):
    project_dir = Path(output_base) / name
    if not project_dir.is_dir():
        print(f"Error: project folder not found: {project_dir}")
        return

    display = display_name(name)
    (project_dir / "query.py").write_text(STANDALONE_QUERY, encoding="utf-8")
    (project_dir / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (project_dir / "CLAUDE.md").write_text(
        CLAUDE_MD_TEMPLATE.format(display=display), encoding="utf-8"
    )

    # Synthesis slash commands + a home for captured insights.
    commands_dir = project_dir / ".claude" / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)
    for cmd_name, template in COMMANDS.items():
        (commands_dir / f"{cmd_name}.md").write_text(template.format(display=display), encoding="utf-8")
    (project_dir / "insights").mkdir(exist_ok=True)

    cmds = ", ".join(f"/{c}" for c in COMMANDS)
    print(f"Packaged: {project_dir}/  (CLAUDE.md, query.py, requirements.txt, commands: {cmds})")


def main():
    parser = argparse.ArgumentParser(description="Package a corpus into a portable query kit")
    parser.add_argument("--name", required=True, help="Project name")
    parser.add_argument("--output", default="output", help="Base output directory (default: output)")
    args = parser.parse_args()
    build_package(args.name, args.output)


if __name__ == "__main__":
    main()
