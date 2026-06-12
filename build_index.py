#!/usr/bin/env python3
"""Generate a compressed macro-map of a scraped corpus into index.md.

Lets an agent scan the architecture of a large corpus at a glance — sections,
files, and their headings — instead of probing the vector DB blindly.

Usage:
    python build_index.py --name couchbase
    python build_index.py --name couchbase --output output
"""

import argparse
import os
import re
from pathlib import Path

# Titles the scraper sometimes captures that carry no real signal.
_JUNK_TITLES = {"Untitled", "For AI agents:", ""}

# Cap headers per file so the map stays a macro-map, not a full dump.
_MAX_HEADERS = 12


def parse_title(filepath):
    """Read the frontmatter title, falling back to a humanized filename."""
    title = None
    with open(filepath, "r", encoding="utf-8") as f:
        in_front = False
        for line in f:
            s = line.strip()
            if s == "---":
                if in_front:
                    break
                in_front = True
                continue
            if in_front and s.startswith("title:"):
                t = s[len("title:"):].strip()
                if t not in _JUNK_TITLES:
                    title = t
                break
    if title:
        return title
    return Path(filepath).stem.replace("_", " ").replace("-", " ")


def extract_headers(filepath):
    """Return up to _MAX_HEADERS (depth, text) H2/H3 section headers from a file."""
    headers = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^(#{2,3})\s+(.*)", line)
            if m and m.group(2).strip():
                headers.append((len(m.group(1)), m.group(2).strip()))
                if len(headers) >= _MAX_HEADERS:
                    break
    return headers


def find_md_files(docs_dir):
    """Recursively find all .md files, excluding _-prefixed ones."""
    md = []
    for root, _dirs, files in os.walk(docs_dir):
        for fname in sorted(files):
            if fname.endswith(".md") and not fname.startswith("_"):
                md.append(os.path.join(root, fname))
    return sorted(md)


def build_index(name, output_base):
    """Write output/<name>/index.md — a section/file/header map of the corpus."""
    project_dir = os.path.join(output_base, name)
    docs_dir = os.path.join(project_dir, "docs")

    if not os.path.isdir(docs_dir):
        print(f"Error: docs directory not found: {docs_dir}")
        return

    md_files = find_md_files(docs_dir)
    if not md_files:
        print(f"No .md files found in {docs_dir}")
        return

    # Group files by category (first subfolder within docs/).
    by_cat = {}
    for fp in md_files:
        rel = os.path.relpath(fp, docs_dir)
        parts = Path(rel).parts
        cat = parts[0] if len(parts) > 1 else "general"
        by_cat.setdefault(cat, []).append((rel, fp))

    out_path = os.path.join(project_dir, "index.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# {name} — Corpus Map\n\n")
        f.write(f"{len(md_files)} documents across {len(by_cat)} sections. "
                f"Scan to locate a topic, then query the RAG DB for the detail.\n\n")
        for cat in sorted(by_cat, key=lambda c: -len(by_cat[c])):
            files = sorted(by_cat[cat])
            f.write(f"## {cat} ({len(files)} files)\n\n")
            for rel, fp in files:
                f.write(f"- `{rel}` — {parse_title(fp)}\n")
                for depth, htext in extract_headers(fp):
                    f.write(f"{'    ' * (depth - 1)}- {htext}\n")
            f.write("\n")

    print(f"Created: {out_path}  ({len(md_files)} files, {len(by_cat)} sections)")


def main():
    parser = argparse.ArgumentParser(description="Generate a macro-map (index.md) of a scraped corpus")
    parser.add_argument("--name", required=True, help="Project name")
    parser.add_argument("--output", default="output", help="Base output directory (default: output)")
    args = parser.parse_args()
    build_index(args.name, args.output)


if __name__ == "__main__":
    main()
