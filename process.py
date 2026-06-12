#!/usr/bin/env python3
"""Combine scraped .md files into a single RAG-ready document, then build vector DB."""

import argparse
import os
import subprocess
import sys
from datetime import date


def parse_frontmatter(filepath):
    """Extract title and source from the YAML frontmatter."""
    title = "Untitled"
    source = ""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines or lines[0].strip() != "---":
        return title, source, "".join(lines)

    end_idx = None
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        return title, source, "".join(lines)

    for line in lines[1:end_idx]:
        if line.startswith("title: "):
            title = line[len("title: "):].strip()
        elif line.startswith("source: "):
            source = line[len("source: "):].strip()

    content = "".join(lines[end_idx + 1:]).strip()
    return title, source, content


def find_md_files(docs_dir):
    """Recursively find all .md files in docs/, excluding _combined.md."""
    md_files = []
    for root, _dirs, files in os.walk(docs_dir):
        for fname in sorted(files):
            if fname.endswith(".md") and not fname.startswith("_"):
                md_files.append(os.path.join(root, fname))
    return sorted(md_files)


def combine(name, output_base):
    """Combine all .md files in a project's docs/ folder into _combined.md."""
    project_dir = os.path.join(output_base, name)
    docs_dir = os.path.join(project_dir, "docs")

    if not os.path.isdir(docs_dir):
        print(f"Error: docs directory not found: {docs_dir}")
        return False

    md_files = find_md_files(docs_dir)

    if not md_files:
        print(f"No .md files found in {docs_dir}")
        return False

    print(f"Combining {len(md_files)} documents from {docs_dir}")

    sections = []
    for filepath in md_files:
        title, source, content = parse_frontmatter(filepath)
        # Get relative path within docs/ for category info
        rel_path = os.path.relpath(filepath, docs_dir)
        if content:
            sections.append((title, source, content, rel_path))

    # Write _combined.md to project root
    output_path = os.path.join(project_dir, "_combined.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# {name} Knowledge Base\n")
        f.write(f"Generated: {date.today().isoformat()}\n")
        f.write(f"Sources: {len(sections)} documents\n\n")

        for title, source, content, rel_path in sections:
            f.write("---\n\n")
            f.write(f"## Document: {title}\n")
            if source:
                f.write(f"Source: {source}\n")
            f.write(f"Path: {rel_path}\n")
            f.write("\n")
            f.write(content)
            f.write("\n\n")

    print(f"Created: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Combine scraped .md files and build RAG database")
    parser.add_argument("--name", required=True, help="Project name (matches output folder name)")
    parser.add_argument("--output", default="output", help="Base output directory (default: output)")
    parser.add_argument("--skip-rag", action="store_true", help="Skip building the RAG database")
    args = parser.parse_args()

    success = combine(args.name, args.output)

    if success:
        script_dir = os.path.dirname(os.path.abspath(__file__))

        print("\nBuilding corpus index...")
        build_index = os.path.join(script_dir, "build_index.py")
        subprocess.run(
            [sys.executable, build_index, "--name", args.name, "--output", args.output],
            check=True
        )

        print("\nPackaging portable query kit...")
        build_package = os.path.join(script_dir, "build_package.py")
        subprocess.run(
            [sys.executable, build_package, "--name", args.name, "--output", args.output],
            check=True
        )

        if not args.skip_rag:
            print("\nBuilding RAG database...")
            build_rag = os.path.join(script_dir, "build_rag.py")
            subprocess.run(
                [sys.executable, build_rag, "--name", args.name, "--output", args.output],
                check=True
            )


if __name__ == "__main__":
    main()
