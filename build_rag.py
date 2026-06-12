#!/usr/bin/env python3
"""Build a ChromaDB vector database from scraped documentation.

Usage:
    python build_rag.py --name eptura
    python build_rag.py --name eptura --output ..
"""

import argparse
import os
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def parse_frontmatter(filepath):
    """Extract title and source from YAML frontmatter."""
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


def clean_text(text):
    """Clean markdown formatting for embedding."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    return text.strip()


def chunk_text(text, max_words=500):
    """Split text into chunks of approximately max_words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk_words = words[i:i + max_words]
        chunk_text = ' '.join(chunk_words)
        if len(chunk_text.strip()) >= 50:
            chunks.append(chunk_text)
    return chunks


def find_md_files(docs_dir):
    """Recursively find all .md files, excluding _combined.md."""
    md_files = []
    for root, _dirs, files in os.walk(docs_dir):
        for fname in sorted(files):
            if fname.endswith(".md") and not fname.startswith("_"):
                md_files.append(os.path.join(root, fname))
    return sorted(md_files)


def get_category(filepath, docs_dir):
    """Extract category from the subfolder path within docs/."""
    rel = os.path.relpath(filepath, docs_dir)
    parts = Path(rel).parts
    # Use first subfolder as category, or 'general' if at root
    return parts[0] if len(parts) > 1 else "general"


# Titles the scraper sometimes captures that carry no real signal.
_JUNK_TITLES = {"Untitled", "For AI agents:", ""}


def build_context_line(title, category, rel_path):
    """Build a 1-sentence situational context line to prepend to each chunk.

    Contextual Retrieval: gives every chunk self-contained awareness of where it
    came from, so a retrieved chunk needs no follow-up lookup. Derived purely from
    metadata (no LLM call). Falls back to a humanized path when the title is junk.
    """
    location = Path(rel_path).with_suffix("").as_posix().replace("_", " ").replace("-", " ")
    if title and title not in _JUNK_TITLES:
        return f'This chunk is from "{title}" in the {category} documentation (path: {location}).'
    return f"This chunk is from the {category} documentation: {location}."


def build_rag(name, output_base):
    """Build ChromaDB vector database from scraped docs."""
    project_dir = os.path.join(output_base, name)
    docs_dir = os.path.join(project_dir, "docs")
    db_path = os.path.join(project_dir, "chroma_db")

    if not os.path.isdir(docs_dir):
        print(f"Error: docs directory not found: {docs_dir}")
        return

    md_files = find_md_files(docs_dir)
    if not md_files:
        print(f"No .md files found in {docs_dir}")
        return

    print(f"Processing {len(md_files)} files from {docs_dir}")

    # Process files into chunks
    all_chunks = []
    for filepath in md_files:
        title, source, content = parse_frontmatter(filepath)
        if not content or len(content) < 50:
            continue

        cleaned = clean_text(content)
        category = get_category(filepath, docs_dir)
        rel_path = os.path.relpath(filepath, docs_dir)

        chunks = chunk_text(cleaned)
        context_line = build_context_line(title, category, rel_path)
        for i, chunk in enumerate(chunks):
            contextual_chunk = f"{context_line}\n{chunk}"
            all_chunks.append({
                "text": contextual_chunk,
                "metadata": {
                    "title": title,
                    "source": source,
                    "category": category,
                    "path": rel_path,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
            })

    if not all_chunks:
        print("No chunks generated!")
        return

    print(f"Created {len(all_chunks)} chunks")

    # Load embedding model
    print("Loading embedding model: all-MiniLM-L6-v2")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate embeddings
    texts = [c["text"] for c in all_chunks]
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = []
    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]
        batch_emb = embedder.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_emb.tolist())

    # Build ChromaDB
    print(f"Building ChromaDB at {db_path}")
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(allow_reset=True, anonymized_telemetry=False)
    )

    collection_name = f"{name}_docs"
    # Reset if exists
    existing = [col.name for col in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": f"{name} documentation knowledge base"}
    )

    # Index in batches
    print("Indexing chunks...")
    index_batch = 500
    for i in tqdm(range(0, len(all_chunks), index_batch), desc="Indexing"):
        end = min(i + index_batch, len(all_chunks))
        collection.add(
            embeddings=embeddings[i:end],
            documents=texts[i:end],
            metadatas=[c["metadata"] for c in all_chunks[i:end]],
            ids=[f"chunk_{j}" for j in range(i, end)]
        )

    # Stats
    print(f"\n{'='*40}")
    print(f"RAG DATABASE BUILT")
    print(f"{'='*40}")
    print(f"Project:    {name}")
    print(f"Documents:  {len(md_files)}")
    print(f"Chunks:     {collection.count()}")
    print(f"Database:   {db_path}")

    # Category breakdown
    categories = {}
    for c in all_chunks:
        cat = c["metadata"]["category"]
        categories[cat] = categories.get(cat, 0) + 1
    print("\nBy category:")
    for cat, cnt in sorted(categories.items()):
        print(f"  {cat}: {cnt} chunks")


def main():
    parser = argparse.ArgumentParser(description="Build RAG vector database from scraped docs")
    parser.add_argument("--name", required=True, help="Project name")
    parser.add_argument("--output", default="output", help="Base output directory (default: output)")
    parser.add_argument("--chunk-size", type=int, default=500, help="Words per chunk (default: 500)")
    args = parser.parse_args()

    build_rag(args.name, args.output)


if __name__ == "__main__":
    main()
