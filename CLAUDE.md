# Doc Scraper

A tool that scrapes company documentation websites into clean markdown files and builds a RAG (vector search) database from them.

## Architecture

Single-command pipeline: **scrape → combine → embed**

- `run.py` — Main entry point. Orchestrates the full pipeline via subprocess calls to the scripts below.
- `scrape.py` — Crawls a documentation URL, converts HTML pages to `.md` files with YAML frontmatter (title, source). Supports resume (skips already-scraped URLs).
- `process.py` — Combines all individual `.md` files into a single `_combined.md`, then calls `build_rag.py`.
- `build_rag.py` — Chunks the markdown, generates embeddings with `all-MiniLM-L6-v2`, and stores them in a ChromaDB persistent database.
- `query.py` — Semantic search against the ChromaDB database. Supports single queries and interactive mode.
- `scrapy.py` / `scrapy` — Interactive CLI wizard that wraps the pipeline with a guided TUI. The `scrapy` shell script runs `scrapy.py` using the local venv.

## Output Structure

All output goes to `output/<name>/` by default (configurable via `--output`):

```
output/<name>/
  docs/           # Individual .md files organized by URL path
  _combined.md    # All docs merged with metadata headers
  chroma_db/      # ChromaDB vector database
```

The output folder is self-contained and portable.

## Key Conventions

- All exported files are markdown (`.md`) only — no other document formats.
- Each `.md` file has YAML frontmatter with `title` and `source` (original URL).
- Files starting with `_` (like `_combined.md`) are excluded from RAG indexing.
- URL-to-filepath mapping: first 2 path segments become folders, remaining segments become the filename.
- Embedding model: `all-MiniLM-L6-v2` (sentence-transformers). Chunk size default: 500 words.
- ChromaDB collection naming: `<name>_docs`.

## Dependencies

Python 3, with: requests, beautifulsoup4, markdownify, lxml, chromadb, sentence-transformers, tqdm. Managed via `requirements.txt` and a local `venv/`.

## Common Commands

```bash
# Full pipeline
python run.py --url https://docs.example.com/guides --name example

# Individual steps
python scrape.py --crawl https://docs.example.com/guides --name example
python process.py --name example
python query.py --name example "search query"

# Interactive wizard
./scrapy
```
