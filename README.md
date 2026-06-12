# Doc Scraper

Scrapes documentation websites into clean Markdown files and builds a local RAG (vector search) database from them. Point it at any docs site and get a fully searchable knowledge base in minutes.

## Features

- Crawls a documentation URL and converts every page to clean `.md` with YAML frontmatter
- Resumes interrupted scrapes — skips already-downloaded pages
- Combines all pages into a single `_combined.md` for easy reading or export
- Builds a [ChromaDB](https://www.trychroma.com/) vector database using `all-MiniLM-L6-v2` embeddings
- Semantic search via CLI — single query or interactive mode
- Agent-friendly `--agent` output mode (compact XML for LLM pipelines)
- Generates a corpus macro-map (`index.md`) summarizing the full knowledge base
- Interactive TUI wizard (`./scrapy`) for guided setup

## Requirements

- Python 3.8+
- `pip install -r requirements.txt`
- For JavaScript-heavy sites: `playwright install chromium`

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/doc-scraper.git
cd doc-scraper
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Full pipeline (recommended)

```bash
python run.py --url https://docs.example.com/guides --name example
```

This runs scrape → combine → embed in sequence. Output lands in `output/example/`.

### Interactive wizard

```bash
./scrapy
```

Guided TUI that walks through the full pipeline step by step.

### Individual steps

```bash
# 1. Scrape only
python scrape.py --crawl https://docs.example.com/guides --name example

# 2. Combine + build RAG database
python process.py --name example

# 3. Query the database
python query.py --name example "how does authentication work?"
python query.py --name example          # interactive mode
```

### Agent mode (for LLM pipelines)

```bash
python query.py --name example --agent "your query"
```

Returns compact XML chunks without decorative UI — designed for piping into Claude or other LLMs.

## Output Structure

```
output/<name>/
  docs/           # Individual .md files organized by URL path
  _combined.md    # All docs merged with metadata headers
  index.md        # Corpus macro-map (section summaries)
  chroma_db/      # ChromaDB vector database
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--url` | — | Starting URL to crawl |
| `--name` | — | Project name (used for output folder and DB collection) |
| `--output` | `output/` | Base output directory |
| `--chunk-size` | `500` | Words per RAG chunk |
| `--skip-rag` | off | Combine only, skip embedding step |

## How it works

1. **Scrape** — `scrape.py` crawls all pages under the given URL, strips navigation/boilerplate, and writes each page as a `.md` file. Uses `requests` + `BeautifulSoup` by default; falls back to Playwright for JS-rendered pages.
2. **Combine** — `process.py` merges every `.md` into `_combined.md` and generates `index.md` (a structured summary of the corpus).
3. **Embed** — `build_rag.py` chunks the combined doc, generates embeddings with `sentence-transformers`, and stores them in a persistent ChromaDB database.
4. **Query** — `query.py` embeds your search query the same way and returns the most semantically similar chunks.

## License

MIT
