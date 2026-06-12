#!/usr/bin/env python3
"""Doc Scraper - One command to scrape, combine, and build RAG from any documentation URL.

Usage:
    python run.py --url https://docs.example.com/guides --name example
    python run.py --url https://docs.example.com/guides --name example --max 500
    python run.py --url https://docs.example.com/guides --name example --output ~/my-docs
    python run.py --url https://docs.example.com/guides --name example --skip-rag
"""

import argparse
import os
import subprocess
import sys

# Re-exec under ./venv if not already running there. The venv has a working
# chromadb; system Python 3.14 does not.
_VENV_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "python3")
if os.path.exists(_VENV_PY) and os.path.realpath(sys.executable) != os.path.realpath(_VENV_PY):
    os.execv(_VENV_PY, [_VENV_PY, os.path.abspath(__file__), *sys.argv[1:]])


def main():
    parser = argparse.ArgumentParser(
        description="Scrape documentation, combine into markdown, and build RAG database - all in one command"
    )
    parser.add_argument("--url", required=True, help="Documentation URL to crawl")
    parser.add_argument("--name", required=True, help="Project name (used as output folder name)")
    parser.add_argument("--output", default="output", help="Base output directory (default: output)")
    parser.add_argument("--max", type=int, default=200, help="Max pages to scrape (default: 200)")
    parser.add_argument("--exclude", action="append", default=[], help="Skip URLs matching pattern (repeatable)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds (default: 1)")
    parser.add_argument("--skip-rag", action="store_true", help="Skip building the RAG database")
    parser.add_argument("--js", choices=["auto", "always", "never"], default="auto",
                        help="JS-rendering fallback: auto-detect SPAs (default), always, or never")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(args.output, args.name)

    print(f"{'='*50}")
    print(f"Doc Scraper")
    print(f"{'='*50}")
    print(f"URL:    {args.url}")
    print(f"Name:   {args.name}")
    print(f"Output: {os.path.abspath(output_dir)}")
    print(f"Max:    {args.max} pages")
    print(f"{'='*50}\n")

    # Step 1: Scrape
    print("Step 1/3: Scraping documentation...\n")
    scrape_cmd = [
        sys.executable, os.path.join(script_dir, "scrape.py"),
        "--crawl", args.url,
        "--name", args.name,
        "--output", args.output,
        "--max", str(args.max),
        "--delay", str(args.delay),
        "--js", args.js,
    ]
    for pattern in args.exclude:
        scrape_cmd.extend(["--exclude", pattern])

    result = subprocess.run(scrape_cmd)
    if result.returncode != 0:
        print("\nScraping failed!")
        sys.exit(1)

    # Step 2: Combine + Build RAG
    rag_flag = "--skip-rag" if args.skip_rag else ""
    print("\nStep 2/3: Combining markdown files...")
    print("Step 3/3: Building RAG database...\n")

    process_cmd = [
        sys.executable, os.path.join(script_dir, "process.py"),
        "--name", args.name,
        "--output", args.output,
    ]
    if args.skip_rag:
        process_cmd.append("--skip-rag")

    result = subprocess.run(process_cmd)
    if result.returncode != 0:
        print("\nProcessing failed!")
        sys.exit(1)

    # Summary
    print(f"\n{'='*50}")
    print(f"COMPLETE!")
    print(f"{'='*50}")
    print(f"Output folder: {os.path.abspath(output_dir)}")
    print(f"\nContents:")
    print(f"  docs/         - Individual markdown files")
    print(f"  _combined.md  - All docs merged into one file")
    if not args.skip_rag:
        print(f"  chroma_db/    - Vector database for semantic search")
    print(f"\nThis folder is portable - move it anywhere you need.")
    if not args.skip_rag:
        print(f"\nTo query: python {os.path.join(script_dir, 'query.py')} --name {args.name} --output {args.output}")


if __name__ == "__main__":
    main()
