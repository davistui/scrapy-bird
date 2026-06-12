#!/usr/bin/env python3
"""Doc Scraper - Crawl documentation sections into clean .md files."""

import argparse
import os
import re
import time
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

import fetcher


def url_to_path(url, crawl_domain):
    """Convert a URL to an organized folder/file path relative to docs/.

    Uses top 2 path segments as folders, remaining segments as filename.
    Example: /condeco/getting-started/setup/page → condeco/getting-started/setup_page.md
    """
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if not path:
        return "index.md"

    segments = [s for s in path.split("/") if s]

    if len(segments) <= 2:
        # 0-2 segments: all become folders, file is index
        folder = os.path.join(*segments) if segments else ""
        filename = "index.md"
    else:
        # First 2 segments become folders, rest become filename
        folder = os.path.join(segments[0], segments[1])
        remaining = segments[2:]
        slug = "_".join(remaining)
        slug = re.sub(r"[^\w\-]", "-", slug)
        slug = re.sub(r"-+", "-", slug).strip("-")
        filename = slug[:200] + ".md"

    return os.path.join(folder, filename) if folder else filename


def extract_main_content(soup):
    """Extract the main content area, stripping nav/header/footer."""
    for tag in soup.find_all(["nav", "header", "footer", "script", "style", "noscript"]):
        tag.decompose()
    for tag in soup.find_all("a", class_="headerlink"):
        tag.decompose()
    for tag in soup.find_all("a", class_="anchor"):
        tag.decompose()

    for selector in ["main", "article", '[role="main"]', ".content", ".docs-content",
                     "#content", "#main-content", ".markdown-body", ".documentation"]:
        main = soup.select_one(selector)
        if main:
            return main

    body = soup.find("body")
    return body if body else soup


def fetch_page(url, session):
    """Fetch a page and return (soup, final_url) or (None, None) on failure."""
    html, final_url = fetcher.fetch_html(url, session)
    if html is None:
        return None, None
    return BeautifulSoup(html, "lxml"), final_url


def page_to_markdown(soup):
    """Convert a BeautifulSoup element to clean markdown."""
    content = extract_main_content(soup)
    markdown = md(str(content), heading_style="ATX", code_language="", strip=["img"])
    markdown = re.sub(r"\s*[\u00b6\u00c2]\s*", " ", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip()


def get_title(soup):
    """Extract page title."""
    h1 = soup.find("h1")
    if h1:
        for a in h1.find_all("a", class_="headerlink"):
            a.decompose()
        return h1.get_text(strip=True)
    title = soup.find("title")
    if title:
        return title.get_text(strip=True)
    return "Untitled"


def find_section_links(soup, base_url, prefix):
    """Find all links on the page that stay within the URL prefix."""
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = urljoin(base_url, href)
        full_url, _ = urldefrag(full_url)
        full_url = full_url.rstrip("/")
        if full_url.startswith(prefix):
            links.add(full_url)
    return links


def save_page(docs_dir, url, title, markdown, crawl_domain):
    """Save a page as a .md file with metadata header, organized by URL path."""
    rel_path = url_to_path(url, crawl_domain)
    filepath = os.path.join(docs_dir, rel_path)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"---\ntitle: {title}\nsource: {url}\n---\n\n")
        f.write(f"# {title}\n\n")
        f.write(markdown)
        f.write("\n")
    return rel_path


def load_visited(docs_dir):
    """Load already-scraped URLs from existing .md files (resume support)."""
    visited = set()
    if not os.path.isdir(docs_dir):
        return visited
    for root, _dirs, files in os.walk(docs_dir):
        for fname in files:
            if fname.endswith(".md") and not fname.startswith("_"):
                filepath = os.path.join(root, fname)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.startswith("source: "):
                                url = line[len("source: "):].strip()
                                visited.add(url)
                                break
                except OSError:
                    pass
    return visited


def crawl(start_url, docs_dir, max_pages, exclude_patterns, delay, session):
    """Crawl a documentation section starting from start_url."""
    fetcher.set_mode(getattr(crawl, "_js_mode", "auto"))
    try:
        prefix = start_url.rstrip("/")
        crawl_domain = urlparse(start_url).netloc
        os.makedirs(docs_dir, exist_ok=True)

        visited = load_visited(docs_dir)
        if visited:
            print(f"Resuming: found {len(visited)} already-scraped pages")
            soup, _ = fetch_page(prefix, session)
            if soup:
                seed_links = find_section_links(soup, prefix + "/", prefix)
                queue = sorted(link for link in seed_links if link not in visited)
            else:
                queue = []
            if delay > 0 and queue:
                time.sleep(delay)
        else:
            queue = [prefix]

        discovered = set(queue) | visited
        count = len(visited)

        while queue and count < max_pages:
            url = queue.pop(0)
            if url in visited:
                continue

            if any(re.search(pat, url) for pat in exclude_patterns):
                continue

            count += 1
            print(f"  [{count}/{max_pages}] {url}")

            soup, final_url = fetch_page(url, session)
            if soup is None:
                continue

            raw_final_url = final_url
            final_url = final_url.rstrip("/")
            visited.add(url)
            visited.add(final_url)

            title = get_title(soup)
            markdown = page_to_markdown(soup)

            if markdown:
                save_page(docs_dir, final_url, title, markdown, crawl_domain)

            new_links = find_section_links(soup, raw_final_url, prefix)
            for link in sorted(new_links):
                if link not in discovered and link not in visited:
                    discovered.add(link)
                    queue.append(link)

            if delay > 0:
                time.sleep(delay)

        if count >= max_pages:
            reason = f"hit max ({max_pages})"
        elif not queue:
            reason = "queue exhausted (no more reachable pages)"
        else:
            reason = "loop exited unexpectedly"
        print(f"\nDone! Scraped {count} pages to {docs_dir}  [exit: {reason}]")
    finally:
        fetcher.close()


def scrape_urls(urls, docs_dir, delay, session):
    """Scrape a list of explicit URLs."""
    fetcher.set_mode(getattr(scrape_urls, "_js_mode", "auto"))
    try:
        os.makedirs(docs_dir, exist_ok=True)
        visited = load_visited(docs_dir)
        total = len(urls)

        for i, url in enumerate(urls, 1):
            url = url.strip().rstrip("/")
            if not url or url in visited:
                continue

            print(f"  [{i}/{total}] {url}")
            soup, final_url = fetch_page(url, session)
            if soup is None:
                continue

            title = get_title(soup)
            markdown = page_to_markdown(soup)
            crawl_domain = urlparse(url).netloc

            if markdown:
                save_page(docs_dir, final_url or url, title, markdown, crawl_domain)

            if delay > 0 and i < total:
                time.sleep(delay)

        print(f"\nDone! Scraped to {docs_dir}")
    finally:
        fetcher.close()


def main():
    parser = argparse.ArgumentParser(description="Scrape documentation into clean .md files")
    parser.add_argument("--name", required=True, help="Name for the output folder")
    parser.add_argument("--crawl", metavar="URL", help="Start URL - crawls all pages under this path")
    parser.add_argument("--file", metavar="PATH", help="Read URLs from a text file (one per line)")
    parser.add_argument("--max", type=int, default=200, help="Max pages to scrape (default: 200)")
    parser.add_argument("--exclude", action="append", default=[], help="Skip URLs matching pattern (repeatable)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds (default: 1)")
    parser.add_argument("--output", default="output", help="Base output directory (default: output)")
    parser.add_argument("--js", choices=["auto", "always", "never"], default="auto",
                        help="JS-rendering fallback: auto-detect SPAs (default), always, or never")
    parser.add_argument("urls", nargs="*", help="Direct URLs to scrape")
    args = parser.parse_args()

    crawl._js_mode = args.js
    scrape_urls._js_mode = args.js

    # Output structure: <output>/<name>/docs/
    docs_dir = os.path.join(args.output, args.name, "docs")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "DocScraper/1.0 (documentation indexing tool)"
    })

    if args.crawl:
        print(f"Crawling: {args.crawl}")
        print(f"Output:   {docs_dir}")
        print(f"Max:      {args.max} pages\n")
        crawl(args.crawl, docs_dir, args.max, args.exclude, args.delay, session)

    elif args.file:
        with open(args.file, "r") as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        print(f"Scraping {len(urls)} URLs from {args.file}")
        print(f"Output: {docs_dir}\n")
        scrape_urls(urls, docs_dir, args.delay, session)

    elif args.urls:
        print(f"Scraping {len(args.urls)} URLs")
        print(f"Output: {docs_dir}\n")
        scrape_urls(args.urls, docs_dir, args.delay, session)

    else:
        parser.error("Provide --crawl <url>, --file <path>, or direct URLs")


if __name__ == "__main__":
    main()
