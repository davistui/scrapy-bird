#!/usr/bin/env python3
"""Scrape PDFs linked from a webpage and convert them to markdown.

For each PDF link found on the page(s), downloads the PDF and converts it
to a .md file with YAML frontmatter (title, source) so the rest of the
pipeline (process.py / build_rag.py) can consume them unchanged.
"""

import argparse
import os
import re
import tempfile
import time
from urllib.parse import urljoin, urldefrag, urlparse

import requests
from bs4 import BeautifulSoup, Tag
import pymupdf4llm

import fetcher


def slugify(text, maxlen=120):
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text[:maxlen] or "untitled"


def _is_section_header(tag):
    """Heuristic: a block-level element whose text is short and all-uppercase.

    Catches section labels that aren't wrapped in <h2>/<h3> (common on
    CMS-rendered sites). The element must contain no PDF anchors itself.
    """
    if not isinstance(tag, Tag) or tag.name not in ("div", "p", "h2", "h3", "h4", "h5", "section"):
        return False
    if tag.find("a", href=lambda h: h and h.lower().endswith(".pdf") if h else False):
        return False
    txt = tag.get_text(" ", strip=True)
    if not txt or len(txt) > 80:
        return False
    if txt.upper() != txt:
        return False
    # Require at least 2 letters; ignore stray punctuation-only text
    letters = re.sub(r"[^A-Za-z]", "", txt)
    return len(letters) >= 2


def find_pdf_links(soup, base_url):
    """Walk DOM in document order, returning [(section, title, pdf_url)].

    `section` is the most recently seen all-caps heading-like element above
    the link, or None if none has been encountered. De-duped by URL (first
    occurrence wins so the assigned section is the canonical one).
    """
    main = soup.find("main") or soup.body or soup
    results = []
    seen = set()
    current = {"section": None}

    def walk(node):
        if isinstance(node, Tag):
            if _is_section_header(node):
                current["section"] = node.get_text(" ", strip=True)
                return
            if node.name == "a" and node.get("href"):
                href = node["href"].strip()
                url_no_frag, _ = urldefrag(urljoin(base_url, href))
                path = urlparse(url_no_frag).path.lower()
                if path.endswith(".pdf") and url_no_frag not in seen:
                    seen.add(url_no_frag)
                    title = node.get_text(" ", strip=True) or node.get("title") or os.path.basename(path)
                    results.append((current["section"], title, url_no_frag))
                return
            for c in node.children:
                walk(c)

    walk(main)
    return results


def download_pdf(url, session, dest_path):
    """Stream a PDF to disk. Returns True on success."""
    try:
        with session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        print(f"    ! download failed: {e}")
        return False


def pdf_to_markdown(pdf_path):
    """Convert a PDF file to markdown text via pymupdf4llm."""
    try:
        md = pymupdf4llm.to_markdown(pdf_path, show_progress=False)
        md = re.sub(r"\n{3,}", "\n\n", md)
        return md.strip()
    except Exception as e:
        print(f"    ! convert failed: {e}")
        return ""


def save_markdown(docs_dir, section, title, source_url, markdown, used_slugs):
    """Save markdown with frontmatter under docs/<section>/<slug>.md.

    `section` is the human-readable header text (or None → falls back to "pdfs").
    """
    sec_slug = slugify(section) if section else "pdfs"
    base_slug = slugify(title)
    key = (sec_slug, base_slug)
    slug = base_slug
    n = 2
    while (sec_slug, slug) in used_slugs:
        slug = f"{base_slug}-{n}"
        n += 1
    used_slugs.add((sec_slug, slug))

    folder = os.path.join(docs_dir, sec_slug)
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"{slug}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"---\ntitle: {title}\nsource: {source_url}\nsection: {section or ''}\n---\n\n")
        f.write(f"# {title}\n\n")
        f.write(markdown)
        f.write("\n")
    return filepath


def existing_sources(docs_dir):
    """Return set of source URLs already saved (for resume)."""
    found = set()
    if not os.path.isdir(docs_dir):
        return found
    for root, _dirs, files in os.walk(docs_dir):
        for fn in files:
            if not fn.endswith(".md") or fn.startswith("_"):
                continue
            try:
                with open(os.path.join(root, fn), "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("source: "):
                            found.add(line[len("source: "):].strip())
                            break
            except OSError:
                pass
    return found


def existing_slugs(docs_dir):
    """Return set of (section_slug, file_slug) already on disk."""
    found = set()
    if not os.path.isdir(docs_dir):
        return found
    for entry in os.listdir(docs_dir):
        sub = os.path.join(docs_dir, entry)
        if not os.path.isdir(sub):
            continue
        for fn in os.listdir(sub):
            if fn.endswith(".md") and not fn.startswith("_"):
                found.add((entry, os.path.splitext(fn)[0]))
    return found


def gather_from_page(page_url, session):
    """Fetch a page and return its (title-or-href, pdf_url) list."""
    html, final_url = fetcher.fetch_html(page_url, session)
    if html is None:
        print(f"  ! could not fetch {page_url}")
        return []
    soup = BeautifulSoup(html, "lxml")
    return find_pdf_links(soup, final_url or page_url)


def main():
    parser = argparse.ArgumentParser(description="Scrape PDFs from a page into markdown")
    parser.add_argument("--name", required=True, help="Output folder name")
    parser.add_argument("--url", action="append", default=[],
                        help="Page URL containing PDF links (repeatable)")
    parser.add_argument("--output", default="output", help="Base output directory")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between PDF downloads (seconds)")
    parser.add_argument("--max", type=int, default=0,
                        help="Cap number of PDFs (0 = no cap)")
    parser.add_argument("--js", choices=["auto", "always", "never"], default="auto")
    args = parser.parse_args()

    if not args.url:
        parser.error("Provide at least one --url")

    fetcher.set_mode(args.js)
    docs_dir = os.path.join(args.output, args.name, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; DocScraper/1.0)"
    })

    try:
        # Gather PDF links (with section) across all source pages
        all_links = []
        seen = set()
        for page_url in args.url:
            print(f"Scanning: {page_url}")
            for section, title, pdf_url in gather_from_page(page_url, session):
                if pdf_url in seen:
                    continue
                seen.add(pdf_url)
                all_links.append((section, title, pdf_url))

        sections_count = len({s for s, _, _ in all_links if s})
        print(f"Found {len(all_links)} unique PDF links across {sections_count} sections")

        already = existing_sources(docs_dir)
        used = existing_slugs(docs_dir)
        todo = [(s, t, u) for (s, t, u) in all_links if u not in already]
        if len(todo) < len(all_links):
            print(f"Resuming: skipping {len(all_links) - len(todo)} already-downloaded")
        if args.max and len(todo) > args.max:
            todo = todo[:args.max]
            print(f"Capped at --max {args.max}")

        ok = fail = 0
        with tempfile.TemporaryDirectory() as tmp:
            for i, (section, title, pdf_url) in enumerate(todo, 1):
                tag = f"[{section}] " if section else ""
                print(f"  [{i}/{len(todo)}] {tag}{title[:60]}")
                tmp_path = os.path.join(tmp, f"f{i}.pdf")
                if not download_pdf(pdf_url, session, tmp_path):
                    fail += 1
                    continue
                md = pdf_to_markdown(tmp_path)
                if not md:
                    fail += 1
                    continue
                save_markdown(docs_dir, section, title, pdf_url, md, used)
                ok += 1
                if args.delay > 0 and i < len(todo):
                    time.sleep(args.delay)

        print(f"\nDone. Converted {ok} PDFs, {fail} failed. Output: {docs_dir}")
    finally:
        fetcher.close()


if __name__ == "__main__":
    main()
