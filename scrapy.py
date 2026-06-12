#!/usr/bin/env python3
"""Scrapy - Interactive green CLI wizard for doc scraping + RAG pipeline."""

import os
import re
import subprocess
import sys
import threading
import time

# Re-exec under ./venv if not already running there. The venv has a working
# chromadb; system Python 3.14 does not.
_VENV_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "python3")
if os.path.exists(_VENV_PY) and os.path.realpath(sys.executable) != os.path.realpath(_VENV_PY):
    os.execv(_VENV_PY, [_VENV_PY, os.path.abspath(__file__), *sys.argv[1:]])


# ── Dependency check ─────────────────────────────────────────────────────────
def _check_deps():
    """Check for missing packages and offer to install them."""
    required = {
        "requests": "requests",
        "bs4": "beautifulsoup4",
        "markdownify": "markdownify",
        "lxml": "lxml",
        "chromadb": "chromadb",
        "sentence_transformers": "sentence-transformers",
        "tqdm": "tqdm",
        "playwright": "playwright",
    }
    missing = []
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)

    if not missing:
        return

    print(f"\033[1;32m  Missing packages: \033[32m{', '.join(missing)}\033[0m")
    print(f"\033[32m  I need these to run. Want me to install them?\033[0m")
    answer = input(f"\033[1;32m  ? Install now? (Y/n) \033[0m").strip()
    if answer and not answer.lower().startswith("y"):
        print("\033[32m  Can't run without them — bye!\033[0m")
        sys.exit(1)

    print()
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--break-system-packages", "-q", *missing
    ])
    print(f"\n\033[1;32m  ✓ All set!\033[0m\n")


_check_deps()


# ── ANSI colors ──────────────────────────────────────────────────────────────
GREEN = "\033[32m"
BOLD_GREEN = "\033[1;32m"
DIM_GREEN = "\033[2;32m"
YELLOW = "\033[38;5;226m"
GOLD = "\033[38;5;220m"
ORANGE = "\033[38;5;208m"
EYE_WHITE = "\033[97m"
PUPIL = "\033[30m"
DIM = "\033[2m"
RESET = "\033[0m"


def green(text):
    return f"{GREEN}{text}{RESET}"


def bold_green(text):
    return f"{BOLD_GREEN}{text}{RESET}"


def dim_green(text):
    return f"{DIM_GREEN}{text}{RESET}"


# ── Green stdout wrapper ────────────────────────────────────────────────────
class GreenWriter:
    """Wraps stdout so sub-module print() output stays green.

    When a Spinner is active, intercepts [count/max] url lines from
    scrape.py and updates the spinner's progress display instead.
    """

    def __init__(self, stream, spinner=None):
        self._stream = stream
        self.spinner = spinner

    def write(self, text):
        if self.spinner and text and text.strip():
            # Try to parse progress lines like "  [23/200] https://..."
            m = re.match(r"\s*\[(\d+)/(\d+)\]\s+(.*)", text.strip())
            if m:
                current, total, url = m.group(1), m.group(2), m.group(3)
                # Shorten url to just the path
                path = re.sub(r"https?://[^/]+", "", url)
                self.spinner.update_progress(int(current), int(total), path)
                return
        if text and text.strip():
            self._stream.write(f"{DIM_GREEN}{text}{RESET}")
        else:
            self._stream.write(text)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


# ── Spinner ──────────────────────────────────────────────────────────────────
class Spinner:
    """Green animated spinner for long-running tasks."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message):
        self.message = message
        self._stop = threading.Event()
        self._thread = None
        self._progress = None  # (current, total, path)
        self._start_time = None
        self._lock = threading.Lock()

    def update_progress(self, current, total, path):
        with self._lock:
            self._progress = (current, total, path)

    def _format_elapsed(self, seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            frame = self.FRAMES[i % len(self.FRAMES)]
            with self._lock:
                progress = self._progress
            if progress:
                current, total, path = progress
                elapsed = self._format_elapsed(time.time() - self._start_time)
                # Truncate path if too long
                if len(path) > 40:
                    path = "..." + path[-37:]
                line = (
                    f"\r{GREEN}{frame} {self.message} "
                    f"{current}/{total} pages | Current: {path} | "
                    f"{elapsed} elapsed{RESET}\033[K"
                )
            else:
                line = f"\r{GREEN}{frame} {self.message}{RESET}\033[K"
            sys.stdout.write(line)
            sys.stdout.flush()
            i += 1
            time.sleep(0.08)

    def __enter__(self):
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        sys.stdout.write(f"\r{BOLD_GREEN}✓ {self.message}{RESET}\033[K\n")
        sys.stdout.flush()


# ── Box drawing helper ───────────────────────────────────────────────────────
def print_box(title, rows, width=44):
    """Print a green bordered box with title and key-value rows."""
    inner = width - 4  # space inside the borders
    print(bold_green(f"╭─ {title} " + "─" * (inner - len(title) - 1) + "╮"))
    for label, value in rows:
        content = f"  {label}  {value}"
        padding = inner - len(content)
        print(bold_green("│") + green(content) + " " * max(padding, 0) + bold_green(" │"))
    print(bold_green("╰" + "─" * (width - 2) + "╯"))


# ── Banner ───────────────────────────────────────────────────────────────────
def _bird_lines():
    """8-bit Flappy-Bird-style mascot. Each line is pre-colored."""
    Y = YELLOW
    G = GOLD
    O = ORANGE
    W = EYE_WHITE
    P = PUPIL
    R = RESET
    # Body: yellow/gold gradient. Eye: white circle, black pupil. Beak: orange.
    return [
        f"        {Y}▄▄▄▄▄▄▄▄{R}",
        f"      {Y}▄██████████▄{R}",
        f"    {Y}▄████{W}▄▄{Y}██{W}▄▄{Y}████▄{R}{O}▄▄▄▄{R}",
        f"   {Y}██████{W}██{G}██{W}██{Y}██████{R}{O}████████{R}",
        f"   {Y}██████{W}██{P}██{W}██{Y}██████{R}{O}▀▀▀▀▀▀▀▀{R}",
        f"   {Y}██████{W}▀▀{G}██{W}▀▀{Y}██████{R}",
        f"    {G}▀████{Y}██████████{G}████▀{R}",
        f"      {G}▀████{Y}████████{G}████▀{R}",
        f"         {G}▀▀████████▀▀{R}",
        f"             {DIM}{G}▀  ▀{R}",
    ]


def print_banner():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.abspath(os.path.join(script_dir, "output"))

    print()
    for line in _bird_lines():
        print(line)
    print()
    print(f"  {BOLD_GREEN}✻ Welcome to Scrapy Bird!{RESET}")
    print()
    print(f"    {DIM_GREEN}/help for help, paste a docs URL to begin{RESET}")
    print()
    print(f"    {DIM}cwd:    {script_dir}{RESET}")
    print(f"    {DIM}output: {output_path}/{RESET}")
    print()
    print(f"  {DIM_GREEN}Tip:{RESET} {dim_green('name auto-derives from the URL hostname')}")
    print(f"       {dim_green('(docs.pega.com → pega)')}")
    print()


# ── Prompt helpers ───────────────────────────────────────────────────────────
def prompt(question, default=None):
    """Green input prompt with optional default."""
    if default:
        display = f"{BOLD_GREEN}? {question} {DIM_GREEN}({default}){RESET} "
    else:
        display = f"{BOLD_GREEN}? {question}{RESET} "
    answer = input(display).strip()
    return answer if answer else default


def prompt_project_name():
    """Prompt for a valid project name (alphanumeric, hyphens, underscores)."""
    while True:
        name = prompt("What's the project name?")
        if not name:
            print(dim_green("  Project name cannot be empty. Use something like: my-docs"))
            continue
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", name):
            print(dim_green("  Use only letters, numbers, hyphens, underscores (no spaces)."))
            continue
        return name


def prompt_url():
    """Prompt for a valid URL starting with http:// or https://."""
    while True:
        url = prompt("URL to crawl?")
        if not url:
            return None
        if not url.startswith(("http://", "https://")):
            print(dim_green("  URL must start with http:// or https://"))
            continue
        return url


def prompt_positive_int(question, default):
    """Prompt for a positive integer."""
    while True:
        raw = prompt(question, default)
        try:
            val = int(raw)
            if val <= 0:
                raise ValueError
            return val
        except (ValueError, TypeError):
            print(dim_green("  Please enter a positive whole number."))


def prompt_positive_float(question, default):
    """Prompt for a positive number."""
    while True:
        raw = prompt(question, default)
        try:
            val = float(raw)
            if val <= 0:
                raise ValueError
            return val
        except (ValueError, TypeError):
            print(dim_green("  Please enter a positive number."))


def prompt_choice(question, choices):
    """Display a menu and return the selected key."""
    print(f"{BOLD_GREEN}? {question}{RESET}")
    for key, label in choices:
        print(f"  {BOLD_GREEN}{key}{RESET} {green(label)}")
    while True:
        answer = input(f"  {BOLD_GREEN}>{RESET} ").strip().upper()
        valid = [k.upper() for k, _ in choices]
        if answer in valid:
            return answer
        print(dim_green(f"  Choose one of: {', '.join(valid)}"))


def prompt_confirm(question, default_yes=True):
    """Y/n confirmation prompt. Returns True/False."""
    hint = "Y/n" if default_yes else "y/N"
    answer = prompt(f"{question} ({hint})", "")
    if not answer:
        return default_yes
    return answer.lower().startswith("y")


# ── Count helpers ────────────────────────────────────────────────────────────
def count_md_files(docs_dir):
    """Count .md files in a docs directory."""
    count = 0
    if os.path.isdir(docs_dir):
        for root, _dirs, files in os.walk(docs_dir):
            for f in files:
                if f.endswith(".md") and not f.startswith("_"):
                    count += 1
    return count


def project_status(name, output_base):
    """Return (doc_count, rag_ready) for a project."""
    project_dir = os.path.join(output_base, name)
    docs_dir = os.path.join(project_dir, "docs")
    db_dir = os.path.join(project_dir, "chroma_db")
    doc_count = count_md_files(docs_dir)
    rag_ready = os.path.isdir(db_dir)
    return doc_count, rag_ready


def print_project_status(name, output_base):
    """Print a status line for the current project."""
    doc_count, rag_ready = project_status(name, output_base)
    rag_str = "ready" if rag_ready else "not built"
    print(dim_green(f"  Project: {name} ({doc_count} docs, RAG: {rag_str})"))
    print()


def format_duration(seconds):
    """Format seconds into a human-readable string."""
    m, s = divmod(int(seconds), 60)
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# ── Actions ──────────────────────────────────────────────────────────────────
MAX_PAGES = 10000
DELAY = 1.0
SECTION_TOP_N = 25


def prompt_section_selection(groups, start_url):
    """Show top sections by page count and let the user pick which to scrape.

    Returns a list of URLs, or None to mean "fall back to crawl-by-links".
    """
    sorted_groups = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    total_pages = sum(len(v) for v in groups.values())
    total_sections = len(groups)

    print()
    print(green(f"  Found sitemap: {total_pages:,} pages across {total_sections:,} sections"))
    print()
    print(bold_green(f"  Top {min(SECTION_TOP_N, total_sections)} sections by page count:"))
    print()
    top = sorted_groups[:SECTION_TOP_N]
    for i, (key, urls) in enumerate(top, 1):
        marker = dim_green("(versioned)") if _is_versioned(key) else ""
        print(f"  {BOLD_GREEN}{i:>3}.{RESET} {green(f'{len(urls):>6,}')}  {green(key)} {marker}")

    if total_sections > SECTION_TOP_N:
        rest = sum(len(g[1]) for g in sorted_groups[SECTION_TOP_N:])
        print()
        print(dim_green(f"  + {total_sections - SECTION_TOP_N} more sections ({rest:,} pages)"))

    print()
    print(green("  Pick what to scrape:"))
    print(dim_green("    numbers (e.g. 1,3,6-9 or just 1)"))
    print(dim_green("    'current' — only top sections without version suffixes"))
    print(dim_green("    'all'     — every URL in the sitemap"))
    print(dim_green("    'skip'    — fall back to crawl-by-links instead"))
    print()
    raw = input(f"  {BOLD_GREEN}? Selection:{RESET} ").strip().lower()

    if not raw or raw == "skip":
        return None
    if raw == "all":
        return [u for _, urls in sorted_groups for u in urls]
    if raw == "current":
        keys = [k for k, _ in top if not _is_versioned(k)]
        return [u for k in keys for u in groups[k]]

    # Parse number list / ranges
    picked_idxs = set()
    for token in raw.replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            try:
                a, b = token.split("-", 1)
                a, b = int(a), int(b)
                for i in range(min(a, b), max(a, b) + 1):
                    picked_idxs.add(i)
            except ValueError:
                pass
        else:
            try:
                picked_idxs.add(int(token))
            except ValueError:
                pass

    picked_keys = [top[i - 1][0] for i in sorted(picked_idxs) if 1 <= i <= len(top)]
    if not picked_keys:
        print(dim_green("  No valid selections — falling back to crawl-by-links."))
        return None

    selected = []
    for k in picked_keys:
        selected.extend(groups[k])

    print()
    print(green(f"  Selected {len(picked_keys)} sections, {len(selected):,} pages"))
    print(dim_green(f"  Sections: {', '.join(picked_keys)}"))
    return selected


def _is_versioned(key):
    from sitemap import is_versioned
    return is_versioned(key)


def do_scrape(name, output_base, url=None):
    """Run the scraper. If url is provided, skips the URL prompt."""
    from scrape import crawl, scrape_urls
    from sitemap import fetch_sitemap_urls, group_urls, filter_to_prefix

    if not url:
        url = prompt_url()
        if not url:
            print(green("  No URL provided, skipping."))
            return 0

    docs_dir = os.path.join(output_base, name, "docs")

    import requests
    session = requests.Session()
    session.headers.update({"User-Agent": "DocScraper/1.0 (documentation indexing tool)"})

    # Try sitemap-first discovery
    print()
    print(dim_green("  Looking for sitemap..."))
    sitemap_urls = fetch_sitemap_urls(url)
    selected_urls = None
    if sitemap_urls:
        scoped = filter_to_prefix(sitemap_urls, url)
        if scoped:
            groups = group_urls(scoped, depth=2)
            selected_urls = prompt_section_selection(groups, url)
        else:
            print(dim_green("  Sitemap found but no URLs match this start path."))
    else:
        print(dim_green("  No sitemap found — using link-following crawl."))

    print()
    if selected_urls:
        print(green(f"  Scraping {len(selected_urls):,} pages from sitemap"))
    else:
        print(green(f"  Scraping {url}"))
        print(dim_green(f"  ({MAX_PAGES} max pages, {DELAY}s between requests)"))
    print(dim_green("  Press Ctrl-C to pause; you'll be asked to confirm before exiting."))
    print()
    start_time = time.time()

    # Confirm-on-Ctrl-C loop: if interrupted, prompt to exit. If declined,
    # crawl restarts and resumes via the markdown files already on disk.
    while True:
        spinner = Spinner("Scraping...")
        old_stdout = sys.stdout
        sys.stdout = GreenWriter(old_stdout, spinner=spinner)
        try:
            with spinner:
                if selected_urls:
                    scrape_urls(selected_urls, docs_dir, DELAY, session)
                else:
                    crawl(url, docs_dir, MAX_PAGES, [], DELAY, session)
            break
        except KeyboardInterrupt:
            sys.stdout = old_stdout
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            print()
            answer = input(
                f"  {BOLD_GREEN}? Really exit? Scrape will be paused. (y/N){RESET} "
            ).strip().lower()
            if answer == "y":
                print(green("  Paused. Run scrapy again with the same URL to resume."))
                return count_md_files(docs_dir)
            print(green("  Resuming..."))
            print()
            continue
        finally:
            sys.stdout = old_stdout

    elapsed = time.time() - start_time
    scraped = count_md_files(docs_dir)

    print()
    print_box("Summary", [
        ("Pages scraped:", str(scraped)),
        ("Output:", os.path.join("..", name, "docs/")),
        ("Time:", format_duration(elapsed)),
    ])
    return scraped


def do_process(name, output_base):
    """Combine docs and build RAG database."""
    from process import combine
    from build_rag import build_rag

    project_dir = os.path.join(output_base, name)
    docs_dir = os.path.join(project_dir, "docs")

    if not os.path.isdir(docs_dir):
        print(green(f"  No docs found at {docs_dir}. Run scrape first."))
        return

    start_time = time.time()

    old_stdout = sys.stdout
    sys.stdout = GreenWriter(old_stdout)
    try:
        with Spinner("Combining docs..."):
            combine(name, output_base)
    finally:
        sys.stdout = old_stdout

    print()
    old_stdout = sys.stdout
    sys.stdout = GreenWriter(old_stdout)
    try:
        build_rag(name, output_base)
    finally:
        sys.stdout = old_stdout

    elapsed = time.time() - start_time
    db_path = os.path.join(project_dir, "chroma_db")

    print()
    print_box("Summary", [
        ("RAG database:", db_path),
        ("Time:", format_duration(elapsed)),
    ])


def do_query(name, output_base):
    """Launch interactive query mode."""
    from query import DocQuery

    print(dim_green("  Loading query engine...\n"))

    old_stdout = sys.stdout
    sys.stdout = GreenWriter(old_stdout)
    try:
        dq = DocQuery(name, output_base)
    finally:
        sys.stdout = old_stdout

    # Custom green query loop
    print(green("  Type a question, or /quit to exit.\n"))
    while True:
        try:
            q = input(f"  {BOLD_GREEN}query>{RESET} ").strip()
            if not q:
                continue
            if q == "/quit":
                break
            results = dq.search(q)
            output = dq.format_results(q, results)
            # Print results in green
            for line in output.split("\n"):
                print(green(f"  {line}"))
            print()
        except (KeyboardInterrupt, EOFError):
            break

    print(green("\n  Done."))


def derive_name_from_url(url):
    """docs.pega.com → pega ; www.servicenow.com → servicenow."""
    host = url.split("://", 1)[-1].split("/", 1)[0]
    for prefix in ("docs.", "www."):
        if host.startswith(prefix):
            host = host[len(prefix):]
    name = host.split(".")[0]
    name = re.sub(r"[^a-zA-Z0-9_-]", "-", name)
    return name or "site"


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Resolve paths relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    output_base = "output"

    print_banner()

    url = prompt_url()
    if not url:
        print(green("  No URL provided. Bye!"))
        return

    name = derive_name_from_url(url)
    print(dim_green(f"  → project name: {name}"))

    print()
    scraped = do_scrape(name, output_base, url=url)
    if scraped > 0:
        print()
        print(green("  Building the RAG database..."))
        print()
        do_process(name, output_base)

    print()
    print(bold_green("  Done. 🦆"))
    print()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print(f"\n\n{BOLD_GREEN}  Interrupted — goodbye! 🦆{RESET}\n")
        sys.exit(0)
