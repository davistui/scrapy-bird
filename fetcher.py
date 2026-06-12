"""Fetch backend abstraction.

Default path: requests + BeautifulSoup (fast).
On the first successful requests fetch in `auto` mode, decide whether the page
looks like an SPA shell. If so, flip to a Playwright (headless Chromium) backend
for the rest of the run.
"""

import os
import subprocess
import sys
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup


_MODE = "auto"            # "auto" | "always" | "never"
_BACKEND = "requests"     # flips to "playwright" once detected
_PW = None                # (playwright, browser, context) — lazy
_USER_AGENT = "DocScraper/1.0 (documentation indexing tool)"


def set_mode(mode):
    global _MODE, _BACKEND
    if mode not in ("auto", "always", "never"):
        mode = "auto"
    _MODE = mode
    _BACKEND = "playwright" if mode == "always" else "requests"


def fetch_html(url, session):
    """Return (html_string, final_url) or (None, None)."""
    global _BACKEND

    if _BACKEND == "playwright":
        return _fetch_pw(url)

    html, final_url = _fetch_requests(url, session)
    if html is None:
        return None, None

    if _MODE == "auto" and _looks_like_spa(html, final_url or url):
        print("  [auto] SPA detected — switching to Playwright")
        _BACKEND = "playwright"
        return _fetch_pw(url)

    return html, final_url


def close():
    """Tear down the Playwright browser if we started it."""
    global _PW
    if _PW is None:
        return
    pw, browser, context = _PW
    try:
        context.close()
    except Exception:
        pass
    try:
        browser.close()
    except Exception:
        pass
    try:
        pw.stop()
    except Exception:
        pass
    _PW = None


def _fetch_requests(url, session):
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "")
        if "text/html" not in ctype:
            return None, None
        return resp.text, resp.url
    except requests.RequestException as e:
        print(f"  [SKIP] Failed to fetch {url}: {e}")
        return None, None


def _looks_like_spa(html, url):
    """Heuristic: empty shell = few internal links AND/OR very little visible text."""
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return False

    for tag in soup.find_all(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    domain = urlparse(url).netloc
    self_url = url.rstrip("/")
    internal = 0
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(url, href)
        full, _ = urldefrag(full)
        full = full.rstrip("/")
        if not full or full == self_url:
            continue
        if urlparse(full).netloc == domain:
            internal += 1

    visible_len = len(soup.get_text(strip=True))
    return internal < 5 or visible_len < 500


def _fetch_pw(url):
    try:
        _ensure_playwright()
    except Exception as e:
        print(f"  [SKIP] Playwright unavailable: {e}")
        return None, None

    _, _, context = _PW
    page = context.new_page()
    try:
        try:
            page.goto(url, wait_until="networkidle", timeout=30000)
        except Exception:
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass
        html = page.content()
        final_url = page.url
        return html, final_url
    except Exception as e:
        print(f"  [SKIP] Playwright fetch failed for {url}: {e}")
        return None, None
    finally:
        try:
            page.close()
        except Exception:
            pass


def _ensure_playwright():
    global _PW
    if _PW is not None:
        return

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise RuntimeError(
            "playwright package not installed; run: pip install playwright"
        ) from e

    pw = sync_playwright().start()
    try:
        browser = pw.chromium.launch(headless=True)
    except Exception as e:
        msg = str(e)
        if "Executable doesn't exist" in msg or "playwright install" in msg:
            _install_chromium_browser()
            browser = pw.chromium.launch(headless=True)
        else:
            pw.stop()
            raise

    context = browser.new_context(user_agent=_USER_AGENT)
    _PW = (pw, browser, context)


def _install_chromium_browser():
    print()
    print("  Playwright needs Chromium (~150 MB) for JS-rendered pages.")
    answer = input("  Install now? (Y/n) ").strip().lower()
    if answer and not answer.startswith("y"):
        raise RuntimeError("Chromium browser required for Playwright fallback")
    print("  Installing Chromium...")
    subprocess.check_call(
        [sys.executable, "-m", "playwright", "install", "chromium"]
    )
    print("  ✓ Chromium installed")
