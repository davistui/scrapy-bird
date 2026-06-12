"""Sitemap discovery and URL grouping.

Tries `<host>/sitemap.xml`, expands sitemap-index files one level, and groups
URLs by the first N path segments so the wizard can show a section breakdown.
"""

import re
from collections import defaultdict
from urllib.parse import urlparse

import requests


_UA = {"User-Agent": "DocScraper/1.0 (sitemap discovery)"}


def fetch_sitemap_urls(start_url, timeout=30):
    """Return a list of URLs from the site's sitemap, or None if unavailable."""
    parsed = urlparse(start_url)
    if not parsed.scheme or not parsed.netloc:
        return None
    base = f"{parsed.scheme}://{parsed.netloc}"

    for path in ("/sitemap.xml", "/sitemap_index.xml"):
        try:
            r = requests.get(base + path, timeout=timeout, headers=_UA)
        except requests.RequestException:
            continue
        if r.status_code == 200 and "<" in r.text:
            return _expand(r.text, timeout)
    return None


def _expand(xml_text, timeout):
    locs = re.findall(r"<loc>\s*(.*?)\s*</loc>", xml_text)
    if "<sitemapindex" in xml_text:
        urls = []
        for sub in locs:
            try:
                r = requests.get(sub, timeout=timeout, headers=_UA)
            except requests.RequestException:
                continue
            if r.status_code == 200:
                urls.extend(re.findall(r"<loc>\s*(.*?)\s*</loc>", r.text))
        return urls
    return locs


def group_urls(urls, depth=2):
    """Group URLs by the first `depth` path segments. Returns {section_key: [urls...]}."""
    groups = defaultdict(list)
    for u in urls:
        path = [s for s in urlparse(u).path.strip("/").split("/") if s]
        if not path:
            key = "(root)"
        else:
            key = "/".join(path[:depth])
        groups[key].append(u)
    return dict(groups)


def filter_to_prefix(urls, start_url):
    """If start_url is deeper than the site root, restrict urls to its sub-tree.

    Path-boundary aware: prefix `/bundle/foo` matches `/bundle/foo` and
    `/bundle/foo/...` but NOT `/bundle/foo-242/...`.
    """
    parsed = urlparse(start_url.rstrip("/"))
    if not parsed.path or parsed.path == "/":
        return urls
    prefix = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
    out = []
    for u in urls:
        u_clean = u.rstrip("/")
        if u_clean == prefix or u_clean.startswith(prefix + "/"):
            out.append(u)
    return out


_VERSION_SUFFIX = re.compile(r"-(?:\d+|\d+[a-z]?|r\d+|v\d+)$", re.IGNORECASE)


def is_versioned(section_key):
    """True if a section key ends with a version-like suffix (e.g. -88, -242, -23, -r25)."""
    last = section_key.rsplit("/", 1)[-1]
    return bool(_VERSION_SUFFIX.search(last))
