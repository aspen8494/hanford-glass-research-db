"""Additional source harvesters: Crossref, arXiv, PNNL publications search.

Each function appends normalized records to data/raw_misc/<name>.json
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
MAILTO = "research-db@localhost"

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(WS, "data", "raw_misc")
os.makedirs(RAW, exist_ok=True)

KEY_FILE = os.path.expanduser("~/.openalex_api_key")
API_KEY = None
if os.path.exists(KEY_FILE):
    API_KEY = open(KEY_FILE).read().strip()


def fetch(url, timeout=60, retries=4, headers=None):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers or {"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"fetch failed: {url} ({last})")


# ------------------------------ Crossref ------------------------------

def crossref_search(query, label, rows=1000, filter_extra=None):
    """Search Crossref works; saves raw items."""
    all_items = []
    offset = 0
    filt = "type:journal-article,type:proceedings-article,type:report-component"
    if filter_extra:
        filt += "," + filter_extra
    while offset < rows:
        params = {
            "query": query,
            "rows": min(200, rows - offset),
            "offset": offset,
            "filter": filt,
            "select": "DOI,title,author,published,container-title,type,publisher,is-referenced-by-count,abstract,subject",
            "mailto": MAILTO,
        }
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
        d = json.loads(fetch(url))
        items = d.get("message", {}).get("items", [])
        all_items.extend(items)
        total = d.get("message", {}).get("total-results", 0)
        offset += len(items)
        if not items or offset >= total:
            break
        time.sleep(1.2)  # polite pool
    with open(os.path.join(RAW, f"crossref_{label}.json"), "w") as f:
        json.dump({"label": label, "query": query, "count": len(all_items),
                   "items": all_items}, f, ensure_ascii=False)
    print(f"[crossref:{label}] {len(all_items)} records (total-results={total})")
    return len(all_items)


# ------------------------------ arXiv ------------------------------

ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}

def arxiv_search(query, label, max_results=300):
    url = ("http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": f"all:{query}", "max_results": max_results}))
    raw = fetch(url)
    import xml.etree.ElementTree as ET
    root = ET.fromstring(raw)
    entries = []
    for e in root.findall("a:entry", ARXIV_NS):
        entries.append({
            "id": e.find("a:id", ARXIV_NS).text,
            "title": " ".join(e.find("a:title", ARXIV_NS).text.split()),
            "summary": " ".join(e.find("a:summary", ARXIV_NS).text.split()),
            "published": e.find("a:published", ARXIV_NS).text,
            "authors": [a.find("a:name", ARXIV_NS).text
                        for a in e.findall("a:author", ARXIV_NS)],
        })
    with open(os.path.join(RAW, f"arxiv_{label}.json"), "w") as f:
        json.dump({"label": label, "query": query, "count": len(entries),
                   "entries": entries}, f, ensure_ascii=False)
    print(f"[arxiv:{label}] {len(entries)} records")
    return len(entries)


# ------------------------------ PNNL publications ------------------------------

def pnnl_pubsearch(query, label, max_pages=10):
    """Scrape PNNL's publications search (signals PNNL-SA report numbers)."""
    results = []
    for page in range(1, max_pages + 1):
        url = ("https://www.pnnl.gov/publications/search?search_api_fulltext="
               + urllib.parse.quote(query) + f"&page={page}")
        try:
            raw = fetch(url, timeout=40)
        except Exception as e:  # noqa: BLE001
            print(f"[pnnl:{label}] page {page} failed: {e}")
            break
        # extract titles + links from the results list
        import re
        for m in re.finditer(
                r'<a href="(/publications/[^"]+)"[^>]*>([^<]+)</a>', raw):
            results.append({"path": m.group(1), "title": m.group(2).strip()})
        if "There are no results" in raw or "no results" in raw.lower():
            break
        time.sleep(0.8)
    # dedupe
    seen, uniq = set(), []
    for r in results:
        if r["path"] not in seen:
            seen.add(r["path"])
            uniq.append(r)
    with open(os.path.join(RAW, f"pnnl_{label}.json"), "w") as f:
        json.dump({"label": label, "query": query, "count": len(uniq),
                   "results": uniq}, f, ensure_ascii=False)
    print(f"[pnnl:{label}] {len(uniq)} records")
    return len(uniq)


def main():
    queries = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else []
    for q in queries:
        label = q["label"]
        src = q.get("source", "crossref")
        try:
            if src == "crossref":
                crossref_search(q["query"], label, rows=q.get("rows", 1000),
                                filter_extra=q.get("filter"))
            elif src == "arxiv":
                arxiv_search(q["query"], label, max_results=q.get("max", 300))
            elif src == "pnnl":
                pnnl_pubsearch(q["query"], label, max_pages=q.get("pages", 10))
        except Exception as e:  # noqa: BLE001
            print(f"[{src}:{label}] FAILED: {e}")
        time.sleep(0.5)


if __name__ == "__main__":
    main()
