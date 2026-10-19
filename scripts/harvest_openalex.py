"""OpenAlex harvester:Hanford / waste-glass literature via the OpenAlex API.

Saves one JSON file per query into data/raw_openalex/.
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
RAW_DIR = os.path.join(WS, "data", "raw_openalex")
os.makedirs(RAW_DIR, exist_ok=True)


"""OpenAlex harvester: Hanford / waste-glass literature via the OpenAlex API.

Uses page-based paging (no API key needed). If ~/.openalex_api_key exists,
uses cursor paging + the key instead (more reliable, higher limits).
Saves one JSON file per query into data/raw_openalex/.
"""
import os
import sys
import time

MAILTO = "research-db@localhost"
KEY_FILE = os.path.expanduser("~/.openalex_api_key")
API_KEY = None
if os.path.exists(KEY_FILE):
    API_KEY = open(KEY_FILE).read().strip()
    print(f"OpenAlex API key loaded from {KEY_FILE}")

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(WS, "data", "raw_openalex")
os.makedirs(RAW_DIR, exist_ok=True)


def fetch(url, timeout=60, retries=5):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode('utf-8','replace')[:120]}"
            time.sleep(5 * (i + 1))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"fetch failed: {url} ({last})")


SELECT = ",".join([
    "id", "doi", "title", "display_name", "publication_year",
    "publication_date", "type", "authorships", "primary_location",
    "locations", "open_access", "cited_by_count", "related_works",
    "concepts", "topics", "abstract_inverted_index",
])


def harvest(search, label, per_page=100, max_records=1000, filter_extra=""):
    out_path = os.path.join(RAW_DIR, f"openalex_{label}.json")
    all_results = []
    common = {"search": search, "per-page": per_page, "mailto": MAILTO,
             "select": SELECT}
    if API_KEY:
        common["api_key"] = API_KEY
    if filter_extra:
        common["filter"] = filter_extra

    if API_KEY:
        # cursor paging (recommended path, needs key)
        cursor = "*"
        got = 0
        while cursor and got < max_records:
            p = dict(common)
            p["cursor"] = cursor
            data = json.loads(fetch("https://api.openalex.org/works?" +
                                    urllib.parse.urlencode(p)))
            results = data.get("results", [])
            all_results.extend(results)
            got += len(results)
            cursor = data.get("meta", {}).get("next_cursor")
            if not results:
                break
            time.sleep(1.0)
    else:
        # page-based paging (no key): works up to ~10k records
        page = 1
        got = 0
        while got < max_records:
            p = dict(common)
            p["page"] = page
            data = json.loads(fetch("https://api.openalex.org/works?" +
                                    urllib.parse.urlencode(p)))
            results = data.get("results", [])
            all_results.extend(results)
            got += len(results)
            if not results:
                break
            page += 1
            time.sleep(1.0)

    with open(out_path, "w") as f:
        json.dump({"label": label, "search": search, "count": len(all_results),
                   "results": all_results}, f, ensure_ascii=False)
    return len(all_results)


def main():
    queries = json.load(open(sys.argv[1]))
    for q in queries:
        label = q["label"]
        try:
            n = harvest(q["search"], label, filter_extra=q.get("filter", ""))
            print(f"[{label}] {n} records")
        except Exception as e:  # noqa: BLE001
            print(f"[{label}] FAILED: {e}")
        time.sleep(0.5)


if __name__ == "__main__":
    main()
