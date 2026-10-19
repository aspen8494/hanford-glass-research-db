"""OSTI (DOE Office of Scientific and Technical Information) harvester.

Queries the public OSTI API v1 and saves every hit as a normalized JSON record.
Handles the XML fallback (OSTI returns XML when rows param is absent) and
rate-limits politely.

Usage: python harvest_osti.py <batch_name> <queries_json_file>
  queries_json_file: JSON list of {"label":..., "params": "field=value&..."} dicts.
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(WS, "data", "raw_osti")
os.makedirs(RAW_DIR, exist_ok=True)


def fetch(url, timeout=60, retries=3):
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace"), {
                    k.lower(): v for k, v in r.headers.items()
                }
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"fetch failed after {retries}: {url} ({last_err})")


def parse_osti(raw):
    """Parse JSON list or XML <records><record>... into list of dicts."""
    s = raw.lstrip()
    if s.startswith("<"):
        # OSTI sometimes returns XML despite Accept: application/json.
        # Strip invalid XML control chars before parsing.
        clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)
        root = ET.fromstring(clean)
        recs = []
        for node in root.iter("record"):
            rec = {}
            for child in node:
                rec[child.tag] = (child.text or "").strip()
            recs.append(rec)
        return recs
    return json.loads(raw)


def harvest_one(label, params, page_size=500, max_rows=5000):
    """Page through one OSTI query; return list of records."""
    all_recs = []
    page = 1
    total = None
    while True:
        q = dict(urllib.parse.parse_qsl(params))
        q["rows"] = str(page_size)
        q["page"] = str(page)
        q["format"] = "json"  # default XML contains invalid control chars
        url = "https://www.osti.gov/api/v1/records?" + urllib.parse.urlencode(q)
        raw, hdrs = fetch(url)
        recs = parse_osti(raw)
        if total is None:
            total = int(hdrs.get("x-total-count", 0) or 0)
        all_recs.extend(recs)
        if len(recs) < page_size or len(all_recs) >= max_rows:
            break
        page += 1
        time.sleep(0.5)
    return all_recs, total


def main():
    batch_name = sys.argv[1]
    queries = json.load(open(sys.argv[2]))
    out_path = os.path.join(RAW_DIR, f"osti_{batch_name}.json")
    merged = {}
    stats = []
    for spec in queries:
        label, params = spec["label"], spec["params"]
        try:
            recs, total = harvest_one(label, params,
                                      max_rows=spec.get("max_rows", 5000))
        except Exception as e:  # noqa: BLE001
            print(f"[{label}] FAILED: {e}")
            stats.append({"label": label, "error": str(e)[:200]})
            continue
        for r in recs:
            oid = r.get("osti_id")
            if oid and oid not in merged:
                merged[oid] = {"_source_query": label, **r}
        stats.append({"label": label, "total": total, "fetched": len(recs),
                      "new_unique": sum(1 for r in recs
                                        if r.get("osti_id") in merged
                                        and merged[r.get("osti_id")]["_source_query"] == label)})
        print(f"[{label}] total={total} fetched={len(recs)} merged={len(merged)}")
        time.sleep(0.5)

        # incremental checkpoint: never lose the batch on a late crash
        with open(out_path, "w") as f:
            json.dump({"batch": batch_name, "stats": stats,
                       "records": sorted(merged.values(),
                                         key=lambda r: str(r.get("osti_id", "")))},
                      f, ensure_ascii=False, indent=1)

    with open(out_path, "w") as f:
        json.dump({"batch": batch_name, "stats": stats,
                   "records": sorted(merged.values(),
                                     key=lambda r: str(r.get("osti_id", "")))},
                  f, ensure_ascii=False, indent=1)
    print(f"\nSaved {len(merged)} unique records -> {out_path}")


if __name__ == "__main__":
    main()
