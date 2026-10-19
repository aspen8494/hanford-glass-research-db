"""Verify a sample of database records resolve to real, accessible documents.

Checks: DOI links via doi.org HEAD, OSTI biblio pages, direct PDF URLs.
Writes data/verification_report.json and prints a summary.
"""
import json
import os
import random
import urllib.request
import urllib.error

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
      "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.8"}


def status(url, timeout=25):
    """Return HTTP status or error string for a URL (HEAD, fallback GET)."""
    if not url:
        return "no-url", None
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(url, headers=UA, method=method)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.headers.get("Content-Type", "")
        except urllib.error.HTTPError as e:
            if method == "GET":
                return e.code, ""
            if e.code in (403, 405, 429):
                continue  # HEAD may be disallowed; try GET
            return e.code, ""
        except Exception as e:  # noqa: BLE001
            if method == "GET":
                return f"error:{type(e).__name__}", ""
            continue
    return "error", ""


def main(sample_n=60, seed=42):
    records = [json.loads(l) for l in open(os.path.join(WS, "data", "records.jsonl"))]
    # only records that have at least one resolvable identifier
    candidates = [r for r in records if r.get("doi") or r.get("osti_id") or r.get("pdf_url")]
    random.Random(seed).shuffle(candidates)
    sample = candidates[:sample_n]

    results = {"ok": 0, "redirect": 0, "fail": 0, "no_url": 0,
               "bot_blocked": 0, "transient": 0, "details": []}
    for r in sample:
        entry = {"id": r["id"], "title": r["title"][:80], "source": r["_sources"]}
        url, kind = "", ""
        if r.get("doi"):
            url, kind = f"https://doi.org/{r['doi']}", "doi"
        elif r.get("osti_id") and r["osti_id"].isdigit():
            url, kind = f"https://www.osti.gov/biblio/{r['osti_id']}", "osti"
        elif r.get("pdf_url"):
            url, kind = r["pdf_url"], "pdf"
        st, ct = status(url)
        entry.update({"checked_url": url, "kind": kind, "status": st, "content_type": ct})
        if isinstance(st, int) and st < 300:
            results["ok"] += 1
        elif isinstance(st, int) and st < 400:
            results["redirect"] += 1
        elif st == 403:
            # publisher bot-block: the record exists but refuses scripted access
            results["bot_blocked"] += 1
        elif st == 404:
            results["fail"] += 1
        else:
            results["transient"] += 1
        results["details"].append(entry)

    total = len(sample)
    print(f"checked {total}: ok={results['ok']} redirect={results['redirect']} fail={results['fail']}")
    fails = [d for d in results["details"]
             if not (isinstance(d["status"], int) and d["status"] < 400)]
    print(f"\n{len(fails)} failures:")
    for d in fails[:15]:
        print(f"  [{d['status']}] {d['kind']}: {d['title']}")

    out = os.path.join(WS, "data", "verification_report.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nreport -> {out}")
    return results


if __name__ == "__main__":
    import sys
    main(sample_n=int(sys.argv[1]) if len(sys.argv) > 1 else 60)
