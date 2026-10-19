"""Normalize + merge + classify all raw harvest files into a unified database.

Reads:  data/raw_osti/*.json, data/raw_openalex/*.json, data/raw_misc/*.json,
        data/raw_web/*.json
Writes: data/records.jsonl        (canonical unified records)
        data/records_summary.json (counts by source/type/year)

Dedup priority: DOI > OSTI ID > OpenAlex ID > fuzzy title+author+year.
"""
import csv
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(WS, "data")

try:
    from rapidfuzz import fuzz
    def tsim(a, b):
        return fuzz.token_sort_ratio(a, b) / 100.0
except ImportError:
    def tsim(a, b):
        return SequenceMatcher(None, a, b).ratio()

# ----------------------------------------------------------------------------
# field normalization helpers
# ----------------------------------------------------------------------------

def norm_title(t):
    t = unicodedata.normalize("NFKD", t or "")
    t = t.lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def norm_doi(d):
    if not d:
        return ""
    d = d.strip().lower()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    d = re.sub(r"^doi:\s*", "", d)
    return d if d.startswith("10.") else ""


def parse_year(s):
    if not s:
        return None
    m = re.search(r"(19|20)\d{2}", str(s))
    return int(m.group(0)) if m else None


def clean_text(s, limit=4000):
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", str(s))          # strip jats/html tags
    s = re.sub(r"^\s*Abstract\s*", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit]


def split_authors_osti(s):
    if not s:
        return []
    if isinstance(s, list):
        return s
    parts = re.split(r";|\s\|\s", s)
    return [p.strip() for p in parts if p.strip() and p.strip().lower() != "et al."]


# ----------------------------------------------------------------------------
# topic classification (rule-based, transparent)
# ----------------------------------------------------------------------------

TAG_RULES = [
    ("hanford", r"hanford"),
    ("LAW", r"low[- ]activity|LAW\b|ILAW|immobilized low"),
    ("HLW", r"high[- ]level waste|HLW\b|sludge glass"),
    ("EWG", r"enhanced waste glass|EWG\b|enhanced glass formulation"),
    ("property-model", r"property[- ]composition|property model|composition[- ]property|"
                       r"viscosity model|liquidus|Tc-\d|electrical conductivity model|"
                       r"property/composition"),
    ("viscosity", r"viscosit"),
    ("liquidus", r"liquidus|TL\b|crystalliz"),
    ("durability", r"durab|corrosion|leach|PCT\b|product consistency|vapor hydration|VHT\b|"
                   r"chemical durability|alteration"),
    ("nepheline", r"nepheline"),
    ("sulfate", r"sulfate|sulphate|SO3"),
    ("halides", r"halide|chloride|iodide|fluoride"),
    ("technetium", r"technetium|Tc-99|pertechnetate"),
    ("redox", r"redox|Fe2\+|Fe3\+|oxygen fugacity"),
    ("melter", r"melter|melting rate|joule[- ]heated|cold cap|coldcap"),
    ("crystals-spinel", r"spinel|crystal|aerosol|particulate"),
    ("offgas", r"off[- ]gas|offgas|emission|volatil"),
    ("formulation", r"formulat|waste loading|GFC\b|glass former|frit"),
    ("dataset", r"database|data development|data package|interim glass property"),
    ("mission-flowsheet", r"flowsheet|feed vector|mission|campaign|HTWOS|WTP\b"),
    ("phosphate-glass", r"phosphate glass|iron phosphate"),
    ("alumina", r"alumin"),
    ("machine-learning", r"machine learning|neural network|deep learning|active learning|"
                         r"surrogate model|Bayesian"),
    ("review", r"review|overview|state of the art|perspective"),
]

FACILITY_RULES = [
    ("LAW", r"low[- ]activity|LAW\b|ILAW"),
    ("HLW", r"high[- ]level|HLW\b|sludge"),
]


def classify(rec):
    text = " ".join([rec.get("title", ""), rec.get("abstract", ""),
                     rec.get("venue", ""), " ".join(rec.get("report_numbers", []))]).lower()
    tags = [tag for tag, pat in TAG_RULES if re.search(pat, text, re.I)]
    fac = [f for f, pat in FACILITY_RULES if re.search(pat, text, re.I)]
    facility = "both" if set(fac) == {"LAW", "HLW"} else (fac[0] if fac else "")
    # relevance
    core = ("hanford" in tags and
            any(t in tags for t in ("LAW", "HLW", "EWG", "formulation", "property-model",
                                    "durability", "melter", "dataset", "nepheline")))
    glassy = any(t in tags for t in ("property-model", "durability", "formulation",
                                     "melter", "viscosity", "liquidus", "nepheline",
                                     "sulfate", "phosphate-glass", "dataset"))
    if "hanford" in tags and glassy:
        rel = "core"
    elif "hanford" in tags:
        rel = "hanford-adjacent"
    elif glassy:
        rel = "waste-glass-related"
    else:
        rel = "tangential"
    return tags, facility, rel


ERA_RULES = [
    ("pioneering (pre-WTP, ≤2000)", 2000),
    ("WTP design era (2001-2012)", 2012),
    ("EWG / DFLAW era (2013-2023)", 2023),
    ("DFHLW / EWG2.5 era (2024+)", 2100),
]

def era_of(year):
    if not year:
        return ""
    for name, top in ERA_RULES:
        if year <= top:
            return name
    return ""


# ----------------------------------------------------------------------------
# record constructors per source
# ----------------------------------------------------------------------------

def as_list(v, sep=";"):
    """Normalize OSTI's inconsistent list/string fields into a list."""
    if v is None or v == "":
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [p.strip() for p in str(v).split(sep) if p.strip()]


def from_osti(r):
    links = []
    if r.get("doi"):
        links.append("https://doi.org/" + norm_doi(r["doi"]))
    purl = (r.get("links") or "")
    full = ""
    if isinstance(r.get("links"), list):
        for l in r["links"]:
            href = l.get("href", "") if isinstance(l, dict) else str(l)
            links.append(href)
    elif isinstance(r.get("links"), str) and r["links"]:
        # OSTI JSON: links may be a dict {"pdf":..., "html":...} serialized oddly
        try:
            ld = json.loads(r["links"])
            for v in (ld.values() if isinstance(ld, dict) else ld):
                if isinstance(v, str):
                    links.append(v)
        except Exception:  # noqa: BLE001
            links.append(r["links"])
    full = next((u for u in links if "servlets/purl" in u), "")
    rn = r.get("report_number") or ""
    rns = [x.strip() for x in re.split(r";", rn) if x.strip()]
    return {
        "title": clean_text(r.get("title", ""), 500),
        "authors": split_authors_osti(r.get("authors")),
        "year": parse_year(r.get("publication_date")),
        "date": (r.get("publication_date") or "")[:10],
        "doi": norm_doi(r.get("doi")),
        "venue": r.get("journal_name") or "",
        "type": ("journal-article" if (r.get("journal_name") or
                str(r.get("product_type", "")).lower().startswith("journal"))
                else "report"),
        "publisher": r.get("research_org") or r.get("publisher") or "",
        "report_numbers": rns,
        "osti_id": str(r.get("osti_id", "")),
        "openalex_id": "",
        "urls": sorted(set(u for u in links if u.startswith("http"))),
        "pdf_url": full or (f"https://www.osti.gov/servlets/purl/{r.get('osti_id')}"
                            if r.get("osti_id") else ""),
        "abstract": clean_text(r.get("description", "")),
        "cited_by": None,
        "open_access": bool(full),
        "source": "OSTI",
        "source_query": r.get("_source_query", ""),
        "research_orgs": as_list(r.get("research_org")),
        "sponsor_orgs": as_list(r.get("sponsor_org")),
        "subjects": as_list(r.get("subjects")),
    }


def from_openalex(w):
    auths = []
    for a in w.get("authorships", []) or []:
        name = a.get("author", {}).get("display_name", "")
        if name:
            auths.append(name)
    prim = w.get("primary_location") or {}
    src = (prim.get("source") or {})
    urls = []
    pdf = ""
    for loc in (w.get("locations") or []):
        u = loc.get("landing_page_url") or ""
        pu = loc.get("pdf_url") or ""
        if pu and not pdf:
            pdf = pu
        if u and u not in urls:
            urls.append(u)
    oa = w.get("open_access") or {}
    return {
        "title": clean_text(w.get("display_name") or w.get("title") or "", 500),
        "authors": auths,
        "year": w.get("publication_year") or parse_year(w.get("publication_date")),
        "date": (w.get("publication_date") or "")[:10],
        "doi": norm_doi(w.get("doi")),
        "venue": src.get("display_name", ""),
        "type": w.get("type", ""),
        "publisher": src.get("host_organization_name", "") or "",
        "report_numbers": [],
        "osti_id": "",
        "openalex_id": (w.get("id") or "").replace("https://openalex.org/", ""),
        "urls": urls,
        "pdf_url": pdf or oa.get("oa_url", ""),
        "abstract": clean_text(_invert_abstract(w.get("abstract_inverted_index"))),
        "cited_by": w.get("cited_by_count"),
        "open_access": bool(oa.get("is_oa")),
        "source": "OpenAlex",
        "source_query": "",
        "research_orgs": [],
        "sponsor_orgs": [],
        "subjects": [t.get("display_name", "") for t in (w.get("topics") or [])[:4]
                     if isinstance(t, dict)],
    }


def _invert_abstract(inv):
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def from_crossref(it):
    auths = [f"{a.get('family','')}, {a.get('given','')}".strip(", ")
             for a in (it.get("author") or []) if a.get("family") or a.get("given")]
    date = ((it.get("published") or {}).get("date-parts") or [[None]])[0]
    year = date[0] if date else None
    title = (it.get("title") or [""])[0]
    return {
        "title": clean_text(title, 500),
        "authors": auths,
        "year": year,
        "date": "-".join(str(x) for x in date if x) if date else "",
        "doi": norm_doi(it.get("DOI")),
        "venue": (it.get("container-title") or [""])[0],
        "type": it.get("type", ""),
        "publisher": it.get("publisher", ""),
        "report_numbers": [],
        "osti_id": "",
        "openalex_id": "",
        "urls": [f"https://doi.org/{it['DOI']}"] if it.get("DOI") else [],
        "pdf_url": "",
        "abstract": clean_text(it.get("abstract", "")),
        "cited_by": it.get("is-referenced-by-count"),
        "open_access": False,
        "source": "Crossref",
        "source_query": "",
        "research_orgs": [],
        "sponsor_orgs": [],
        "subjects": it.get("subject", []) or [],
    }


def from_glassproperties(r):
    year = parse_year(r.get("date"))
    urls = [r["url"]] if r.get("url") else []
    return {
        "title": clean_text(r.get("title", ""), 500),
        "authors": split_authors_osti(r.get("authors", "")),
        "year": year,
        "date": r.get("date", ""),
        "doi": "",
        "venue": "PNNL technical report series",
        "type": "report",
        "publisher": "Pacific Northwest National Laboratory",
        "report_numbers": [r["report_number"]] if r.get("report_number") else [],
        "osti_id": r.get("osti_id", ""),
        "openalex_id": "",
        "urls": urls,
        "pdf_url": urls[0] if urls else "",
        "abstract": "",
        "cited_by": None,
        "open_access": bool(urls),
        "source": "glassproperties.com",
        "source_query": "bibliography",
        "research_orgs": ["Pacific Northwest National Laboratory"],
        "sponsor_orgs": [],
        "subjects": [],
    }


# ----------------------------------------------------------------------------
# merge logic
# ----------------------------------------------------------------------------

def rec_id(rec):
    h = hashlib.sha1()
    h.update((rec["source"] + "|" + rec["title"][:300]).encode())
    return "r" + h.hexdigest()[:12]


def merge_into(bucket, rec):
    """Try to merge rec into an existing bucket entry; return record id."""
    doi = rec["doi"]
    if doi:
        hit = bucket["by_doi"].get(doi)
        if hit:
            _absorb(hit, rec)
            return hit["id"]
    if rec["osti_id"]:
        hit = bucket["by_osti"].get(rec["osti_id"])
        if hit:
            _absorb(hit, rec)
            return hit["id"]
    if rec["openalex_id"]:
        hit = bucket["by_oa"].get(rec["openalex_id"])
        if hit:
            _absorb(hit, rec)
            return hit["id"]
    # fuzzy title match
    nt = norm_title(rec["title"])
    if nt and len(nt) > 15:
        cand = bucket["by_title"].get(nt[:60])
        if cand and tsim(nt, norm_title(cand["title"])) >= 0.90:
            # distinct report numbers with no overlap => distinct documents
            # (e.g. EWG2.5 vs EWG3.0 model updates share near-identical titles)
            a_rns = {r.upper().replace(".", "").replace(" REV", "REV")
                     for r in (rec.get("report_numbers") or []) if r.strip()}
            b_rns = {r.upper().replace(".", "").replace(" REV", "REV")
                     for r in (cand.get("report_numbers") or []) if r.strip()}
            same_series = bool(a_rns and b_rns and not
                               (a_rns & b_rns))
            # but a record number like "PNNL-35884 Rev 1" also belongs with
            # "PNNL-35884": check containment by prefix token
            if same_series:
                def base_tokens(rns):
                    return {r.split()[0].replace("REV", "").rstrip("-")
                            for r in rns}
                if base_tokens(a_rns) & base_tokens(b_rns):
                    same_series = False
            if not same_series:
                _absorb(cand, rec)
                return cand["id"]
    # new record
    rid = rec_id(rec)
    entry = {"id": rid, **rec}
    bucket["all"][rid] = entry
    if doi:
        bucket["by_doi"][doi] = entry
    if rec["osti_id"]:
        bucket["by_osti"][rec["osti_id"]] = entry
    if rec["openalex_id"]:
        bucket["by_oa"][rec["openalex_id"]] = entry
    if nt:
        bucket["by_title"][nt[:60]] = entry
    entry.setdefault("_sources", []).append(rec["source"])
    return rid


def _absorb(target, rec):
    """Merge rec's fields into target, filling blanks and merging lists."""
    for srcs in (target,):
        pass
    src = rec["source"]
    if src not in target.setdefault("_sources", []):
        target["_sources"].append(src)
    for key in ("title", "venue", "publisher", "pdf_url", "abstract", "year", "date", "type"):
        if not target.get(key) and rec.get(key):
            target[key] = rec[key]
    if not target.get("doi") and rec.get("doi"):
        target["doi"] = rec["doi"]
        target["urls"] = sorted(set(target.get("urls", []) + rec.get("urls", [])))
    if not target.get("osti_id") and rec.get("osti_id"):
        target["osti_id"] = rec["osti_id"]
    if not target.get("openalex_id") and rec.get("openalex_id"):
        target["openalex_id"] = rec["openalex_id"]
    if not target.get("pdf_url") and rec.get("pdf_url"):
        target["pdf_url"] = rec["pdf_url"]
    for key in ("authors", "urls", "report_numbers", "research_orgs", "sponsor_orgs"):
        merged = set(target.get(key) or []) | set(rec.get(key) or [])
        target[key] = sorted(merged, key=str)
    if target.get("cited_by") is None and rec.get("cited_by") is not None:
        target["cited_by"] = rec["cited_by"]
    if rec.get("open_access"):
        target["open_access"] = True
    if len(rec.get("abstract", "")) > len(target.get("abstract", "")):
        target["abstract"] = rec["abstract"]
    if not target.get("source_query") and rec.get("source_query"):
        target["source_query"] = rec["source_query"]


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def load_json(path):
    try:
        return json.load(open(path))
    except Exception as e:  # noqa: BLE001
        print(f"  ! failed to parse {path}: {e}")
        return None


def main():
    bucket = {"all": {}, "by_doi": {}, "by_osti": {}, "by_oa": {}, "by_title": {}}
    counts = Counter()

    def add(rec, src):
        if not rec.get("title") or len(rec.get("title", "")) < 8:
            return
        merge_into(bucket, rec)
        counts[src] += 1

    # OSTI
    osti_dir = os.path.join(DATA, "raw_osti")
    for fn in sorted(os.listdir(osti_dir)):
        data = load_json(os.path.join(osti_dir, fn))
        if not data:
            continue
        n0 = len(bucket["all"])
        for r in data.get("records", []):
            add(from_osti(r), f"OSTI:{fn}")
        print(f"OSTI {fn}: +{len(bucket['all']) - n0}")

    # OpenAlex
    oa_dir = os.path.join(DATA, "raw_openalex")
    for fn in sorted(os.listdir(oa_dir)):
        data = load_json(os.path.join(oa_dir, fn))
        if not data:
            continue
        n0 = len(bucket["all"])
        for w in data.get("results", []):
            rec = from_openalex(w)
            lbl = data.get("label", "")
            rec["source_query"] = f"openalex:{lbl}"
            add(rec, f"OpenAlex:{fn}")
        print(f"OpenAlex {fn}: +{len(bucket['all']) - n0}")

    # Crossref / misc
    misc_dir = os.path.join(DATA, "raw_misc")
    if os.path.isdir(misc_dir):
        for fn in sorted(os.listdir(misc_dir)):
            data = load_json(os.path.join(misc_dir, fn))
            if not data:
                continue
            n0 = len(bucket["all"])
            if fn.startswith("crossref_"):
                for it in data.get("items", []):
                    rec = from_crossref(it)
                    rec["source_query"] = f"crossref:{data.get('label','')}"
                    add(rec, f"Crossref:{fn}")
            elif fn.startswith("pnnl_"):
                for r in data.get("results", []):
                    rec = from_glassproperties(r)
                    add(rec, f"pnnl:{fn}")
            elif fn.startswith("arxiv_"):
                for e in data.get("entries", []):
                    rec = {
                        "title": clean_text(e["title"], 500),
                        "authors": e.get("authors", []),
                        "year": parse_year(e.get("published")),
                        "date": (e.get("published") or "")[:10],
                        "doi": "", "venue": "arXiv", "type": "preprint",
                        "publisher": "arXiv", "report_numbers": [],
                        "osti_id": "",
                        "openalex_id": "",
                        "urls": [e["id"]],
                        "pdf_url": e["id"].replace("abs", "pdf"),
                        "abstract": clean_text(e.get("summary", "")),
                        "cited_by": None, "open_access": True,
                        "source": "arXiv", "source_query": f"arxiv:{data.get('label','')}",
                        "research_orgs": [], "sponsor_orgs": [], "subjects": [],
                    }
                    add(rec, f"arXiv:{fn}")
            print(f"misc {fn}: +{len(bucket['all']) - n0}")

    # glassproperties bibliography
    gp = os.path.join(DATA, "raw_web", "glassproperties_pnnl_records.json")
    if os.path.exists(gp):
        data = load_json(gp) or []
        n0 = len(bucket["all"])
        for r in data:
            add(from_glassproperties(r), "glassproperties")
        print(f"glassproperties: +{len(bucket['all']) - n0}")

    # hand-curated lineage records (URL-verified, incl. administrative-record docs)
    cu = os.path.join(DATA, "raw_curated", "curated.json")
    if os.path.exists(cu):
        data = load_json(cu) or []
        n0 = len(bucket["all"])
        for r in data:
            rec = {
                "title": clean_text(r.get("title", ""), 500),
                "authors": r.get("authors", []),
                "year": r.get("year"),
                "date": "",
                "doi": "",
                "venue": r.get("venue", ""),
                "type": r.get("type", "report"),
                "publisher": r.get("publisher", ""),
                "report_numbers": r.get("report_numbers", []),
                "osti_id": str(r.get("osti_id", "") or ""),
                "openalex_id": "",
                "urls": ([r["pdf_url"]] if r.get("pdf_url") else []),
                "pdf_url": r.get("pdf_url", ""),
                "abstract": clean_text(r.get("abstract", "")),
                "cited_by": None,
                "open_access": bool(r.get("pdf_url")),
                "source": "curated",
                "source_query": "curated-lineage",
                "research_orgs": [r.get("publisher", "")] if r.get("publisher") else [],
                "sponsor_orgs": [],
                "subjects": r.get("topics", []),
                "note": r.get("note", ""),
            }
            rid = merge_into(bucket, rec)
            # apply curated topics onto the merged entry
            hit = bucket["all"].get(rid)
            if hit and r.get("topics"):
                hit["curated_topics"] = r["topics"]
            if hit and r.get("note"):
                hit["note"] = r["note"]
        print(f"curated: +{len(bucket['all']) - n0}")

    # classify everything
    records = list(bucket["all"].values())
    for rec in records:
        tags, facility, rel = classify(rec)
        rec["topics"] = tags
        rec["facility"] = facility
        rec["relevance"] = rel
        rec["era"] = era_of(rec.get("year"))

    # stable sort: relevance, year desc, title
    order = {"core": 0, "hanford-adjacent": 1, "waste-glass-related": 2, "tangential": 3}
    records.sort(key=lambda r: (order.get(r["relevance"], 9),
                                -(r.get("year") or 0),
                                r.get("title", "")))

    out_jsonl = os.path.join(DATA, "records.jsonl")
    with open(out_jsonl, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # summary
    by_rel = Counter(r["relevance"] for r in records)
    by_type = Counter(r.get("type", "") for r in records)
    by_decade = Counter(f"{(r.get('year') or 0)//10*10}s" for r in records if r.get("year"))
    top_tags = Counter(t for r in records for t in r["topics"]).most_common(25)
    summary = {
        "total_records": len(records),
        "by_relevance": dict(by_rel),
        "by_type": dict(by_type.most_common(15)),
        "by_decade": dict(sorted(by_decade.items())),
        "top_tags": dict(top_tags),
        "by_source": dict(counts),
        "unique_by_source": {
            "OSTI": sum(1 for r in records if "OSTI" in r["_sources"]),
            "OpenAlex": sum(1 for r in records if "OpenAlex" in r["_sources"]),
            "Crossref": sum(1 for r in records if "Crossref" in r["_sources"]),
            "arXiv": sum(1 for r in records if "arXiv" in r["_sources"]),
            "glassproperties.com": sum(1 for r in records if "glassproperties.com" in r["_sources"]),
        },
    }
    with open(os.path.join(DATA, "records_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print(json.dumps(summary, indent=1))
    print(f"\nwrote {len(records)} records -> {out_jsonl}")


if __name__ == "__main__":
    main()
