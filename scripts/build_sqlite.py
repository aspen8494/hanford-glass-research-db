"""Build the SQLite database + exports from data/records.jsonl.

Outputs (in db/):
  hanford_glass_research.db   SQLite with records, authors, topics tables + FTS5
  hanford_glass_research.csv  flat CSV (one row per record)
  ui/records_ui.json          trimmed JSON for the static search UI
"""
import csv
import json
import os
import sqlite3

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(WS, "data")
DB_DIR = os.path.join(WS, "db")
UI_DIR = os.path.join(WS, "ui")
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(UI_DIR, exist_ok=True)

DB_PATH = os.path.join(DB_DIR, "hanford_glass_research.db")
CSV_PATH = os.path.join(DB_DIR, "hanford_glass_research.csv")
UI_PATH = os.path.join(UI_DIR, "records_ui.json")

records = [json.loads(line) for line in open(os.path.join(DATA, "records.jsonl"))]
# published artifacts keep the curated tiers; 'tangential' stays in records.jsonl only
n_all = len(records)
records = [r for r in records if r.get("relevance") != "tangential"]
print(f"loaded {n_all} records; publishing {len(records)} (excluding 'tangential')")

# ---------------------------------------------------------------- SQLite
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
con = sqlite3.connect(DB_PATH)
cur = con.cursor()
cur.executescript("""
CREATE TABLE records (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  year INTEGER,
  date TEXT,
  doi TEXT,
  venue TEXT,
  type TEXT,
  publisher TEXT,
  osti_id TEXT,
  openalex_id TEXT,
  pdf_url TEXT,
  abstract TEXT,
  cited_by INTEGER,
  open_access INTEGER,
  facility TEXT,
  relevance TEXT,
  era TEXT,
  sources TEXT,         -- JSON array of contributing sources
  source_query TEXT
);
CREATE TABLE urls (
  record_id TEXT,
  url TEXT
);
CREATE TABLE report_numbers (
  record_id TEXT,
  report_number TEXT
);
CREATE TABLE research_orgs (
  record_id TEXT,
  org TEXT
);
CREATE TABLE sponsor_orgs (
  record_id TEXT,
  org TEXT
);
CREATE TABLE authors (
  record_id TEXT,
  position INTEGER,
  name TEXT
);
CREATE TABLE topics (
  record_id TEXT,
  topic TEXT
);
CREATE INDEX idx_authors_name ON authors(name);
CREATE INDEX idx_topics_topic ON topics(topic);
CREATE INDEX idx_records_year ON records(year);
CREATE INDEX idx_records_relevance ON records(relevance);
""")
cur.execute("""CREATE VIRTUAL TABLE records_fts USING fts5(
  id UNINDEXED, title, abstract, authors, venue, report_numbers,
  tokenize='porter unicode61')""")

for r in records:
    cur.execute(
        "INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (r["id"], r["title"], r.get("year"), r.get("date", ""), r.get("doi", ""),
         r.get("venue", ""), r.get("type", ""), r.get("publisher", ""),
         r.get("osti_id", ""), r.get("openalex_id", ""), r.get("pdf_url", ""),
         r.get("abstract", ""), r.get("cited_by"), 1 if r.get("open_access") else 0,
         r.get("facility", ""), r.get("relevance", ""), r.get("era", ""),
         json.dumps(r.get("_sources", [])), r.get("source_query", "")))
    for i, a in enumerate(r.get("authors", [])):
        cur.execute("INSERT INTO authors VALUES (?,?,?)", (r["id"], i, a))
    for t in r.get("topics", []):
        cur.execute("INSERT INTO topics VALUES (?,?)", (r["id"], t))
    for u in r.get("urls", []):
        cur.execute("INSERT INTO urls VALUES (?,?)", (r["id"], u))
    for rn in r.get("report_numbers", []):
        cur.execute("INSERT INTO report_numbers VALUES (?,?)", (r["id"], rn))
    for o in r.get("research_orgs", []):
        cur.execute("INSERT INTO research_orgs VALUES (?,?)", (r["id"], o))
    for o in r.get("sponsor_orgs", []):
        cur.execute("INSERT INTO sponsor_orgs VALUES (?,?)", (r["id"], o))
    cur.execute("INSERT INTO records_fts VALUES (?,?,?,?,?,?)",
                (r["id"], r["title"], r.get("abstract", ""),
                 "; ".join(r.get("authors", [])), r.get("venue", ""),
                 " ".join(r.get("report_numbers", []))))

con.commit()
con.execute("VACUUM")
n = cur.execute("SELECT count(*) FROM records").fetchone()[0]
print(f"sqlite: {n} records -> {DB_PATH} ({os.path.getsize(DB_PATH)/1e6:.0f} MB)")
con.close()

# ---------------------------------------------------------------- CSV
cols = ["id", "title", "authors", "year", "date", "doi", "venue", "type",
        "publisher", "osti_id", "openalex_id", "pdf_url", "relevance",
        "facility", "era", "topics", "report_numbers", "cited_by",
        "open_access", "urls", "abstract"]
with open(CSV_PATH, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(cols)
    for r in records:
        w.writerow([
            r["id"], r["title"], "; ".join(r.get("authors", [])),
            r.get("year", ""), r.get("date", ""), r.get("doi", ""),
            r.get("venue", ""), r.get("type", ""), r.get("publisher", ""),
            r.get("osti_id", ""), r.get("openalex_id", ""), r.get("pdf_url", ""),
            r.get("relevance", ""), r.get("facility", ""), r.get("era", ""),
            ";".join(r.get("topics", [])), ";".join(r.get("report_numbers", [])),
            r.get("cited_by", ""), 1 if r.get("open_access") else 0,
            " | ".join(r.get("urls", [])),
            (r.get("abstract", "") or "")[:1500],
        ])
print(f"csv -> {CSV_PATH}")

# ---------------------------------------------------------------- UI JSON
ui_records = []
for r in records:
    if r.get("relevance") == "tangential":
        continue
    ui_records.append({
        "id": r["id"],
        "t": r["title"],
        "a": "; ".join(r.get("authors", [])[:6]) + (" et al." if len(r.get("authors", [])) > 6 else ""),
        "y": r.get("year"),
        "doi": r.get("doi", ""),
        "v": r.get("venue", ""),
        "ty": r.get("type", ""),
        "rel": r.get("relevance", ""),
        "fac": r.get("facility", ""),
        "era": r.get("era", ""),
        "tp": r.get("topics", []),
        "rn": r.get("report_numbers", []),
        "pdf": r.get("pdf_url", ""),
        "urls": r.get("urls", [])[:4],
        "ost": r.get("osti_id", ""),
        "cb": r.get("cited_by"),
        "oa": 1 if r.get("open_access") else 0,
        "ab": (r.get("abstract", "") or "")[:1200],
        "src": r.get("_sources", []),
    })
with open(UI_PATH, "w") as f:
    json.dump({"generated": "see data/records_summary.json for counts",
               "count": len(ui_records), "records": ui_records}, f,
              ensure_ascii=False, separators=(",", ":"))
print(f"ui json ({len(ui_records)} records, {os.path.getsize(UI_PATH)/1e6:.1f} MB) -> {UI_PATH}")
