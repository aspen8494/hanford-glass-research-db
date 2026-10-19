# Hanford Glass Research Database

An open-source, searchable compilation of publicly available research on
**Hanford waste-glass composition modeling and glass formulation algorithm
development** — spanning the pioneering vitrification work of the 1970s through
the current Enhanced Waste Glass (EWG) property-composition models used for the
Direct-Feed High-Level Waste (DFHLW) and Low-Activity Waste (LAW) facilities.

[![status](https://img.shields.io/badge/records-17k%2B-blue)](data/records_summary.json)
[![license](https://img.shields.io/badge/license-CC0%201.0-green)](#license)

## What this is

Every record is a **published document** (journal article, technical report,
conference paper, or preprint) related to glass composition, glass property
models, glass formulation algorithms, or vitrification for Hanford tank waste.

Each record carries:

- title, authors, year, venue/publisher, document type
- identifiers: DOI, OSTI ID, OpenAlex ID, PNNL/VSL/SRNL/WRPS report numbers
- links: DOI resolver, OSTI biblio page, direct PDF (when open access)
- abstract (when available from the source API)
- topic tags (rule-based: `LAW`, `HLW`, `EWG`, `property-model`, `durability`,
  `nepheline`, `sulfate`, `melter`, `viscosity`, `machine-learning`, …)
- facility relevance (`LAW` / `HLW` / `both`) and an era bucket:

| Era | Meaning |
|---|---|
| pioneering (pre-WTP, ≤2000) | HWVP-era and early LAW vitrification studies |
| WTP design era (2001–2012) | baseline property models (PNNL-13573/14060, WTP contracts) |
| EWG / DFLAW era (2013–2023) | Enhanced Waste Glass program, DFLAW flowsheets |
| DFHLW / EWG2.5 era (2024+) | current DFHLW feed-vector glass models |

- relevance tiers (rule-based):

| Tier | Meaning |
|---|---|
| `core` | Hanford + glass composition/model/formulation focus |
| `hanford-adjacent` | Hanford waste processing but not glass-composition focused |
| `waste-glass-related` | waste glass science from other sites (Savannah River, West Valley, etc.) |
| `tangential` | matched a query but marginal to the topic |

## Quick start

```bash
# search the SQLite database (full-text)
sqlite3 db/hanford_glass_research.db \
  "SELECT r.year, r.title, r.doi FROM records_fts f
   JOIN records r ON r.id = f.id
   WHERE records_fts MATCH 'nepheline AND hanford' LIMIT 20;"

# or filter the flat CSV
python3 -c "import pandas as pd; df = pd.read_csv('db/hanford_glass_research.csv'); \
  print(df[df.relevance=='core'][['year','title','doi']].head(30))"

# or open the zero-dependency UI
cd ui && python3 -m http.server 8000
# then visit http://localhost:8000
```

## Data sources & method

| Source | What it contributed |
|---|---|
| [OSTI](https://www.osti.gov/api/v1/docs) (DOE OSTI API v1) | DOE technical reports (PNNL, SRNL, VSL, WRPS, early PNL) — the backbone for gray literature |
| [OpenAlex](https://docs.openalex.org) | journal articles with citation counts, OA links, topics |
| [Crossref](https://api.crossref.org) | journal metadata + abstracts for non-DOE venues |
| arXiv | niche preprints (glass modeling, ML surrogates) |
| Curated bibliographies | glassproperties.com PNNL vitrification bibliography |

All harvesting scripts are in [`scripts/`](scripts/) and are re-runnable:

```bash
python3 scripts/harvest_osti.py      <batch> <queries.json>   # DOE reports
python3 scripts/harvest_openalex.py  <queries.json>           # journal literature
python3 scripts/harvest_misc.py      <queries.json>           # Crossref/arXiv/PNNL
python3 scripts/build_records.py                              # normalize+merge+classify
python3 scripts/build_sqlite.py                               # SQLite + CSV + UI data
python3 scripts/verify_records.py 60                          # spot-check link rot
```

The exact queries used are versioned under [`queries/`](queries/).

## Known caveats

- **Coverage is broad, not exhaustive.** Journal articles from 2025–2026 may be
  missing where indexing lags; very early Hanford reports (pre-1976) exist only
  in the Hanford_DEClassified_Document Repository / Hanford Public Information
  Repository and are not yet systematically included.
- **VSL report numbers** (VSL-XXR####-1) are generally *not* indexed by OSTI;
  they appear here only when referenced by an indexed PNNL/SRNL report or when
  they have DOIs. Many VSL final reports are only available via ORP document
  request (the MACT Equivalency / administrative record).
- **Relevance/era/tag classification is rule-based** — designed to be
  transparent and re-runnable, not perfect. The `tangential` tier exists to
  keep recall high; filter to `core` for precision.
- **Abstracts** are included only where the source API provided them (OSTI and
  OpenAlex usually do; Crossref sometimes).
- Some OSTI records are metadata-only (no PDF) — check `urls`.

## License / reuse

- **The database (records.jsonl, CSV, SQLite) is released CC0-1.0.**
- Harvesting scripts are MIT-licensed.
- The underlying bibliographic metadata comes from OSTI/OpenAlex/Crossref —
  each has its own (permissive) reuse terms; DOIs link to publisher pages whose
  content remains copyrighted by the publisher.

## Provenance

Built October 2026 by automated harvesting (scripts in this repo, no manual
curation). Regenerate the entire database from scratch with the four steps
above. `data/records_summary.json` holds the build statistics for every run.
