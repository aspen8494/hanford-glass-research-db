"""Parse the glassproperties.com PNNL bibliography HTML table into JSON records."""
import json
import os
import re

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML_PATH = os.path.join(WS, "data", "raw_web", "glassproperties_pnnl.html")
OUT = os.path.join(WS, "data", "raw_web", "glassproperties_pnnl_records.json")

html = open(HTML_PATH).read()

# split into <tr>...</tr> rows
rows = re.findall(r"<tr.*?</tr>", html, re.S | re.I)
records = []
for row in rows:
    # report number: first <a> inside span CLASS='result-id'
    m_rid = re.search(r"<span\s+CLASS='result-id'>\s*<a href='([^']+)'>\s*([^<]+?)\s*</a>",
                      row, re.S | re.I)
    # title: the <td> containing font-weight: bold
    m_title = re.search(r"font-weight:\s*bold;?'?>([^<]+?)<", row, re.S | re.I)
    if not m_title:
        m_title = re.search(r"<p[^>]*>\s*([A-Z][^<]{15,})<", row)
    # authors + date cells
    cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S | re.I)
    authors, date = "", ""
    if len(cells) >= 4:
        authors = re.sub(r"<[^>]+>", "", cells[2]).strip()
        date = re.sub(r"<[^>]+>", "", cells[3]).strip()
        date = re.sub(r"&nbsp;?", " ", date)
        date = re.sub(r"\s+", " ", date).strip()
    if not (m_rid or m_title):
        continue
    rec = {
        "source": "glassproperties_bibliography",
        "report_number": (m_rid.group(2).strip() if m_rid else ""),
        "url": (m_rid.group(1).strip() if m_rid else ""),
        "title": re.sub(r"\s+", " ", m_title.group(1)).strip() if m_title else "",
        "authors": re.sub(r"\s+", " ", authors),
        "date": date,
    }
    oid = re.search(r"osti_id=(\d+)", rec["url"])
    if oid:
        rec["osti_id"] = oid.group(1)
    if rec["title"]:
        records.append(rec)

with open(OUT, "w") as f:
    json.dump(records, f, indent=1)
print(f"parsed {len(records)} bibliography records -> {OUT}")
for r in records[:5]:
    print("  -", r.get("report_number", "")[:30], "|", r["title"][:70])
