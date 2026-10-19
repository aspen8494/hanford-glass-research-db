# OSTI (DOE Office of Scientific and Technical Information) — API quirks learned
# while building the Hanford glass research database. Keep with the harvest scripts.
#
# 1. User-Agent: bare/short UA strings get an HTML WAF page (HTTP 200, not JSON).
#    Use a full browser UA string.
# 2. Response format is controlled by the `Accept: application/json` HEADER.
#    The `format=json` query param is ignored by some endpoints — without the
#    header you can get XML that contains invalid control characters, which
#    breaks ElementTree. Sanitize `[\x00-\x08\x0b\x0c\x0e-\x1f]` before parsing.
# 3. `X-Total-Count` header = number of records IN THIS PAGE, not the total.
#    To find totals, keep paging until a short page.
# 4. Field params silently fall back to "everything" when given an unknown
#    field name (e.g. `authors=Vienna` is WRONG -> matches all records;
#    the correct param is `author=Vienna`).
# 5. Searchable fields (per /api/v1/docs): author, title, identifier,
#    fulltext, biblio, subject, sponsor_org, research_org, doi, osti_id,
#    publication_date_start/end, has_fulltext, sort, order, rows, page.
# 6. `identifier=<report number>` does a NARROW code match that MISSES many
#    real PNNL reports. `fulltext=<report number>` is the reliable way to find
#    a record by report number (e.g. PNNL-35884 is only findable via fulltext=).
# 7. `report_number=` is a fuzzy/semantic match — mostly garbage for exact
#    report-number lookup.
# 8. Max rows per page ~500; page param starts at 1.
# 9. Rate: no documented limit; 0.5s between calls is safely polite.
# 10. Record `links` field is inconsistent: sometimes a JSON-ish string, sometimes
#     a list, sometimes absent. PDF lives at
#     https://www.osti.gov/servlets/purl/<osti_id> when has_fulltext.
