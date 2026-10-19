# OpenAlex API — quirks learned while building the Hanford glass research database.

1. Cursor paging (`cursor=*` then `meta.next_cursor`) now requires a free API
   key; without one, page 2+ returns HTTP 401 with body:
   `{"error":"API key required for cursor paging"}`.
   Get one at https://openalex.org/settings/api (1 minute, email signup).
2. Classic page-based paging (`page=2,3,...`) still works WITHOUT a key,
   up to 10,000 records per query — use it when keyless.
3. Anonymous rate limit ≈ 1,000 credits per rolling window (each request costs
   10 credits → ~100 requests/window). Bursting page 2 of many queries in
   parallel triggers 401s that LOOK like auth errors. With a key: 100k credits/day.
4. Always send `mailto=` (polite pool) and a browser-like User-Agent.
5. `search=` is fuzzy/relevance-ranked; combine with `filter=` for precision
   (e.g. `raw_author_name.search:Vienna`). Check `meta.count` before paging.
6. Use `select=` to trim payloads (abstract_inverted_index is the big one;
   it must be re-inflated: position→word, sort by position).
7. Key handling in this repo: scripts read `~/.openalex_api_key` if present.
   NEVER commit the key — it's user-specific.
