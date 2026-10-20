"""Export findings.json -> compact JSON embedded in an interactive HTML widget.

Produces ui/topic_map.html: concept map (force-ish grid of seed words with NN
lists), decade x topic heatmap (relative profiles), era bridges, anomaly docs.
Uses only the app theme vars and vanilla JS so it runs inside ::preview.
"""
import json
import os

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NN = os.path.join(WS, "data", "nn")

f = json.load(open(os.path.join(NN, "findings.json")))
payload = {
    "labels": f["labels"],
    "coherence": f["coherence"],
    "decade_rel": f["decade_rel_profiles"],
    "docs_per_decade": f["docs_per_decade"],
    "concept_map": f["concept_map"],
    "era_bridges": f["era_bridges_v2"],
    "anomalies": f["anomalies"],
    "topics_detail": f["topics_detail"],
    "slopes": f["slopes"],
}
DECADES = ["1940", "1950", "1960", "1970", "1980", "1990", "2000", "2010", "2020"]

html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  :root { color-scheme: light dark; }
  body { margin:0; padding:14px 16px; font: 13px/1.45 -apple-system, var(--foreground);
         color: var(--foreground); background: transparent; }
  h2 { font-size: 15px; margin: 18px 0 6px; }
  h2:first-child { margin-top: 0; }
  .sub { color: var(--muted-foreground); margin: 0 0 10px; font-size: 12px; }
  .tabs { display:flex; gap:6px; margin-bottom:10px; flex-wrap:wrap; }
  .tab { padding:4px 12px; border:1px solid var(--border); border-radius:8px;
         cursor:pointer; user-select:none; background: var(--card); }
  .tab.on { background: var(--accent); color:#fff; border-color: var(--accent); }
  table { border-collapse: collapse; width:100%; }
  td, th { border:1px solid var(--border); padding:3px 7px; font-size:12px; }
  th { color: var(--muted-foreground); text-align:left; font-weight:600; }
  .cell { width:52px; text-align:center; cursor:default; }
  .map { display:flex; flex-wrap:wrap; gap:8px; }
  .seed { border:1px solid var(--border); border-radius:10px; padding:8px 10px;
          background: var(--card); min-width:170px; }
  .seed b { cursor:pointer; }
  .seed b:hover { color: var(--accent); }
  .seed .nn { color: var(--muted-foreground); font-size:11.5px; margin-top:2px; }
  .bridge, .anom { padding:6px 10px; border:1px solid var(--border); border-radius:8px;
                   margin-bottom:6px; background: var(--card); }
  .muted { color: var(--muted-foreground); }
  .topiclist { display:flex; flex-direction:column; gap:4px; }
  .trow { display:flex; gap:8px; align-items:baseline; padding:3px 8px;
          border:1px solid var(--border); border-radius:8px; background: var(--card); }
  .trow .coh { font-variant-numeric: tabular-nums; min-width:46px; text-align:right;
               color: var(--muted-foreground); font-size:11.5px; }
  .trow .terms { color: var(--muted-foreground); font-size:12px; }
  .detail { margin:6px 0 10px 18px; padding:8px 12px; border-left:3px solid var(--accent);
            display:none; background: var(--card); border-radius:0 8px 8px 0; }
  .detail.show { display:block; }
  .bar { height:8px; border-radius:4px; background: var(--accent); display:inline-block;
         vertical-align:middle; margin-right:6px; }
</style>
</head>
<body>
<div id="app"></div>
<script>
const D = __PAYLOAD__;
const DECADES = __DECADES__;
const esc = s => String(s).replace(/[&<>String.fromCharCode(34)]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c] ?? c));

let view = 'map';
function render() {
  const app = document.getElementById('app');
  if (view === 'map') app.innerHTML = tabs() + conceptMap();
  else if (view === 'heat') app.innerHTML = tabs() + heatmap();
  else if (view === 'bridge') app.innerHTML = tabs() + bridges();
  else if (view === 'anom') app.innerHTML = tabs() + anomalies();
  else app.innerHTML = tabs() + topicList();
  wire();
}
function tabs() {
  const names = [['map','Concept map'],['heat','Decades x topics'],['bridge','Era bridges'],['anom','Anomalies'],['topics','Topic browser']];
  return `<div class="tabs">` + names.map(([v, n]) =>
    `<div class="tab ${view===v?'on':''}" data-v="${v}">${n}</div>`).join('') + `</div>`;
}
function conceptMap() {
  const seeds = Object.entries(D.concept_map);
  let h = `<h2>Learned concept neighborhoods</h2>
  <p class="sub">Skip-gram embeddings trained on 6,719 Hanford glass documents (128-d, 25 epochs).
  Click a seed to probe it below.</p><div class="map">`;
  for (const [seed, nns] of seeds) {
    h += `<div class="seed" data-seed="${esc(seed)}"><b>${esc(seed)}</b>
          <div class="nn">${nns.slice(0,6).map(([w,s]) => esc(w)).join(' · ')}</div></div>`;
  }
  return h + `</div><div id="probe"></div>`;
}
function heatmap() {
  const topics = Object.keys(D.decade_rel[DECADES[0]]).map(Number).sort((a,b)=>a-b);
  const maxv = Math.max(2.0, ...DECADES.flatMap(d => Object.values(D.decade_rel[d]).map(Number)));
  let h = `<h2>Decade × topic relative activity</h2>
  <p class="sub">Cell = topic share in decade ÷ overall share (1.0 = typical). Hover for numbers.
  Green = over-represented, red = under-represented.</p>
  <table><tr><th>topic</th>${DECADES.map(d=>`<th>${d}s</th>`).join('')}</tr>`;
  const sorted = topics.slice().sort((a,b) => (D.coherence[b]||-1) - (D.coherence[a]||-1));
  for (const k of sorted) {
    h += `<tr><td class="muted">T${k} ${esc(D.labels[k]||'')}</td>`;
    for (const d of DECADES) {
      const v = Number(D.decade_rel[d]?.[k] ?? 0);
      const n = D.docs_per_decade[d] || 0;
      const g = v >= 1 ? `rgba(48,164,108,${Math.min(1, v/maxv)})` : `rgba(220,80,80,${Math.min(1,(1-v)/maxv)})`;
      const style = `background:${g};color:${v>maxv*0.45?'#fff':'inherit'}`;
      h += `<td class="cell" style="${style}" title="${d}s: ${v.toFixed(2)}x overall (n=${n} docs)">${v?v.toFixed(1):''}</td>`;
    }
    h += `</tr>`;
  }
  return h + `</table>`;
}
function bridges() {
  let h = `<h2>Cross-era bridges</h2><p class="sub">Correlation between decade topic-mix profiles
  (relative activity), skipping adjacent decades — which research interests echo across eras.</p>`;
  const items = Object.entries(D.era_bridges).sort((a,b)=>b[1].corr-a[1].corr);
  for (const [pair, v] of items) {
    const [a,b] = pair.split('-');
    const shared = v.shared_topics.map(([k,lab,r]) =>
      `<span title="${esc(lab)}">T${k}</span> <span class="muted">(${esc(r)})</span>`).join(', ');
    const w = Math.max(4, Math.round(v.corr*120));
    h += `<div class="bridge"><b>${a}s ↔ ${b}s</b> <span class="muted">corr ${v.corr.toFixed(2)}</span>
          <span class="bar" style="width:${w}px"></span><br><span class="muted">via</span> ${shared}</div>`;
  }
  return h;
}
function anomalies() {
  let h = `<h2>Outlier documents</h2><p class="sub">Documents whose topic mixture is most unlike
  every other document (isolated profiles — candidates for manual review).</p>`;
  for (const a of D.anomalies) {
    h += `<div class="anom"><b>${esc(a.title)}</b><br>
          <span class="muted">${a.year ?? 'n.d.'} · dominant topic T${a.dominant_topic} (${esc(a.label)}) ·
          θmax ${a.theta_max} · nearest-doc cos ${a.max_sim_to_any_doc}</span></div>`;
  }
  return h;
}
function topicList() {
  const ks = Object.keys(D.topics_detail).map(Number).sort((a,b)=>D.coherence[b]-D.coherence[a]);
  let h = `<h2>Topic browser</h2><p class="sub">All 30 ProdLDA topics, sorted by NPMI coherence.
  Click to expand terms, exemplar documents, and emergence era.</p><div class="topiclist">`;
  for (const k of ks) {
    const t = D.topics_detail[k];
    h += `<div class="trow" data-k="${k}"><span class="coh">${t.coherence>=0?'+':''}${t.coherence.toFixed(3)}</span>
          <b>T${k}</b><span>${esc(t.label)}</span></div>
          <div class="detail" id="d${k}">
            <div><b>Top terms:</b> ${t.top_terms.map(([w,v])=>esc(w)).join(', ')}</div>
            <div style="margin-top:4px"><b>Emergence:</b> ${t.emergence ? t.emergence+'s' : 'active across all eras'}</div>
            <div style="margin-top:4px"><b>Exemplars:</b> ${t.exemplars.map(e =>
              `<div class="muted">[${e.year ?? 'n.d.'}] ${esc(e.title)}</div>`).join('')}</div>
          </div>`;
  }
  return h + `</div>`;
}
function wire() {
  document.querySelectorAll('.tab').forEach(el =>
    el.onclick = () => { view = el.dataset.v; render(); });
  document.querySelectorAll('.seed b').forEach(el =>
    el.onclick = () => {
      const seed = el.textContent, nns = D.concept_map[seed];
      document.getElementById('probe').innerHTML =
        `<div class="bridge"><b>${esc(seed)}</b>: ` +
        nns.map(([w,s]) => `${esc(w)} <span class="muted">${s}</span>`).join(' · ') + `</div>`;
    });
  document.querySelectorAll('.trow').forEach(el =>
    el.onclick = () => document.getElementById('d'+el.dataset.k).classList.toggle('show'));
}
render();
</script>
</body>
</html>
"""
html = html.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
html = html.replace("__DECADES__", json.dumps(DECADES))


# --- post-inject "Modern era (DOE)" tab (added 2026-10-19) ---
import re as _re
ma = json.load(open(os.path.join(NN, "modern_doe_analysis.json")))
payload_mod = {"labels": ma["labels"], "coherence": ma["coherence"],
               "period_shares": ma["period_shares"], "growth": ma["growth"],
               "top_terms": ma["top_terms"], "exemplars": ma["exemplars"],
               "n_docs": ma["n_docs"]}
html = html.replace("['topics','Topic browser']",
                    "['topics','Topic browser'],['moddoe','Modern era (DOE)']")
renderer = open(os.path.join(WS, "scripts", "nn", "_moddoe_renderer.js")).read()
html = html.replace("function render() {", renderer + "\nfunction render() {")
html = html.replace("else app.innerHTML = tabs() + topicList();",
                    "else if (view === 'moddoe') app.innerHTML = tabs() + modernDOE();\n  else app.innerHTML = tabs() + topicList();")
html = html.replace("<script>\n", "<script>\nwindow.__MODDOE__ = " + json.dumps(payload_mod, ensure_ascii=False) + ";\n")
html = html.replace("document.querySelectorAll('.trow').forEach(el =>",
                    "document.querySelectorAll('.mrow').forEach(el =>\n    el.onclick = () => { const r = document.getElementById('md'+el.dataset.k); r.style.display = r.style.display==='none'?'':'none'; });\n  document.querySelectorAll('.trow').forEach(el =>")


# --- post-inject "Hanford strict" tab (added 2026-10-19, run 2) ---
ma_h = json.load(open(os.path.join(NN, "modern_hanford_analysis.json")))
payload_h = {"labels": ma_h["labels"], "coherence": ma_h["coherence"],
             "period_shares": ma_h["period_shares"], "growth": ma_h["growth"],
             "top_terms": ma_h["top_terms"], "exemplars": ma_h["exemplars"],
             "n_docs": ma_h["n_docs"]}
html = html.replace("['moddoe','Modern era (DOE)']",
                    "['moddoe','Modern era (DOE)'],['hanford','Hanford strict']")
renderer_h = open(os.path.join(WS, "scripts", "nn", "_hanford_renderer.js")).read()
html = html.replace("function render() {", renderer_h + "\nfunction render() {")
html = html.replace("else if (view === 'moddoe') app.innerHTML = tabs() + modernDOE();",
                    "else if (view === 'moddoe') app.innerHTML = tabs() + modernDOE();\n  else if (view === 'hanford') app.innerHTML = tabs() + strictHanford();")
html = html.replace("<script>\n", "<script>\nwindow.__HANFORD__ = " + json.dumps(payload_h, ensure_ascii=False) + ";\n")

out = os.path.join(WS, "ui", "topic_map.html")
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out, "w").write(html)
print("wrote", out, os.path.getsize(out), "bytes")
