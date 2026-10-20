
function modernDOE() {
  const M = window.__MODDOE__;
  const periods = ['2010-14','2015-19','2020-26'];
  let h = `<h2>Modern era (2010-2026), DOE-signature subset</h2>
  <p class="sub">ProdLDA K=12 on ${M.n_docs} deduplicated DOE waste-glass documents.
  Mean NPMI +0.062 (best model of the project). Bars = share per period (1.0 = all docs).</p>
  <table><tr><th>topic</th>${periods.map(p=>`<th>${p}</th>`).join('')}<th>trajectory</th></tr>`;
  const ks = Object.keys(M.labels).map(Number).sort((a,b)=>M.coherence[b]-M.coherence[a]);
  for (const k of ks) {
    const s = M.period_shares[k];
    const g = M.growth[k];
    const maxs = Math.max(...periods.map(p=>s[p]||0), 0.001);
    const cells = periods.map(p => {
      const v = s[p]||0;
      return `<td class="cell"><span class="bar" style="width:${Math.round(24*v/maxs)}px"></span>${v.toFixed(3)}</td>`;
    }).join('');
    const arrow = g==null?'' :(g>0.01?`<span style="color:#30a46c">&#9650; +${Math.round(100*g/(s[periods[0]]||1e-9))}%</span>`:(g<-0.01?`<span style="color:#d05454">&#9660; ${Math.round(100*g/(s[periods[0]]||1e-9))}%</span>`:'&#8776;'));
    h += `<tr class="mrow" data-k="${k}"><td><b>T${k}</b> <span class="muted">${esc(M.labels[k]||'')}</span></td>${cells}<td>${arrow}</td></tr>
    <tr class="mdetail" id="md${k}" style="display:none"><td colspan="6" style="padding:0 12px 10px">
      <div class="muted">${esc((M.top_terms[k]||[]).join(', '))}</div>
      ${(M.exemplars[k]||[]).map(e=>`<div class="muted">[${e.year??'n.d.'}] ${esc(e.title)}</div>`).join('')}
    </td></tr>`;
  }
  return h + '</table>';
}
