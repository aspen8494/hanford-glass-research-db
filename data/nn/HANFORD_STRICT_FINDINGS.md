# Strict-Hanford Model (2010–2026): The Hanford Tank-Farm Research Conversation

*ProdLDA K=12 on 441 strict-Hanford documents (title-matched: Hanford, WTP, WRPS, tank waste, DST, tank-farm terms; 2010–2026). Vocab: 1,042 terms. Mean NPMI **+0.098 — the most coherent model of the entire project** (vs +0.062 DOE-modern, −0.025 full corpus). K sweep: 6→+0.039, 8→+0.041, 10→+0.088, **12→+0.098**, 16→+0.032, 20→+0.016. Trained 2026-10-19.*

## The 12 themes of strict-Hanford modern research

| Topic | NPMI | 2010-14 → 2020-26 | Theme (with exemplars) |
|---|---|---|---|
| T09 operate/granular/fluidized | +0.453 | 0.064→0.050 ↓ | **FBSR** (fluidized bed steam reforming) — the alternative LAW treatment track |
| T07 pacific/testing/test | +0.267 | 0.098→0.109 ≈ | PNNL test-program reports (boilerplate-heavy; skeptical of +0.267) |
| T10 low-activity/immobilization/plant | +0.209 | 0.106→0.094 ↓ | LAW immobilization & WTP plant product/evaporation |
| T08 iodine/effluent/salt | +0.114 | 0.101→0.079 ↓ | Iodine & salt speciation, effluent treatment facility |
| T02 melter/system/simulants | +0.156 | 0.092→0.089 ≈ | Melter engineering + simulant testing (incl. pretreatment R&D: electrochemical denitration 2025) |
| T06 melter/disposal/vitrification | +0.130 | 0.097→0.065 ↓ | Vitrification product compliance & disposal (LDR regulatory spike testing, delisting) |
| T05 storage/tanks/performance | +0.107 | 0.067→0.108 **↑** | **Tank-farm integrity & performance** — DST vapor corrosion inhibitors, liner corrosion, performance assessment |
| T03 monitoring/technetium/site | +0.085 | 0.092→0.075 ↓ | Technetium management & tank monitoring (Tc inventory/speciation, Ag/AgCl reference electrodes) |
| T00 solids/impacts/sodium | +0.028 | 0.069→0.099 **↑** | **Waste chemistry effects on glass** — halide/chromate/phosphate impacts on LAW glass, Na-aluminosilicate crystallization, aluminate equilibrium, boehmite dissolution |
| T01 tank/analysis/solids | −0.012 | 0.091→0.112 **↑** | Tank waste characterization/evaluation (feed to WTP) |
| T04 findings/slow/technique | −0.233 | 0.058→0.058 ≈ | Analytical-technique grab-bag (STEM of Pu particles, passive films, dawsonite) — mixed coherence |
| T11 columbia/along/response | −0.127 | 0.065→0.063 ≈ | Hanford site/history/policy (incl. organizational history books) |

## Reading: what the strict-Hanford conversation looks like

**1. The tank farm, not the melter, is where modern Hanford research is going.** The two biggest risers are tank-farm storage integrity (T05 ↑61%: DST corrosion mitigation, liner performance) and tank-waste characterization (T01 ↑23%). The melter/vitrification-product themes (T06 ↓33%, T10 ↓11%) are shrinking shares. The story matches the mission timeline: WTP's vitrification technology matured, and the binding constraint moved upstream to *what's in the tanks* (feed characterization, corrosion limits on how long waste can stay there) and downstream to *product compliance*.

**2. Glass chemistry survives as the "impacts" theme.** T00 — halide/chromate/phosphate impacts on LAW glass, sodium-aluminosilicate (nepheline-family) crystallization, aluminate/boehmite chemistry — is the direct continuation of the glass-formulation modeling lineage, and it's growing (↑44%). This is where waste composition variability meets glass acceptance constraints.

**3. Technetium is the constant here too** (T03: Tc inventory, distribution, speciation in Hanford tanks; supernatant availability studies) — echoing the DOE-modern model's finding of 16 years of flat, sustained Tc work.

**4. FBSR shows as the coherent alternative-track topic** (T09 +0.453, highest coherence): fluidized-bed steam reforming of LAW — the "don't vitrify everything" research thread, running 2012–2023 alongside the glass program.

**5. The waste-chemistry frontier documents:** boehmite/trace-metal incorporation (2024), aluminate equilibrium in specific tanks (AN-105, AP-10x), Li/Fe/B effects on Na-aluminosilicate crystallization (2016) — these exemplars read like the input side of the property-composition models from the DOE-modern T10. The two models are complementary halves of one story: **feed chemistry rising in importance, glass property modeling rising in importance, melter engineering declining.**

## Caveats

- 441 docs is small; per-topic shares are noisier (n≈127/151/164 per period).
- T07 (+0.267) and T09 (+0.453) coherence is inflated by homogeneous test-program report series (the same lesson as WRPS T08 in the DOE model).
- Strict filter is title-regex; some Hanford-relevant docs with generic titles (e.g. "Component Slope Linear Model…") fall outside it — that's why the DOE-modern model remains the reference for property-modeling lineage.
- T04's −0.233 comes from pairing an analytical-techniques cluster with a history book ("When the Whistle Didn't Blow") — noise, not science.

## Artifacts & reproduce

```bash
scripts/nn/train_modern_hanford.py 12      # K sweep: 6/8/10/12/16/20 -> 12 wins
scripts/nn/topic_coherence_hanford.py 12
# artifacts: prodlda_{theta,beta,model}_modhanford12, coherence_modhanford12.json,
#            modern_hanford_analysis.json, corpus_modhanford.json, vocab_modhanford.json
```
