# Modern-Era Deep-Dive (2010s–2020s): Hanford/DOE Waste-Glass Research Themes

*ProdLDA trained on the deduplicated 2010–2026 DOE-signature subset (1,452 docs; 156 near-duplicates removed first from the 3,964-doc modern slice). K=12 selected via NPMI sweep (K=12/16/20/25 → +0.062/+0.034/+0.042/−0.061). Mean NPMI +0.062 — the most coherent model of the entire project. Trained 2026-10-19.*

## Why this rerun was needed

The first modern-era model (K=25 on all 3,964 modern docs, +0.036) mixed **Hanford program literature** with the database's large tail of **world materials-science** papers (recycled glass in concrete, geopolymers, plasma-physics ML surrogates — 70% of the modern slice). Filtering to DOE-signature documents (title-matched: vitrification, melters, LAW/HLW, Hanford/SRS/INL facilities, radioactive waste forms) isolates the themes that matter to Hanford glass formulation.

**K sweep results (DOE subset):** K=12 (+0.062) ≫ K=16 (+0.034) > K=20 (+0.042) > K=25 (−0.061). The 1,452-doc corpus supports ~12 clean themes — fewer, sharper topics than the full-corpus K=30.

## The 12 modern DOE themes (with 2010-14 → 2020-26 share trajectory)

| Topic | NPMI | 2010-14 | 2015-19 | 2020-26 | Trajectory | Theme |
|---|---|---|---|---|---|---|
| T10 property/properties/statistical | **+0.178** | 0.053 | 0.055 | 0.084 | **↑ +58%** | Glass **property–composition modeling & statistical regression** — incl. the component-slope partial-molar-model lineage; ML-adjacent |
| T07 x-ray/phase/simulated | **+0.178** | 0.099 | 0.125 | 0.095 | ↘ peaked 2015-19 | Phase/crystallization characterization (XRD, spinel liquidus) |
| T11 dwpf/nuclear/processing | +0.151 | 0.124 | 0.090 | 0.079 | **↓ −36%** | DWPF/Savannah flowsheet ops: redox models, refractory corrosion |
| T09 borosilicate/nuclear/formation | +0.092 | 0.080 | 0.110 | 0.116 | **↑ +45%** | Borosilicate waste-form chemistry: SON68 alteration, glass-ceramics (zirconolite/powellite), Tc/Re VHT |
| T02 nuclear/high-level/cold | +0.095 | 0.119 | 0.107 | 0.090 | ↓ −24% | HLW vitrification/cold-cap/melter system engineering |
| T03 technetium/off-gas/effect | +0.056 | 0.084 | 0.087 | 0.091 | ≈ flat | Tc/I retention & off-gas in LAW glass |
| T01 forms/radioactive/phosphate | +0.077 | 0.095 | 0.087 | 0.097 | ≈ flat | Alternative waste forms: phosphate, cementitious, ceramic |
| T08 tank/testing/northwest | +0.417* | 0.130 | 0.150 | 0.107 | ↓ (2020 docs skew) | Hanford tank-farm ops/testing (WRPS reports) — highest single-topic coherence but boilerplate-heavy |
| T06 corrosion/concrete/steel | −0.023 | 0.050 | 0.049 | 0.084 | **↑ +68%** | Tank/asset integrity corrosion science (DST nitrate stress-corrosion etc.) |
| T00 composite/contaminants/infrastructure | −0.121 | 0.041 | 0.046 | 0.066 | ↑ +61% | Disposition/secondary-waste infrastructure |
| T05 pilot/joule/simulation | −0.023 | 0.068 | 0.053 | 0.048 | ↓ | Pilot-scale melter testing (DM10/Duramelter) |
| T04 cold/stone/cast | −0.353 | 0.057 | 0.042 | 0.043 | ↓ | CCIM/cold-crucible & cementitious leach testing |

*T08's +0.417 is boilerplate-driven (large homogeneous WRPS report series) — treat its coherence skeptically, as with all high-coherence single-source series.

## What the modern era is actually about (the follow-up answer)

1. **The modeling torch passed to the data.** The property/statistical-modeling theme (T10) is the strongest riser (+58%), and its exemplars show the direct lineage from Hanford's component-slope partial-molar models (2013) to ML/metaheuristic property prediction (2023–2026). The 2010s–20s era is when property–composition modeling itself became the dominant published output — the field's重心 shifted from *building melters* to *fitting models*.
2. **Borosilicate chemistry is still growing** (T09 ↑45%): alteration studies, glass-ceramics for Mo/RE (zirconolite/powellite), Tc/Re vapor-hydration work — international literature engaging with the Hanford/SRS formulation problem.
3. **Melter engineering is in decline** (T02 ↓24%, T05 ↓, T11 ↓36%): cold-cap, pilot melters, DWPF flowsheet work peaked mid-decade (2015-19) and is now a smaller share — consistent with the WTP build-out completing and research pivoting.
4. **Technetium remains the constant** (T03 flat across 16 years): Tc retention in LAW glass — the unsolved Hanford formulation problem — sustains steady publication the entire modern era.
5. **Corrosion/asset-integrity science is rising** (T06 ↑68%, T00 ↑61%): the tanks themselves (DST nitrate SCC, vapor corrosion inhibitors) are drawing research attention as glass formulation matures — the frontier moved from *waste form* to *waste facility*.
6. **The ML signature within DOE work is real but not yet dominant**: T00 of the earlier all-modern model (mortars/neural/accuracy) rose +0.26, but its exemplars were partly external (plasma surrogates, plastic-waste classification). Inside the strict DOE subset, ML appears as method adoption within T10 (property modeling), not as its own theme — i.e., machine learning is being absorbed into glass-property modeling rather than emerging as a separate research program.

## Method notes / caveats

- Near-duplicate detection first (156 dropped from modern slice, e.g. ALTGLASS v3/v4, revised annual reports — kept longest-abstract version). Non-Latin titles never cluster (empty-normalization guard).
- DOE-signature filter is title-regex based; 30% of the unfiltered modern slice matched. A stricter Hanford-only subset (442 docs) is saved at `data/nn/corpus_modern_hanford_strict.json` for future use.
- Period shares are θ means per period; trajectory = 2020-26 share − 2010-14 share. "n=" per period: 375 / 428 / 648.
- Corpus slices: full-corpus K=30 (6,719 docs) for the historical arc; this DOE-modern K=12 for the 2010s+ themes. They answer different questions; both are saved with distinct suffixes.

## Reproduce

```bash
scripts/nn/train_modern.py 25              # modern slice (all 3,964) — K sweep 20/25/30/35/45
scripts/nn/train_modern_doe.py 12         # DOE-signature subset (1,452) — K sweep 12/16/20/25
scripts/nn/topic_coherence_doe.py 12     # NPMI per topic
# artifacts: prodlda_{theta,beta,model}_moddoe12, coherence_moddoe12.json,
#            modern_doe_analysis.json, corpus_modern_doe.json, vocab_modern_doe.json
```
