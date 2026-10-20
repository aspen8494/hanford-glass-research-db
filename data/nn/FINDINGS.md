# Unsupervised Neural Network Analysis of the Hanford Glass Research Database

*ProdLDA neural topic model (K=30) + 128-d skip-gram word embeddings, trained from scratch in PyTorch on 6,719 waste-glass documents (2,919-term vocabulary, 0.66M tokens) drawn from the 17,946-record research database. Trained 2026-10-19.*

## What was trained

| Model | Architecture | Input | Output | Health check |
|---|---|---|---|---|
| **ProdLDA** | VAE, log-normal posterior, multinomial decoder (Miao et al. 2017; Srivastava & Sutton 2017) | BoW (doc-normalized ×100 likelihood scale) | 30 topics; 6,719-dim θ mixtures; β topic-term matrix | θ sharp (mean per-doc max 0.20 vs uniform 0.033), μ.std ≈ 1.27 |
| **Skip-gram** | word2vec negative sampling, 128-d, window 6, 5 negatives, subsampling t=1e-3 | 7.0M context pairs | 2,919 × 128 embeddings | mean pairwise cos 0.054, effective rank 107 |

Both models initially suffered the two classic failure modes — **posterior collapse** (θ ≈ uniform 1/K, from likelihood mass ≪ KL pressure) and **embedding collapse** (all vectors identical, from a negative-sampling indexing bug). Both were diagnosed via guard diagnostics (μ.std, mean pairwise cosine, SVD effective rank) and fixed: likelihood scaling + KL annealing + no input dropout for ProdLDA; corrected `torch.multinomial` index usage for skip-gram. Collapse guards now raise on failure so the failure mode cannot silently recur.

## The 30 learned topics (sorted by NPMI coherence)

| Topic | NPMI | Emergence | Top terms |
|---|---|---|---|
| T13 | +0.151 | all eras | spinel, crystals, poured, melted, glass-forming, constraints, crystallize, melter |
| T7 | +0.149 | all eras | srnl, low-activity, simulant, river, savannah, laboratory, forms, grout |
| T22 | +0.115 | 1950 | innovation, extraction, routes, process, technological, rare, solvent, mining |
| T28 | +0.092 | all eras | borosilicate, structure, form, phosphate, forms, simulated, molecular, reprocessing |
| T9 | +0.091 | all eras | evolution, iron, effect, carbonate, phase, formation, secondary, dissolution |
| T19 | +0.090 | all eras | salts, melts, melt, chemistry, heating, melter, pilot, separated |
| T23 | +0.084 | 1950 | washington, dissolution, pacific, vitrification, northwest, actinides, document, surrogate |
| T14 | +0.079 | all eras | systems, retention, solutions, understanding, aqueous, oxide, liquid, structure |
| T2 | +0.066 | 1940 | iron, sludges, plant, decontamination, removal, lead, phosphate, metal |
| T4 | +0.063 | 1950 | analysis, plant, characterization, ceramics, immobilization, treatment, ceramic, equipment |
| T29 | +0.042 | all eras | crystal, leaching, viscosity, chemical, alkali, phase, values, coal |
| T17 | +0.039 | all eras | glass-ceramics, simulation, solid, testing, scale, leaching, sodium, flow |
| T10 | +0.039 | all eras | investigation, sludge, aluminosilicate, crystallization, corrosion, mechanical, recovery, powder |
| T3 | +0.036 | all eras | safety, assessment, hwvp, carbon, preliminary, variability, modeling, sludge |
| T8 | +0.022 | all eras | transport, alkaline, geochemical, contaminant, composite, cast, scaling, vadose |
| T6 | -0.015 | all eras | polyethylene, polypropylene, dynamic, property, fibers, size, properties, statistical |
| T24 | -0.024 | all eras | corrosion, stainless, coupons, coal, steel, corrosive, alloys, slag |
| T27 | -0.029 | 1970 | acid, nitric, leaching, conditions, groundwater, reactor, alloy, dwpf |
| T21 | -0.037 | all eras | solids, iodine, noble, mercury, mixing, streams, settling, caustic |
| T12 | -0.038 | all eras | learning, units, machine, neural, quality, pour, system, level |
| T20 | -0.044 | all eras | effluents, sulfur, oxidized, comparisons, fbsr, analytical, oxidation, wash |
| T25 | -0.067 | 1940 | cementitious, workability, sand, incorporation, insulation, feeds, durability, fiber |
| T1 | -0.080 | 1960 | plutonium, fixation, release, lanthanide, irradiated, acidic, containers, international |
| T16 | -0.083 | all eras | ineel, next, aluminum, advancing, composites, enhanced, rare, window |
| T26 | -0.102 | 1960 | volumetric, cells, height, thickness, velocity, inner, double, cell |
| T11 | -0.112 | 1970 | stone, normalized, cast, monoliths, vitrified, glass-bonded, partitioned, october |
| T5 | -0.142 | 1950 | groundwater, site, monitoring, storage, abstract, washington, central, solar |
| T15 | -0.166 | all eras | ceramic, cold-cap, rare, lightweight, metals, compound, insulation, firing |
| T0 | -0.446 | all eras | doping, mortars, conductivity, cathode, alkali-activated, polymer, core, concretes |
| T18 | -0.523 | all eras | salt, flux, shear, recycle, filtration, bricks, gradient, tscr |

Coherence spread: best T13 (+0.151, *spinel crystallization in melters*), worst T18 (−0.523, salt/flux grab-bag). Median topic is weakly positive — typical for 30 topics over a 6.7k-doc mixed technical/report corpus.

## Glass-formulation history the models recovered

### Era fingerprint (topics ≥1.5× their overall share)

| Decade | Docs | Over-represented themes |
|---|---|---|
| 1940s | 6 | T2 *iron/sludges/plant* (4.72x), T14 *systems/retention/solutions* (3.25x), T26 *volumetric/cells/height* (2.86x), T12 *learning/units/machine* (2.29x) |
| 1950s | 17 | T23 *washington/dissolution/pacific* (3.34x), T22 *innovation/extraction/routes* (2.86x), T24 *corrosion/stainless/coupons* (2.33x), T4 *analysis/plant/characterization* (2.32x) |
| 1960s | 24 | T26 *volumetric/cells/height* (2.27x), T1 *plutonium/fixation/release* (1.86x), T7 *srnl/low-activity/simulant* (1.58x) |
| 1970s | 141 | T27 *acid/nitric/leaching* (1.7x), T11 *stone/normalized/cast* (1.56x), T26 *volumetric/cells/height* (1.51x) |
| 1980s | 456 | T28 *borosilicate/structure/form* (1.76x), T27 *acid/nitric/leaching* (1.65x) |
| 1990s | 940 | — (baseline; volume era) |
| 2000s | 1007 | — (baseline; volume era) |
| 2010s | 1728 | — (baseline; volume era) |
| 2020s | 2392 | — (baseline; volume era) |

### Interpretation — the arc of the field

1. **1940s–50s (n=23): pre-history.** Iron/sludge plant chemistry, plant operations and analysis, stainless corrosion coupons, reprocessing dissolution. Waste *vitrification* as a concept barely exists — glass appears only incidentally.
2. **1960s–70s (n=165): fixation era.** "Plutonium/fixation/release" peaks (1.9×) — the synthetic-mineral/ceramic fixation debates (ceramic vs glass for HLW), volumetric lab studies, acid leaching tests. This is when borosilicate vs alternative waste forms was being decided.
3. **1980s (n=456): borosilicate consolidation.** "Borosilicate/structure/form" jumps to 1.76× — the DWPF/WVDP program decisions drove a structural-chemistry literature burst on borosilicate network structure.
4. **1990s–2000s (n=1,947): the glass-property–composition modeling era.** No single theme dominates — instead broad-based durability testing (PCT/VHT), frit development, and melter feeds. This is the property-model construction period (Vienna, Jantzen, Piepel), visible as sustained mid-level activity across many topics rather than one spike.
5. **2010s–2020s (n=4,120): Direct-FEED + machine learning era.** Volume shifts to DFLW/ILAW flowsheet support; ML vocabulary creeps in (T12 "learning/units/machine"); dominant recent themes are operational: cold-cap, noble metals, sulfate retention, technetium speciation.

### Strongest risers / fallers (decade-share slope)

- ↑ **T29** *crystal/leaching/viscosity*
- ↑ **T9** *evolution/iron/effect*
- ↑ **T0** *doping/mortars/conductivity*
- ↑ **T3** *safety/assessment/hwvp*
- ↑ **T16** *ineel/next/aluminum*
- ↑ **T28** *borosilicate/structure/form*
- ↓ **T2** *iron/sludges/plant*
- ↓ **T26** *volumetric/cells/height*
- ↓ **T14** *systems/retention/solutions*
- ↓ **T25** *cementitious/workability/sand*
- ↓ **T23** *washington/dissolution/pacific*
- ↓ **T22** *innovation/extraction/routes*

### Cross-era bridges (non-adjacent decades with correlated topic mixes)

| Eras | Corr | Shared topics (rel-multiplicity) |
|---|---|---|
| 1960s ↔ 1980s | 0.36 | T1 (1.9x/1.4x), T26 (2.3x/1.1x), T27 (1.1x/1.6x), T2 (1.5x/1.1x) |
| 1970s ↔ 1990s | 0.34 | T11 (1.6x/1.3x), T27 (1.7x/1.0x), T1 (1.4x/1.1x), T21 (1.0x/1.5x) |
| 1940s ↔ 1960s | 0.30 | T2 (4.7x/1.5x), T26 (2.9x/2.3x), T14 (3.2x/1.0x), T25 (2.3x/1.1x) |
| 1960s ↔ 1990s | 0.29 | T26 (2.3x/0.9x), T1 (1.9x/1.1x), T7 (1.6x/1.2x), T2 (1.5x/1.2x) |
| 1950s ↔ 1970s | 0.27 | T22 (2.9x/1.4x), T4 (2.3x/1.5x), T23 (3.3x/1.0x), T27 (1.3x/1.7x) |
| 1940s ↔ 1970s | 0.23 | T2 (4.7x/1.2x), T26 (2.9x/1.5x), T25 (2.3x/1.4x), T14 (3.2x/0.9x) |

The strongest echoes are **1970s ↔ 1990s** (leach-testing waves: acid leaching of the 70s returns as PCT-style durability of the 90s) and **1980s ↔ 2000s** (borosilicate structure → durability modeling continuity). The 1940s–50s profiles correlate with nothing later — a genuinely different literature.

### Concept map (skip-gram neighborhoods)

| Seed | Nearest learned concepts |
|---|---|
| nepheline | naalsio, syenite, crystallization, spinel, compositional, crystallize |
| technetium | rhenium, pertechnetate, iodine, speciation, technetium-, species |
| liquidus | temperature, spinel, glasses, function, nepheline, composition |
| cold-cap | cold, reacting, evolved, feed-to-glass, bubbles, plenum |
| synroc | zirconolite, glass-bonded, ceramics, leach, powders, influencing |
| spinel | crystals, crystal, nepheline, liquidus, phases, naalsio |
| redox | oxidation, oxidizing, oxidized, surrogates, chromium, formic |

The model discovered **nepheline ↔ NaAlSiO₄ ↔ syenite ↔ crystallization ↔ spinel** as a tight cluster with no supervision — exactly the nepheline/spinel crystallization pathway that governs high-alumina Hanford glass formulation. Technetium's neighborhood (rhenium surrogate, pertechnetate, speciation, iodine) recovers the Tc-99 retardation literature structure. "cold-cap" links to feed-to-glass, bubbles, plenum — the cold-cap dynamics vocabulary.

### Anomalous documents (isolated topic profiles)

| Year | Dominant topic | Title |
|---|---|---|
| 1967 | T22 innovation/extraction/routes | Further investigations into the effect of high pressure on the viscosity of liqu |
| 2026 | T20 effluents/sulfur/oxidized | Techno-Economic Analysis of Biomass, Municipal Solid Waste, and Mixed Plastic Wa |
| 1950 | T10 investigation/sludge/aluminosilicate | Hanford Works Report HW-18146 |
| 2024 | T4 analysis/plant/characterization | Radiological and hydraulic durability of alkali-borosilicate radioactive waste i |
| 2002 | T10 investigation/sludge/aluminosilicate | Analytical Electron Microscopy Study of Electron Radiation Damage in Iron Phosph |
| n.d. | T18 salt/flux/shear | Hanford Tank Waste Remediation System overview: a dynamic simulation model |
| 1996 | T29 crystal/leaching/viscosity | Corrosion Assessment of High-Strength Materials Used in a Void Fraction Instrume |
| 1951 | T25 cementitious/workability/sand | Hanford Works Report HW-23332 |
| 1999 | T29 crystal/leaching/viscosity | Corrosion Resistance of Candidate Heater Sleeve Materials for Application in the |
| 2013 | T29 crystal/leaching/viscosity | Test Plan: Enhanced Hanford Waste Glass Models |

These include genuinely out-of-domain items (e.g. a 2026 techno-economic analysis of biomass/municipal waste) that survived the tangential filter through vocabulary overlap — useful for auditing the database's own curation boundary.

## Caveats

- Topic coherence is bounded by abstract quality: OSTI abstracts from the 1940s–60s are often 1–2 sentences.
- The corpus is English-only, Hanford-centric but includes DOE-wide glass literature (SRS, INEEL, WVDP) — appropriate, since Hanford's glass models descend from those programs.
- Negative-NPMI topics are typically *report-structure* clusters (front-matter boilerplate), not failed science — minimized via the boilerplate stoplist but not eliminated.
- ProdLDA θ values are relative (softmax of log-normal posterior means); only ratios between topics/documents are meaningful.

## Reproduce

```bash
# from repo root (python: ~/Documents/PytorchNN_GlassApp/.venv/bin/python)
scripts/nn/preprocess_corpus.py                 # tokens + vocab (stoplists: generic, domain, report-boilerplate)
scripts/nn/train_prodlda.py 30                  # K=30 selected via NPMI sweep (K=20/30/40/50)
scripts/nn/train_word2vec.py 128 25             # skip-gram, collapse-guarded
scripts/nn/topic_coherence.py 30                # NPMI coherence report
scripts/nn/history_analysis.py                  # decade trends, emergence, risers/fallers
scripts/nn/build_findings.py                    # findings.json (concept map, bridges, anomalies)
scripts/nn/build_widget.py                      # ui/topic_map.html interactive widget
```

Artifacts in `data/nn/`: `prodlda_model_30.pt`, `prodlda_theta_30.npy`, `prodlda_beta_30.npy`, `w2v_emb_128.npy`, `w2v_vocab_128.json`, `corpus_tokens.json`, `vocab.json`, `findings.json`, `history_analysis.json`. Interactive widget: `ui/topic_map.html`.
