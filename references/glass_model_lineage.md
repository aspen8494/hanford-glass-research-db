# Hanford Glass Composition & Algorithm Development — Research Lineage

This is the curated entry point into the database: the document lineages that
define how Hanford glass modeling evolved, with the report numbers to search
for in `db/hanford_glass_research.db`. Everything listed here is in the
database (`SELECT * FROM records JOIN report_numbers USING(record_id)`).

## 1. The big picture

Hanford's 56M gallons of tank waste are being immobilized in borosilicate
glass at two facilities:

- **LAW Facility** (low-activity waste, steam-heated melter, LAW glass with
  ~16-24 wt% Na2O, high SO3/sulfate constraints)
- **HLW Facility** (formerly the "Direct Feed" HLW path, joule-heated melter,
  sludge glasses, high Al2O3/Cr2O3/Fe2O3, noble-metal and Tc issues)

Glass composition research answers: *how much waste can go into each canister
of glass while meeting durability (PCT/VHT), processing (viscosity, liquidus,
electrical conductivity), and regulatory constraints?* The **waste loading**
achieved directly scales the number of canisters, mission duration, and cost —
hence 50 years of model development.

## 2. Property-composition model lineage (the "production" models)

Each generation adds data and extends the valid composition region.

| Model set | Key reports | Notes |
|---|---|---|
| **HWVP era (1994-2001)** | `PNL-10359` (v1+v2, Hrma et al. 1994), `PNNL-10987`, `PNNL-13573` (2001), `PNNL-14060` (2002) | First systematic Hanford HLW glass property models; the 1150 °C melting basis |
| **WTP baseline (2003-2012)** | `PNNL-15164`, Piepel et al. VSL datasets, `PNNL-22631 Rev 1` (2013, includes prior models) | WTP contract models; largely VSL-measured data (VSL-03R3460-1 etc.) |
| **EWG1 (2013-2016)** | `TP-EWG-00001` (test plan), `PNNL-25835` (2016 update), PNNL-26630, PNNL-28838 (LAW data phases) | Enhanced Waste Glass program starts; Na2O to ~24 wt%, SO3 to 1.5 wt% |
| **EWG2 / DFLAW (2019-2023)** | `PNNL-30932 Rev 1/Rev 2` (2021, current LAW WTP operating models; OSTI 1986346/1862823), `PNNL-30932 Rev 1`, PNNL-32866 | LAW Facility startup basis |
| **EWG2.5 / DFHLW (2024-present)** | `PNNL-35884 Rev 1` (2025, EWG-RPT-046), `PNNL-38953` (EWG-RPT-058), PNNL-35037 (high-fluoride), PNNL-38071 (high-alumina, EWG-RPT-045), PNNL-36304 (high-Al expansion) | Current models for the HLW Facility mission; 13-term PCT PQM, VFT viscosity, sulfate MT/TT limits, nepheline discriminator |

Search tip: the EWG program also numbers reports `EWG-RPT-###` — query
`report_numbers LIKE '%EWG-RPT%'`.

## 3. VSL (Vitreous State Laboratory, Catholic University of America)

The bulk of glass *data* (as opposed to models) came from VSL under ORP task
plans: VSL-03R3460-1 (baseline LAW), VSL-05R5900-1 (envelope C), VSL-07R1130-1
(enhanced LAW formulation), VSL-12R2540-1 (sulfate management), VSL-13R2940-1 /
VSL-14R3050-1 (LAW model updates), VSL-19R4480-1 (HLW model enhancement).
OSTI does not index most VSL reports; they enter the database via citing PNNL
reports. The ORP administrative record / MACT equivalency process is the
access path for the raw VSL documents.

## 4. Algorithm development (glass formulation)

- **Early** — "glass formulation strategy" reports (`PNNL-11051`), phase
  1/phase 2 formulation testing (`PNNL-11127`, `PNNL-11013`), frit development
  (`PNNL-11041`), and optimization (`PNNL-10918`, Feng et al. 1996).
- **WTP algorithm** — the Immobilized HLW/LAW formulation algorithm documents
  (`24590-LAW-RPT-RT-04-0003`, 2005; `24590-HLW-RPT-RT-05-001`, 2008,
  Vienna & Kim): the pearl-necklace / subsequent-dilution numerical search
  constrained by property models — the ancestor of every later algorithm.
- **Modern** — per-campaign maximize-waste-loading under property+constraint
  sets (the EWG2.5 approach implemented in the PNNL-35884 model suite);
  inequality-constrained statistical model updates (OSTI 3675531,
  PNNL-SA-220377); ML surrogates and active learning (ANS coverage May 2024 /
  Apr 2026, J. Non-Crystalline Solids).
- **Mission integration** — feed-vector/campaign models: `RPP-RPT-64878`
  (DFHLW feed vectors, WRPS 2024), HTWOS, and the DWPF-style campaign
  blending studies (Britton, WRPS-2300881).

## 5. Special topics (where the constraints come from)

| Topic | What to search |
|---|---|
| Sulfate (SO3) solubility / phase separation | topics: sulfate; "sulfate solubility", MT/TT limit |
| Nepheline (Al2O3 constraint) | topics: nepheline; "nepheline discriminator" |
| Liquidus / spinel / crystallization | topics: liquidus, crystals-spinel |
| Redox (Fe2+/Fe3+) | topics: redox |
| Tc-99 / halide / iodide retention | topics: technetium, halides |
| PCT / VHT durability models | topics: durability |
| Melting rate, cold cap, Joule heating | topics: melter |
| Iron-phosphate alternative matrices | topics: phosphate-glass |
| Off-gas / volatilization (Ru, Tc, I, Cs) | topics: offgas |
| High-alumina glasses | topics: alumina; PNNL-38071, PNNL-36304 |

## 6. Complementary programs (useful context, non-Hanford)

The database tags these `waste-glass-related`: **DWPF** (Savannah River —
the operational precursor; Jantzen's durability models), **West Valley**
(first US HLW vitrification), **INL** (calcine/waste forms), plus the
international programs (France's R7/T7, UK MWVP) via journal literature.

## 7. Starting-point reading order (suggested)

1. `PNNL-22631 Rev 1` — the 2013 "everything before" compendium + constraint rationale
2. `PNNL-30682` — the operating LAW model set with full data lineage
3. `PNNL-35884 Rev 1` — the current DFHLW/EWG2.5 model suite (13-term PCT PQM,
   VFT B-sum viscosity, sulfate MT, K-3 corrosivity, nepheline constraint)
4. `24590-HLW-RPT-RT-05-001` — the formulation *algorithm* that ties models to
   a recipe search
5. `RPP-RPT-64878` — how feed vectors + campaigns consume the glass models
6. Vienna et al., "2016 Update" (`PNNL-25835`) for the EWG program rationale
