
import json, os
import numpy as np

NN = "data/nn"
docs = json.load(open(f"{NN}/corpus_tokens.json"))
vocab = json.load(open(f"{NN}/vocab.json"))
theta = np.load(f"{NN}/prodlda_theta_30.npy")
beta = np.load(f"{NN}/prodlda_beta_30.npy")
coh = json.load(open(f"{NN}/coherence_30.json"))["per_topic"]
hist = json.load(open(f"{NN}/history_analysis.json"))
emb = np.load(f"{NN}/w2v_emb_128.npy")
embn = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
w2v_vocab = json.load(open(f"{NN}/w2v_vocab_128.json"))
w2i = {w: i for i, w in enumerate(w2v_vocab)}

# 1) concept map: for each of 60 seed concepts (domain-significant), 8 NNs
SEEDS = ["nepheline", "spinel", "liquidus", "viscosity", "durability", "borosilicate",
         "technetium", "sulfate", "redox", "chromium", "alumina", "silica", "boron",
         "sodium", "phosphate", "iron", "zirconolite", "sodalite", "zeolite", "ion-exchange",
         "melting", "crystallization", "phase", "segregation", "noble", "molybdenum",
         "volatility", "leaching", "corrosion", "pct", "vapor-hydration", "alteration",
         "glass-ceramics", "glass-bonded", "synroc", "geopolymer", "cementitious", "grout",
         "cold-cap", "joule-heated", "ccim", "bubblers", "off-gas", "feed", "rheology",
         "electrode", "foaming", "sucrose", "nitrate", "nitrite", "simulant", "waste-loading",
         "canister", "repository", "ywreck", "ilaw", "hlw", "dflaw", "ramese", "env"]
concept_map = {}
for s in SEEDS:
    if s not in w2i:
        continue
    i = w2i[s]
    sims = embn @ embn[i]
    idx = np.argsort(-sims)[1:9]
    concept_map[s] = [(w2v_vocab[j], round(float(sims[j]), 3)) for j in idx]

# 2) cross-era bridges: decade centroid in theta space -> topic profile per decade
years = np.array([d["year"] if d.get("year") else -1 for d in docs])
decades = {}
for dec in (1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020):
    m = (years >= dec) & (years < dec + 10)
    if m.sum() >= 5:
        decades[str(dec)] = theta[m].mean(axis=0)

# bridge = topics strongly shared between non-adjacent eras (e.g. 1980s & 2020s)
era_pairs = {}
dl = sorted(decades)
for a_i in range(len(dl)):
    for b_i in range(a_i + 2, len(dl)):
        a, b = decades[dl[a_i]], decades[dl[b_i]]
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        shared = np.argsort(-(a * b))[:4]
        era_pairs[f"{dl[a_i]}-{dl[b_i]}"] = {
            "cosine": round(cos, 3),
            "shared_topics": [(int(t), hist["labels"][int(t)]) for t in shared],
        }

# 3) anomalies: docs whose dominant topic has low overall share (rare-theme docs)
overall = theta.mean(axis=0)
anomalies = []
for i in range(len(docs)):
    k = int(np.argmax(theta[i]))
    if overall[k] < 0.02 and theta[i, k] > 0.35 and len(anomalies) < 25:
        anomalies.append({"title": docs[i]["title"][:100], "year": docs[i].get("year"),
                          "topic": k, "label": hist["labels"][k], "theta": round(float(theta[i, k]), 3)})

# 4) per-topic exemplars (already in hist) + top terms with weights
topics_detail = {}
for k in range(30):
    top = np.argsort(-beta[:, k])[:12]
    topics_detail[k] = {"label": hist["labels"][k], "coherence": round(coh[k], 3),
                        "top_terms": [(vocab[i], round(float(beta[i, k]), 5)) for i in top],
                        "exemplars": hist["exemplars"][str(k)],
                        "emergence": hist["emergence"].get(str(k)),
                        "neighbors": hist["topic_neighbors"][str(k)]}

out = {"concept_map": concept_map, "era_bridges": era_pairs,
       "anomalies": anomalies, "topics_detail": topics_detail,
       "decade_shares": hist["decade_shares"], "docs_per_decade": hist["docs_per_decade"],
       "slopes": hist["slopes"], "labels": hist["labels"], "coherence": coh}
json.dump(out, open(f"{NN}/findings.json", "w"), indent=1)
print("findings.json written:", os.path.getsize(f"{NN}/findings.json"), "bytes")
print("concept map seeds found:", len(concept_map), "/", len(SEEDS))
print("era pairs:", len(era_pairs))
print("anomalies:", len(anomalies))
# peek: strongest era bridges
for k, v in sorted(era_pairs.items(), key=lambda kv: -kv[1]["cosine"])[:5]:
    print(f"bridge {k}: cos {v['cosine']} via {[t[0] for t in v['shared_topics']]}")
