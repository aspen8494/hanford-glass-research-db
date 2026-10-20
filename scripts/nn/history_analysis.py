"""History + findings analysis over the trained ProdLDA topics.

Reads prodlda artifacts (theta, beta, corpus) and produces:
  - topic trends per era/decade (which themes rose and fell)
  - topic emergence years (first sustained activity)
  - cross-era topic continuity (topic drift)
  - document exemplars per topic
  - per-topic inter-topic similarity (concept-neighborhood)
Saves data/nn/history_analysis.json and prints a readable digest.
"""
import json
import os
from collections import defaultdict

import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NN_DIR = os.path.join(WS, "data", "nn")
SUFFIX = "30"  # coherence-selected model


def main():
    docs = json.load(open(os.path.join(NN_DIR, "corpus_tokens.json")))
    vocab = json.load(open(os.path.join(NN_DIR, "vocab.json")))
    theta = np.load(os.path.join(NN_DIR, f"prodlda_theta_{SUFFIX}.npy"))
    beta = np.load(os.path.join(NN_DIR, f"prodlda_beta_{SUFFIX}.npy"))
    coh = json.load(open(os.path.join(NN_DIR, f"coherence_{SUFFIX}.json")))["per_topic"]
    K = theta.shape[1]
    years = np.array([d["year"] if d.get("year") else np.nan for d in docs])

    # short labels from top-3 distinctive words
    labels = []
    for k in range(K):
        top = [vocab[i] for i in np.argsort(-beta[:, k])[:3]]
        labels.append("/".join(top))

    # ---- topic share per decade ----
    decades = sorted({int(y // 10 * 10) for y in years if not np.isnan(y)})
    shares = {}
    for dec in decades:
        m = (years >= dec) & (years < dec + 10)
        if m.sum() < 5:
            continue
        shares[str(dec)] = theta[m].mean(axis=0).tolist()
    n_docs_dec = {str(dec): int(((years >= dec) & (years < dec + 10)).sum())
                  for dec in decades
                  if ((years >= dec) & (years < dec + 10)).sum() >= 5}

    # ---- emergence: first decade where topic share >= 1.5x its overall mean
    # and stays >= overall mean for the next 2 consecutive active decades
    overall = theta.mean(axis=0)
    active_decs = list(shares.keys())
    emergence = {}
    for k in range(K):
        for i, dec in enumerate(active_decs):
            if shares[dec][k] >= 1.5 * overall[k]:
                rest = [shares[d][k] for d in active_decs[i:i + 3]]
                if len(rest) >= 2 and all(v >= overall[k] * 0.9 for v in rest):
                    emergence[k] = dec
                    break

    # ---- strongest risers/fallers: share slope over the active decades
    slopes = {}
    x = np.arange(len(active_decs))
    for k in range(K):
        y = np.array([shares[d][k] for d in active_decs])
        # normalize by total docs per decade to avoid volume bias
        y_norm = np.array([shares[d][k] * n_docs_dec[d] for d in active_decs])
        slopes[k] = float(np.polyfit(x, y, 1)[0])

    # ---- top docs per topic (highest theta, with year+title)
    exemplars = {}
    for k in range(K):
        idx = np.argsort(-theta[:, k])[:5]
        exemplars[k] = [{
            "title": docs[i]["title"][:110],
            "year": docs[i].get("year"),
            "relevance": docs[i].get("relevance"),
            "theta": round(float(theta[i, k]), 3),
        } for i in idx]

    # ---- topic-topic similarity (cosine over beta columns)
    bn = beta / np.clip(np.linalg.norm(beta, axis=0, keepdims=True), 1e-9, None)
    sim = bn.T @ bn
    np.fill_diagonal(sim, 0)
    neighbors = {}
    for k in range(K):
        j = np.argsort(-sim[k])[:3]
        neighbors[k] = [{"topic": int(t), "label": labels[int(t)],
                         "sim": round(float(sim[k, t]), 3)} for t in j]

    out = {
        "suffix": SUFFIX,
        "n_docs": len(docs),
        "labels": labels,
        "coherence": coh,
        "overall_share": overall.tolist(),
        "decade_shares": shares,
        "docs_per_decade": n_docs_dec,
        "emergence": {str(k): emergence.get(k) for k in range(K)},
        "slopes": {str(k): round(slopes[k], 5) for k in range(K)},
        "exemplars": {str(k): exemplars[k] for k in range(K)},
        "topic_neighbors": {str(k): neighbors[k] for k in range(K)},
    }
    with open(os.path.join(NN_DIR, "history_analysis.json"), "w") as f:
        json.dump(out, f, indent=1)

    # ---- readable digest ----
    print("=== TOPIC LABELS (K=30, by coherence) ===")
    for k in sorted(range(K), key=lambda k: -coh[k]):
        em = emergence.get(k, "")
        print(f"T{k:02d} {coh[k]:+.3f} [{em or 'always':>4}] {labels[k]}")

    print("\n=== DECADE x TOPIC (top-3 risers per decade vs overall) ===")
    for dec in active_decs:
        s = np.array(shares[dec])
        ratio = s / np.clip(overall, 1e-9, None)
        top = np.argsort(-ratio)[:3]
        fmt = ", ".join(f"T{k}({ratio[k]:.1f}x)" for k in top)
        print(f"{dec}s (n={n_docs_dec[dec]:4d}): {fmt}")

    print("\n=== STRONGEST RISERS / FALLERS (share slope) ===")
    order = sorted(range(K), key=lambda k: -slopes[k])
    for k in order[:6]:
        print(f"  RISING  T{k:02d} {labels[k]}")
    for k in order[-6:]:
        print(f"  FALLING T{k:02d} {labels[k]}")


if __name__ == "__main__":
    main()
