"""NPMI coherence for the modern DOE-subset ProdLDA.

Usage: topic_coherence_doe.py <K> [topn]   (suffix _moddoe{K})
"""
import json
import os
import sys
from itertools import combinations

import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NN = os.path.join(WS, "data", "nn")


def main():
    K = sys.argv[1]
    topn = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    docs = json.load(open(os.path.join(NN, "corpus_modern_doe.json")))
    vocab = json.load(open(os.path.join(NN, "vocab_modern_doe.json")))
    beta = np.load(os.path.join(NN, f"prodlda_beta_moddoe{K}.npy"))
    doc_sets = [set(d["tokens"]) for d in docs]
    n_docs = len(docs)

    def npmi(w1, w2):
        d1 = sum(1 for s in doc_sets if w1 in s)
        d2 = sum(1 for s in doc_sets if w2 in s)
        d12 = sum(1 for s in doc_sets if w1 in s and w2 in s)
        if d12 == 0:
            return -1.0
        return np.log((d12 / n_docs) / ((d1 / n_docs) * (d2 / n_docs))) / (-np.log(d12 / n_docs))

    n_top = beta.shape[1]
    cohs = []
    for k in range(n_top):
        top = [vocab[i] for i in np.argsort(-beta[:, k])[:topn]]
        cohs.append(float(np.mean([npmi(a, b) for a, b in combinations(top, 2)])))
    cohs = np.array(cohs)
    print(f"K={n_top}: mean NPMI = {cohs.mean():.4f} | median = {np.median(cohs):.4f}")
    order = np.argsort(-cohs)
    for k in order[:8]:
        top = [vocab[i] for i in np.argsort(-beta[:, k])[:8]]
        print(f"  T{k:02d} {cohs[k]:+.3f}  {' '.join(top)}")
    print("worst:")
    for k in order[-4:]:
        top = [vocab[i] for i in np.argsort(-beta[:, k])[:8]]
        print(f"  T{k:02d} {cohs[k]:+.3f}  {' '.join(top)}")
    json.dump({"mean": float(cohs.mean()), "per_topic": cohs.tolist()},
              open(os.path.join(NN, f"coherence_moddoe{K}.json"), "w"))


if __name__ == "__main__":
    main()
