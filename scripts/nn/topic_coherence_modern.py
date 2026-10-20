"""NPMI coherence for the modern-era ProdLDA topic-word matrices.

Usage: topic_coherence_modern.py <suffix> [topn]
  e.g. topic_coherence_modern.py mod45
"""
import json
import os
import sys
from itertools import combinations

import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NN = os.path.join(WS, "data", "nn")


def main():
    suffix = sys.argv[1]
    topn = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    docs = json.load(open(os.path.join(NN, "corpus_modern.json")))
    vocab = json.load(open(os.path.join(NN, "vocab_modern.json")))
    beta = np.load(os.path.join(NN, f"prodlda_beta_{suffix}.npy"))
    doc_sets = [set(d["tokens"]) for d in docs]
    n_docs = len(docs)

    def npmi(w1, w2):
        i1, i2 = vocab.index(w1), vocab.index(w2)
        d1 = sum(1 for s in doc_sets if w1 in s)
        d2 = sum(1 for s in doc_sets if w2 in s)
        d12 = sum(1 for s in doc_sets if w1 in s and w2 in s)
        if d12 == 0:
            return -1.0
        pmi = np.log((d12 / n_docs) / ((d1 / n_docs) * (d2 / n_docs)))
        return pmi / (-np.log(d12 / n_docs))

    K = beta.shape[1]
    cohs = []
    for k in range(K):
        top = [vocab[i] for i in np.argsort(-beta[:, k])[:topn]]
        vals = [npmi(a, b) for a, b in combinations(top, 2)]
        cohs.append(float(np.mean(vals)))
    cohs = np.array(cohs)
    print(f"K={K}: mean NPMI = {cohs.mean():.4f} | median = {np.median(cohs):.4f}")
    order = np.argsort(-cohs)
    print("\nbest topics:")
    for k in order[:10]:
        top = [vocab[i] for i in np.argsort(-beta[:, k])[:8]]
        print(f"  T{k:02d} {cohs[k]:+.3f}  {' '.join(top)}")
    print("worst topics:")
    for k in order[-6:]:
        top = [vocab[i] for i in np.argsort(-beta[:, k])[:8]]
        print(f"  T{k:02d} {cohs[k]:+.3f}  {' '.join(top)}")
    json.dump({"mean": float(cohs.mean()), "per_topic": cohs.tolist()},
              open(os.path.join(NN, f"coherence_{suffix}.json"), "w"))


if __name__ == "__main__":
    main()
