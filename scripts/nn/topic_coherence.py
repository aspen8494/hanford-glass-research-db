"""NPMI topic coherence for ProdLDA topic-word matrices.

Higher = topics whose top words co-occur in documents more than chance.
Usage: topic_coherence.py <suffix> [topn]
"""
import json
import os
import sys
from collections import Counter
from itertools import combinations

import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NN_DIR = os.path.join(WS, "data", "nn")


def npmi_score(w1, w2, doc_sets, n_docs):
    """NPMI(w1, w2) over the corpus: PMI normalized to [-1, 1]."""
    d1 = sum(1 for s in doc_sets if w1 in s)
    d2 = sum(1 for s in doc_sets if w2 in s)
    d12 = sum(1 for s in doc_sets if w1 in s and w2 in s)
    if d12 == 0:
        return -1.0
    pmi = np.log((d12 / n_docs) / ((d1 / n_docs) * (d2 / n_docs)))
    return pmi / (-np.log(d12 / n_docs))


def main():
    suffix = sys.argv[1]
    topn = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    docs = json.load(open(os.path.join(NN_DIR, "corpus_tokens.json")))
    vocab = json.load(open(os.path.join(NN_DIR, "vocab.json")))
    beta = np.load(os.path.join(NN_DIR, f"prodlda_beta_{suffix}.npy"))
    doc_sets = [set(d["tokens"]) for d in docs]
    n_docs = len(docs)

    K = beta.shape[1]
    cohs = []
    for k in range(K):
        top = [vocab[i] for i in np.argsort(-beta[:, k])[:topn]]
        vals = [npmi_score(a, b, doc_sets, n_docs) for a, b in combinations(top, 2)]
        cohs.append(float(np.mean(vals)))
    cohs = np.array(cohs)
    print(f"K={K}: mean NPMI = {cohs.mean():.4f} | median = {np.median(cohs):.4f}")
    order = np.argsort(-cohs)
    print("\nbest topics:")
    for k in order[:8]:
        top = [vocab[i] for i in np.argsort(-beta[:, k])[:8]]
        print(f"  T{k:02d} {cohs[k]:+.3f}  {' '.join(top)}")
    print("worst topics:")
    for k in order[-5:]:
        top = [vocab[i] for i in np.argsort(-beta[:, k])[:8]]
        print(f"  T{k:02d} {cohs[k]:+.3f}  {' '.join(top)}")
    json.dump({"mean": float(cohs.mean()), "per_topic": cohs.tolist()},
              open(os.path.join(NN_DIR, f"coherence_{suffix}.json"), "w"))


if __name__ == "__main__":
    main()
