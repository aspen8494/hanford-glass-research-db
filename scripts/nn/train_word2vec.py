"""Skip-gram word2vec with negative sampling, from scratch in PyTorch.

Trains on the Hanford glass corpus; saves embeddings + nearest-neighbor maps.

Usage: train_word2vec.py [dim] [epochs]
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NN_DIR = os.path.join(WS, "data", "nn")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    dim = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    window = 6
    ns_rate = 5  # negative samples per positive
    min_count = 8

    docs = json.load(open(os.path.join(NN_DIR, "corpus_tokens.json")))
    sentences = []
    counts = Counter()
    for d in docs:
        sentences.append(d["tokens"])
        counts.update(d["tokens"])
    vocab = [w for w, c in counts.most_common() if c >= min_count]
    w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    freqs = np.array([counts[w] for w in vocab], dtype=np.float64)
    # unigram^0.75 noise distribution
    noise = freqs ** 0.75
    noise /= noise.sum()

    # generate (center, context) pairs doc-by-doc (abstracts are short; window 6)
    pairs_c, pairs_o = [], []
    for s in sentences:
        idx = [w2i[t] for t in s if t in w2i]
        L = len(idx)
        for i, c in enumerate(idx):
            lo, hi = max(0, i - window), min(L, i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    pairs_c.append(c)
                    pairs_o.append(idx[j])
    pairs_c = np.array(pairs_c, dtype=np.int64)
    pairs_o = np.array(pairs_o, dtype=np.int64)
    print(f"vocab={V} pairs={len(pairs_c):,}")

    # subsample frequent words (p_keep = (sqrt(t/f)+t/f), t=1e-3)
    t = 1e-3
    keep_p = np.minimum(1.0, np.sqrt(t / (freqs / freqs.sum())) + t / (freqs / freqs.sum()))
    m_keep = keep_p[pairs_o] >= np.random.rand(len(pairs_o))
    pairs_c, pairs_o = pairs_c[m_keep], pairs_o[m_keep]
    print(f"after subsample: {len(pairs_c):,} pairs")

    c_t = torch.from_numpy(pairs_c)
    o_t = torch.from_numpy(pairs_o)

    class SG(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Embedding(V, dim)
            self.out = nn.Embedding(V, dim)
            nn.init.uniform_(self.inp.weight, -0.5 / dim, 0.5 / dim)
            nn.init.zeros_(self.out.weight)

        def forward(self, c, o, neg):
            vc = self.inp(c)                      # (B, d)
            vo = self.out(o)                      # (B, d)
            vn = self.out(neg)                    # (B, ns, d)
            pos = torch.sum(vc * vo, dim=1)       # (B,)
            neg = torch.bmm(vn, vc.unsqueeze(2)).squeeze(2)  # (B, ns)
            pos_loss = F.logsigmoid(pos).sum()
            neg_loss = F.logsigmoid(-neg).sum()
            return -(pos_loss + neg_loss)

    import torch.nn.functional as F
    model = SG()
    # neg sampling table as tensor for fast sampling
    noise_t = torch.from_numpy(noise.astype(np.float32))
    B = 512  # small batches -> many updates (skip-gram is update-hungry)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: max(0.0001, 1.0 - ep / (epochs * 1.2)))
    n = len(c_t)
    steps_per_epoch = max(1, n // B)
    print_every = max(1, steps_per_epoch // 4)

    for ep in range(epochs):
        perm = torch.randperm(n)
        c_t, o_t = c_t[perm], o_t[perm]
        tot = 0.0
        for s in range(steps_per_epoch):
            c = c_t[s * B:(s + 1) * B]
            o = o_t[s * B:(s + 1) * B]
            if len(c) < 64:
                continue
            # torch.multinomial RETURNS integer indices sampled from noise_t.
            # (Indexing noise_t by them fetches float probabilities; .long()
            # then truncated every negative to token 0 -> rank-1 collapse.)
            neg = torch.multinomial(noise_t, len(c) * ns_rate, replacement=True)
            neg = neg.view(len(c), ns_rate)
            opt.zero_grad()
            loss = model(c, o, neg)
            loss.backward()
            opt.step()
            tot += loss.item()
        scheduler.step()
        print(f"epoch {ep+1:2d}/{epochs}  loss {tot/steps_per_epoch:.2f}  lr {opt.param_groups[0]['lr']:.5f}")

    emb = model.inp.weight.detach().cpu().numpy()
    # L2-normalize rows for cosine NN
    embn = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
    # collapse guard: healthy skip-gram has mean pairwise cos << 0.5 and high
    # effective rank; degenerate training yields mean cos ~0.98, rank ~1
    sims_chk = embn @ embn.T
    np.fill_diagonal(sims_chk, 0)
    mean_cos = float(sims_chk.mean())
    sv = np.linalg.svd(emb - emb.mean(0), compute_uv=False)
    eff_rank = int(np.searchsorted(np.cumsum(sv ** 2) / np.sum(sv ** 2), 0.9) + 1)
    print(f"diagnostics: mean pairwise cos {mean_cos:.3f} | effective rank {eff_rank}")
    if mean_cos > 0.6:
        raise SystemExit("EMBEDDING COLLAPSE detected — check negative sampling")
    np.save(os.path.join(NN_DIR, f"w2v_emb_{dim}.npy"), emb)
    json.dump(vocab, open(os.path.join(NN_DIR, f"w2v_vocab_{dim}.json"), "w"))

    # nearest neighbors preview
    sims = embn @ embn.T
    print("\n--- sample nearest neighbors ---")
    probes = ["nepheline", "durability", "viscosity", "spinel", "technetium",
              "sulfate", "redox", "liquidus", "alumina", "leaching"]
    for p in probes:
        if p not in w2i:
            print(f"{p:14s} (not in vocab)")
            continue
        i = w2i[p]
        nn_idx = np.argsort(-sims[i])[1:7]
        print(f"{p:14s} -> " + ", ".join(vocab[j] for j in nn_idx))


if __name__ == "__main__":
    main()
