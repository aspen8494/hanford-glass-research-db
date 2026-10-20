"""ProdLDA on the strict-Hanford modern subset (2010-2026 Hanford-tank-farm docs).

Reads corpus_modern_hanford_strict.json, builds subset vocab (df 0.5%-70%,
min 8), trains ProdLDA with K sweep support, saves artifacts with suffix
_modhanford{K}.

Usage: train_modern_hanford.py [n_topics]
"""
import json
import os
import sys
from collections import Counter

import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NN = os.path.join(WS, "data", "nn")
SEED = 42


def main():
    n_topics = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    docs = json.load(open(os.path.join(NN, "corpus_modern_hanford_strict.json")))

    # subset vocab
    df = Counter()
    for d in docs:
        df.update(set(d["tokens"]))
    n = len(docs)
    vocab = sorted(w for w, c in df.items()
                   if c >= max(8, 0.005 * n) and c <= 0.70 * n)
    vset = set(vocab)
    for d in docs:
        d["tokens"] = [t for t in d["tokens"] if t in vset]
    kept = [d for d in docs if len(d["tokens"]) >= 8]
    print(f"strict-Hanford corpus: {len(kept)} docs | vocab {len(vocab)}")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    X = np.zeros((len(kept), V), dtype=np.float32)
    for i, d in enumerate(kept):
        for t in d["tokens"]:
            X[i, w2i[t]] += 1.0
    Xn = X / np.clip(X.sum(axis=1, keepdims=True), 1e-9, None)
    Xt = torch.tensor(Xn * 100.0)
    dl = DataLoader(TensorDataset(Xt), batch_size=64, shuffle=True)

    K, hidden = n_topics, 256
    enc1 = nn.Linear(V, hidden)
    enc_mu = nn.Linear(hidden, K)
    enc_sig = nn.Linear(hidden, K)
    dec = nn.Linear(K, V, bias=False)
    nn.init.uniform_(dec.weight, -0.1, 0.1)
    params = (list(enc1.parameters()) + list(enc_mu.parameters())
              + list(enc_sig.parameters()) + list(dec.parameters()))
    opt = torch.optim.Adam(params, lr=0.001, betas=(0.99, 0.999))

    def kl_ln(mu, lnsig):
        sig2 = (2 * lnsig).exp()
        return (0.5 * (sig2 + (mu - 1.0) ** 2) - 0.5) - lnsig

    epochs = 150
    for ep in range(epochs):
        beta_kl = min(1.0, ep / 30)
        tot = 0.0
        for (xb,) in dl:
            opt.zero_grad()
            h = F.softplus(enc1(xb))
            mu, lnsig = enc_mu(h), enc_sig(h)
            z = mu + torch.randn_like(mu) * lnsig.exp()
            recon = F.softmax(dec(z), dim=1)
            rec = (xb * torch.log(recon + 1e-10)).sum(dim=1)
            kl = kl_ln(mu, lnsig).sum(dim=1)
            loss = -(rec - beta_kl * kl).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tot += loss.item() * len(xb)
        if ep % 30 == 0 or ep == epochs - 1:
            print(f"epoch {ep:3d}  loss/doc {tot/len(kept):.1f}  beta_kl {beta_kl:.2f}")

    with torch.no_grad():
        mu_all = []
        for i in range(0, len(kept), 256):
            h = F.softplus(enc1(Xt[i:i + 256]))
            mu_all.append(enc_mu(h))
        mu_all = torch.cat(mu_all)
        theta = torch.softmax(mu_all, dim=1).numpy()
    std = float(mu_all.std())
    print(f"posterior: mu.std {std:.3f} | per-doc max {theta.max(axis=1).mean():.3f}")
    if std < 0.05:
        raise SystemExit("POSTERIOR COLLAPSE")

    beta = dec.weight.detach().numpy()
    beta_p = np.exp(beta - beta.max(axis=0, keepdims=True))
    beta_p /= beta_p.sum(axis=0, keepdims=True)

    np.save(os.path.join(NN, f"prodlda_theta_modhanford{n_topics}.npy"), theta)
    np.save(os.path.join(NN, f"prodlda_beta_modhanford{n_topics}.npy"), beta_p)
    torch.save({"enc1": enc1.state_dict(), "enc_mu": enc_mu.state_dict(),
                "enc_sig": enc_sig.state_dict(), "dec": dec.state_dict()},
               os.path.join(NN, f"prodlda_model_modhanford{n_topics}.pt"))
    json.dump({"n_docs": len(kept), "vocab_size": V, "n_topics": K,
               "epochs": epochs, "seed": SEED},
              open(os.path.join(NN, f"prodlda_meta_modhanford{n_topics}.json"), "w"),
              indent=1)
    json.dump(kept, open(os.path.join(NN, "corpus_modhanford.json"), "w"))
    json.dump(vocab, open(os.path.join(NN, "vocab_modhanford.json"), "w"))

    va = np.array(vocab)
    years = np.array([d.get("year") or 0 for d in kept])
    periods = [("2010-14", 2010, 2015), ("2015-19", 2015, 2020), ("2020-26", 2020, 2027)]
    print("\n=== TOPICS ===")
    for k in range(K):
        top = va[np.argsort(-beta_p[:, k])[:10]]
        line = f"T{k:02d} "
        for name, lo, hi in periods:
            m = (years >= lo) & (years < hi)
            line += f"[{name}:{theta[m, k].mean():.3f}] " if m.sum() else f"[{name}:—] "
        print(line + " ".join(top))
    print(f"\nsaved (suffix _modhanford{n_topics})")


if __name__ == "__main__":
    main()
