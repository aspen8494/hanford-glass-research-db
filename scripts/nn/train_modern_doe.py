"""ProdLDA on the modern DOE-signature subset (2010s-2020s Hanford/DOE waste-glass docs).

Reads corpus_modern_doe.json + vocab_modern_doe.json (built by the caller),
trains ProdLDA, saves artifacts with suffix _moddoe{K}, prints topics + a
period-share table (2010-14 / 2015-19 / 2020-26).

Usage: train_modern_doe.py [n_topics]
"""
import json
import os
import sys

import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NN = os.path.join(WS, "data", "nn")
SEED = 42


def main():
    n_topics = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    docs = json.load(open(os.path.join(NN, "corpus_modern_doe.json")))
    vocab = json.load(open(os.path.join(NN, "vocab_modern_doe.json")))
    print(f"DOE modern corpus: {len(docs)} docs, vocab {len(vocab)}")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    X = np.zeros((len(docs), V), dtype=np.float32)
    for i, d in enumerate(docs):
        for t in d["tokens"]:
            X[i, w2i[t]] += 1.0
    Xn = X / np.clip(X.sum(axis=1, keepdims=True), 1e-9, None)
    Xt = torch.tensor(Xn * 100.0)
    dl = DataLoader(TensorDataset(Xt), batch_size=128, shuffle=True)

    K, hidden = n_topics, 384
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

    epochs = 100
    for ep in range(epochs):
        beta_kl = min(1.0, ep / 25)
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
        if ep % 25 == 0 or ep == epochs - 1:
            print(f"epoch {ep:3d}  loss/doc {tot/len(docs):.1f}")

    with torch.no_grad():
        mu_all = []
        for i in range(0, len(docs), 512):
            h = F.softplus(enc1(Xt[i:i + 512]))
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

    np.save(os.path.join(NN, f"prodlda_theta_moddoe{n_topics}.npy"), theta)
    np.save(os.path.join(NN, f"prodlda_beta_moddoe{n_topics}.npy"), beta_p)
    torch.save({"enc1": enc1.state_dict(), "enc_mu": enc_mu.state_dict(),
                "enc_sig": enc_sig.state_dict(), "dec": dec.state_dict()},
               os.path.join(NN, f"prodlda_model_moddoe{n_topics}.pt"))
    json.dump({"n_docs": len(docs), "vocab_size": V, "n_topics": K,
               "epochs": epochs, "seed": SEED},
              open(os.path.join(NN, f"prodlda_meta_moddoe{n_topics}.json"), "w"), indent=1)

    # period shares
    years = np.array([d.get("year") or 0 for d in docs])
    periods = [("2010-14", 2010, 2015), ("2015-19", 2015, 2020), ("2020-26", 2020, 2027)]
    va = np.array(vocab)
    print("\n=== TOPICS ===")
    for k in range(K):
        top = va[np.argsort(-beta_p[:, k])[:10]]
        shares = []
        for name, lo, hi in periods:
            m = (years >= lo) & (years < hi)
            shares.append(f"{theta[m, k].mean():.3f}" if m.sum() else "—")
        print(f"T{k:02d} [{shares[0]} {shares[1]} {shares[2]}] {' '.join(top)}")
    print(f"\nsaved (suffix _moddoe{n_topics})")


if __name__ == "__main__":
    main()
