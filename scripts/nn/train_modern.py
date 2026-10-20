"""Modern-era (2010s-2020s) ProdLDA pipeline.

Slice: year >= 2010 (4,120 docs). Dedupes near-identical titles (keep the
longest-abstract version), trains ProdLDA K=45 with modern-era vocabulary,
reports NPMI coherence, saves theta/beta/meta with suffix _mod45.

Usage: train_modern.py [n_topics]
"""
import json
import os
import re
import sys
from collections import Counter
from difflib import SequenceMatcher

import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NN = os.path.join(WS, "data", "nn")
SEED = 42


def norm_title(t):
    return re.sub(r"[^a-z0-9 ]", "", (t or "").lower()).strip()


def dedupe(docs):
    """Cluster docs with near-identical normalized titles; keep the doc with the
    longest token list (richest abstract). Docs with empty normalized titles
    (non-Latin scripts) are never clustered together — norm '' is skipped."""
    srt = sorted(docs, key=lambda d: norm_title(d["title"]))
    kept, dropped = [], 0
    i = 0
    while i < len(srt):
        base = norm_title(srt[i]["title"])
        j = i + 1
        if base:
            # extend cluster: fuzzy match against the cluster's FIRST doc
            while (j < len(srt)
                   and norm_title(srt[j]["title"])
                   and SequenceMatcher(None, base, norm_title(srt[j]["title"])).ratio() >= 0.92):
                j += 1
            cluster = srt[i:j]
            best = max(cluster, key=lambda d: len(d["tokens"]))
            kept.append(best)
            dropped += len(cluster) - 1
            i = j
        else:
            kept.append(srt[i])
            i += 1
    return kept, dropped


def main():
    n_topics = int(sys.argv[1]) if len(sys.argv) > 1 else 45
    docs = json.load(open(os.path.join(NN, "corpus_tokens.json")))
    modern = [d for d in docs if (d.get("year") or 0) >= 2010]
    modern, dropped = dedupe(modern)
    print(f"modern slice: {len(modern)} docs (dropped {dropped} near-duplicates)")

    # --- era-specific vocab: df in [0.3%, 60%] of docs, min 10 occurrences ---
    df = Counter()
    tf = Counter()
    for d in modern:
        df.update(set(d["tokens"]))
        tf.update(d["tokens"])
    n = len(modern)
    vocab = sorted(w for w, c in df.items() if c >= max(10, 0.003 * n) and c <= 0.60 * n)
    vset = set(vocab)
    for d in modern:
        d["tokens"] = [t for t in d["tokens"] if t in vset]
    kept = [d for d in modern if len(d["tokens"]) >= 8]
    print(f"vocab: {len(vocab)} | docs kept: {len(kept)}")

    # --- save the modern corpus for reuse ---
    json.dump(kept, open(os.path.join(NN, "corpus_modern.json"), "w"))
    json.dump(vocab, open(os.path.join(NN, "vocab_modern.json"), "w"))

    # --- BoW ---
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
    dl = DataLoader(TensorDataset(Xt), batch_size=256, shuffle=True)

    K, hidden = n_topics, 512
    enc1 = nn.Linear(V, hidden)
    enc_mu = nn.Linear(hidden, K)
    enc_sig = nn.Linear(hidden, K)
    dec = nn.Linear(K, V, bias=False)
    nn.init.uniform_(dec.weight, -0.1, 0.1)
    params = list(enc1.parameters()) + list(enc_mu.parameters()) + list(enc_sig.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=0.001, betas=(0.99, 0.999))

    def kl_ln(mu, lnsig):
        sig2 = (2 * lnsig).exp()
        return (0.5 * (sig2 + (mu - 1.0) ** 2) - 0.5) - lnsig

    epochs = 80
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
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"epoch {ep:3d}  loss/doc {tot/len(kept):.1f}  beta_kl {beta_kl:.2f}")

    # ---- extract ----
    with torch.no_grad():
        mu_all = []
        for i in range(0, len(kept), 512):
            h = F.softplus(enc1(Xt[i:i + 512]))
            mu_all.append(enc_mu(h))
        mu_all = torch.cat(mu_all)
        theta = torch.softmax(mu_all, dim=1).numpy()
        assert theta.shape[0] == len(kept)
    print(f"posterior: mu.std {float(mu_all.std()):.3f} "
          f"(healthy ~1+; collapsed <0.05), per-doc max {theta.max(axis=1).mean():.3f}")
    if float(mu_all.std()) < 0.05:
        raise SystemExit("POSTERIOR COLLAPSE")

    beta = dec.weight.detach().numpy()
    beta_p = np.exp(beta - beta.max(axis=0, keepdims=True))
    beta_p /= beta_p.sum(axis=0, keepdims=True)

    np.save(os.path.join(NN, f"prodlda_theta_mod{n_topics}.npy"), theta)
    np.save(os.path.join(NN, f"prodlda_beta_mod{n_topics}.npy"), beta_p)
    torch.save({"enc1": enc1.state_dict(), "enc_mu": enc_mu.state_dict(),
                "enc_sig": enc_sig.state_dict(), "dec": dec.state_dict()},
               os.path.join(NN, f"prodlda_model_mod{n_topics}.pt"))
    json.dump({"n_docs": len(kept), "vocab_size": V, "n_topics": K,
               "epochs": epochs, "dropped_dups": dropped, "seed": SEED},
              open(os.path.join(NN, f"prodlda_meta_mod{n_topics}.json"), "w"), indent=1)

    va = np.array(vocab)
    for k in range(min(K, 50)):
        top = np.argsort(-beta_p[:, k])[:10]
        print(f"T{k:02d}: " + " ".join(va[top]))
    print(f"\nsaved theta/beta/model/meta (suffix _mod{n_topics})")


if __name__ == "__main__":
    main()
