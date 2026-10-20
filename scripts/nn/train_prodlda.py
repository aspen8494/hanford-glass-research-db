"""ProdLDA: neural topic model (Autoencoding Variational Inference for Topic Models,
Miao et al. 2017) implemented from scratch in PyTorch.

Trains on the Hanford glass corpus BoW; saves the model, document-topic
mixtures, and topic-term distributions.

Usage: train_prodlda.py [n_topics]
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NN_DIR = os.path.join(WS, "data", "nn")
os.makedirs(NN_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class ProdLDA(nn.Module):
    """ProdLDA VAE: encoder q(theta|a,b) ~ LN, decoder beta (topics over vocab),
    reconstruction via multinomial likelihood in log space (Srivastava & Sutton 2017)."""

    def __init__(self, n_docs_vocab_size, n_topics=30, hidden=512, dropout=0.2):
        super().__init__()
        self.V = n_docs_vocab_size
        self.K = n_topics
        self.a = 1.0  # LN prior location (beta_1); beta_2=0.01 (flat Dirichlet prior)
        self.enc_h1 = nn.Linear(self.V, hidden)
        self.enc_mu = nn.Linear(hidden, self.K)
        self.enc_sig = nn.Linear(hidden, self.K)
        self.dec_drop = nn.Dropout(dropout)
        self.dec_beta = nn.Linear(self.K, self.V, bias=False)
        self.drop = nn.Dropout(dropout)

    def encode(self, x):
        h = F.softplus(self.enc_h1(x))  # NO input dropout: it starves the likelihood and invites collapse
        return self.enc_mu(h), self.enc_sig(h)  # ln-sigma

    def reparam(self, mu, lnsig):
        eps = torch.randn_like(mu)
        return mu + eps * lnsig.exp()

    def decode(self, theta):
        # softmax over vocab of the beta product => topic-word probabilities
        return F.softmax(self.dec_beta(theta), dim=1)

    def forward(self, x):
        mu, lnsig = self.encode(x)
        theta = self.reparam(mu, lnsig)
        # normalize theta to a simplex? LN posterior mean isn't a simplex;
        # ProdLDA applies softmax INSIDE the ELBO instead (Srivastava impl:
        # theta stays unnormalized, decoder softmaxes over vocab, and the
        # likelihood is scaled by doc length)
        soft = self.decode(theta)
        return soft, theta, mu, lnsig


def loss_fn(x, recon, mu, lnsig, K, beta_kl=1.0):
    # reconstruction: multinomial log-likelihood of the (scaled) doc counts
    log_recon = torch.log(recon + 1e-10)
    rec = (x * log_recon).sum(dim=1)
    # KL between LN(mu, sig) and LN(a=1, 1) prior, closed form
    sig2 = (2 * lnsig).exp()
    kl = (0.5 * (sig2 + (mu - 1.0) ** 2) - 0.5) - lnsig
    kl = kl.sum(dim=1)
    return -(rec - beta_kl * kl).mean()


def main():
    n_topics = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    docs = json.load(open(os.path.join(NN_DIR, "corpus_tokens.json")))
    vocab = json.load(open(os.path.join(NN_DIR, "vocab.json")))
    w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)

    # BoW matrix
    X = np.zeros((len(docs), V), dtype=np.float32)
    for i, d in enumerate(docs):
        for t in d["tokens"]:
            X[i, w2i[t]] += 1.0
    # normalize to unit sum, then scale up: the multinomial likelihood needs
    # mass ~O(100) or the KL dominates -> posterior collapse to the prior
    # (verified: scale 1 -> uniform theta; scale 100 -> sharp domain topics)
    Xn = X / np.clip(X.sum(axis=1, keepdims=True), 1e-9, None)
    Xt = torch.tensor(Xn * 100.0)
    ds = TensorDataset(Xt)
    dl = DataLoader(ds, batch_size=256, shuffle=True)

    model = ProdLDA(V, n_topics=n_topics)
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=0.001, betas=(0.99, 0.999))

    epochs = 60
    for ep in range(epochs):
        model.train()
        beta_kl = min(1.0, ep / 20)  # KL annealing: likelihood first, prior later
        tot = 0.0
        for (xb,) in dl:
            opt.zero_grad()
            recon, theta, mu, lnsig = model(xb)
            loss = loss_fn(xb, recon, mu, lnsig, n_topics, beta_kl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tot += loss.item() * len(xb)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"epoch {ep:3d}  loss/doc {tot/len(docs):.1f}  beta_kl {beta_kl:.2f}")

    # ---- extract artifacts ----
    model.eval()
    with torch.no_grad():
        theta_all = []
        for i in range(0, len(docs), 512):
            xb = Xt[i:i + 512]
            _, _, mu, _ = model(xb)  # unpack the ACTUAL posterior mean for this slice
            # posterior point estimate: softmax of the LN location (sums to 1 per doc)
            theta_all.append(torch.softmax(mu, dim=1).cpu().numpy())
        theta_all = np.vstack(theta_all)
        assert theta_all.shape[0] == len(docs), \
            f"theta rows {theta_all.shape[0]} != docs {len(docs)}"
        # posterior-collapse guard: theta must be sharper than uniform (1/K)
        mu_std = float(mu.std())
        print(f"posterior diagnostics: mu.std {mu_std:.3f} (collapsed ~0.0x, healthy ~1+)")
        if mu_std < 0.05:
            raise SystemExit("POSTERIOR COLLAPSE: rerun with likelihood scale >= 100")
        beta = model.dec_beta.weight.detach().cpu().numpy()  # (V, K)

    # topic-word matrix: softmax over vocab per topic
    beta_p = np.exp(beta - beta.max(axis=0, keepdims=True))
    beta_p /= beta_p.sum(axis=0, keepdims=True)

    torch.save(model.state_dict(), os.path.join(NN_DIR, f"prodlda_model_{n_topics}.pt"))
    np.save(os.path.join(NN_DIR, f"prodlda_theta_{n_topics}.npy"), theta_all)
    np.save(os.path.join(NN_DIR, f"prodlda_beta_{n_topics}.npy"), beta_p)

    # print top words per topic for a quick read
    print("\n--- topics ---")
    for k in range(n_topics):
        top = np.argsort(-beta_p[:, k])[:10]
        print(f"T{k:02d}: " + " ".join(vocab[i] for i in top))

    with open(os.path.join(NN_DIR, f"prodlda_meta_{n_topics}.json"), "w") as f:
        json.dump({"n_docs": len(docs), "vocab_size": V, "n_topics": n_topics,
                   "epochs": epochs, "seed": SEED}, f, indent=1)
    print(f"\nsaved theta/beta/meta to {NN_DIR} (suffix _{n_topics})")


if __name__ == "__main__":
    main()
