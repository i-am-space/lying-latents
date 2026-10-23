"""Deep knockoffs for SAE latents (config/preregistration.yaml saebench_amendment_6).

A learned knockoff generator in the style of Romano, Sesia and Candes (2019, "Deep knockoffs", JASA): a
network maps (X, noise) to X~ and is trained so that (X, X~) is close in distribution to its swaps, with
second-moment matching and a decorrelation term that stops X~ from copying X. Two changes for p = 2,048:

- Residual on block-MVR Gaussian knockoffs: X~ = jump(Gaussian MVR knockoff + f(X, E)), with f's last layer
  zero-initialised, so training starts from the second-order knockoff the Gaussian arm uses and learns the
  correction beyond second moments. `jump` sets a value below the latent's smallest observed positive
  activation to 0, so the generator can produce the zero atom and the JumpReLU floor. A latent that is never 0
  in the data (always-on latents of ReLU SAEs) is clamped at its smallest value instead, so no knockoff value
  falls outside the latent's observed support (saebench_amendment_6 revision_2).
- The MMD swap loss is taken on the full (X, X~) vector and on neighbourhoods (an anchor latent and its
  most correlated latents), where the swap property is local and the full-dimension kernel is insensitive.

Cross-fitting: the rows are split in two folds; the generator applied to a fold's rows is trained on the
other fold only, so no knockoff comes from a network that saw its row. Validity is checked by the swap tests
and the planted-signal benchmark, not assumed.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


def build_net(p: int, dk: dict) -> nn.Module:
    layers, d = [], 2 * p
    for _ in range(dk["layers"]):
        layers += [nn.Linear(d, dk["hidden"]), nn.LayerNorm(dk["hidden"]), nn.GELU()]
        d = dk["hidden"]
    last = nn.Linear(d, p)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return nn.Sequential(*layers, last)


def mmd(a: torch.Tensor, b: torch.Tensor, scales) -> torch.Tensor:
    """Biased MMD^2 with a mixture of Gaussian kernels, bandwidths relative to the median cross distance."""
    xx, yy, xy = torch.cdist(a, a) ** 2, torch.cdist(b, b) ** 2, torch.cdist(a, b) ** 2
    med = xy.detach().median().clamp(min=1e-6)
    k = lambda D: sum(torch.exp(-D / (c * med)) for c in scales)
    return k(xx).mean() + k(yy).mean() - 2 * k(xy).mean()


def swap(Z: torch.Tensor, Zk: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[Z, Zk] with the columns in mask exchanged."""
    return torch.cat([torch.where(mask, Zk, Z), torch.where(mask, Z, Zk)], dim=1)


def cov(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return (A - A.mean(0)).T @ (B - B.mean(0)) / (A.shape[0] - 1)


class DeepKnockoffSampler:
    """Cross-fitted residual deep knockoffs. Draws are on the standardised scale of D["Z"].

    X: raw latents (n, p), >= 0. Z = (X - mu) / sd. base: the block-MVR GaussKnockoffGPU of the real latents
    (its A, Lt, mu); s: its diag(S), so 1 - s is the target correlation of X_j with X~_j.
    """

    def __init__(self, X: np.ndarray, Z: np.ndarray, mu: np.ndarray, sd: np.ndarray, base, s: np.ndarray,
                 dk: dict, device, seed: int, cache: Path | None = None):
        n, p = Z.shape
        self.dk, self.device, self.base = dk, device, base
        f32 = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device)
        self.Z = f32(Z)
        self.mu_t, self.sd_t = f32(mu), f32(sd)
        floor = np.where(X > 0, X, np.inf).min(0)
        self.floor_t = f32(np.where(np.isfinite(floor), floor, np.inf))
        self.below_t = f32(np.where((X == 0).any(0), 0.0, floor))      # value given below the floor: 0, or the floor
        self.zero_mass = (X == 0).mean(0)
        self.target = f32(np.clip(1.0 - s, 0.0, 1.0))
        perm = np.random.default_rng([seed, 0]).permutation(n)
        fold = np.zeros(n, dtype=np.int64)
        fold[perm[n // 2:]] = 1
        self.fold = torch.from_numpy(fold).to(device)
        C = (self.Z.T @ self.Z / n).abs()
        self.neigh = C.topk(dk["local_size"], dim=1).indices          # each latent and its most correlated ones
        self.nets = [build_net(p, dk).to(device) for _ in range(2)]
        self.info: dict = {"cache": str(cache) if cache else None}
        if cache is not None and cache.exists():
            st = torch.load(cache, map_location=device)
            for net, sd_ in zip(self.nets, st["nets"]):
                net.load_state_dict(sd_)
            self.info.update(st["info"], loaded_from_cache=True)
            print(f"  deep knockoffs: generators loaded from {cache}", flush=True)
        else:
            t = time.time()
            self.info["folds"] = [self._train(f, seed + 1 + f) for f in range(2)]
            self.info["train_seconds"] = time.time() - t
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"nets": [net.state_dict() for net in self.nets],
                            "info": {k: v for k, v in self.info.items() if k != "cache"}}, cache)
        for net in self.nets:
            net.eval()

    def _knock(self, net: nn.Module, Zb: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        b = self.base
        v = b.mu + (Zb - b.mu) @ b.A + E @ b.Lt + net(torch.cat([Zb, E], dim=1))
        t = self.mu_t + self.sd_t * v
        x = torch.where(t >= self.floor_t, t, self.below_t.expand_as(t))
        return (x - self.mu_t) / self.sd_t

    def _train(self, f: int, seed: int) -> dict:
        """Train generator f on the rows of the other fold."""
        dk, net = self.dk, self.nets[f]
        rows = torch.nonzero(self.fold != f).squeeze(1)
        p = self.Z.shape[1]
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        torch.manual_seed(int(seed))                                   # initialisation of the hidden layers
        net.load_state_dict(build_net(p, dk).to(self.device).state_dict())
        opt = torch.optim.Adam(net.parameters(), lr=dk["lr"])
        h = dk["batch"] // 2
        t0, hist = time.time(), []
        for step in range(dk["steps"]):
            idx = rows[torch.randint(len(rows), (dk["batch"],), generator=gen, device=self.device)]
            Zb = self.Z[idx]
            E = torch.randn(Zb.shape, generator=gen, device=self.device)
            Zk = self._knock(net, Zb, E)
            mask = torch.rand(p, generator=gen, device=self.device) < 0.5
            ones = torch.ones(p, dtype=torch.bool, device=self.device)
            P1 = torch.cat([Zb[:h], Zk[:h]], dim=1)
            l_full = mmd(P1, swap(Zb[h:], Zk[h:], mask), dk["mmd_scales"]) + mmd(P1, swap(Zb[h:], Zk[h:], ones), dk["mmd_scales"])
            anchors = torch.randint(p, (dk["local_blocks"],), generator=gen, device=self.device)
            l_loc = 0.0
            for a in anchors:
                c = self.neigh[a]
                m = torch.rand(len(c), generator=gen, device=self.device) < 0.5
                l_loc = l_loc + mmd(torch.cat([Zb[:h, c], Zk[:h, c]], dim=1), swap(Zb[h:, c], Zk[h:, c], m), dk["mmd_scales"])
            l_loc = l_loc / dk["local_blocks"]
            Czz = cov(Zb, Zb)
            nz = (Czz ** 2).sum()
            off = ~torch.eye(p, dtype=torch.bool, device=self.device)
            l_mom = (((Zk.mean(0) - Zb.mean(0)) ** 2).mean() + ((cov(Zk, Zk) - Czz) ** 2).sum() / nz
                     + (((cov(Zb, Zk) - Czz) * off) ** 2).sum() / nz)
            corr = (cov(Zb, Zk).diagonal() / (Zb.std(0) * Zk.std(0)).clamp(min=1e-6))
            l_dec = ((corr - self.target) ** 2).mean()
            loss = dk["gamma_mmd"] * (l_full + l_loc) + dk["lambda_moments"] * l_mom + dk["delta_decorrelation"] * l_dec
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % dk["log_every"] == 0 or step == dk["steps"] - 1:
                r = {"step": step, "loss": loss.item(), "mmd_full": l_full.item(), "mmd_local": float(l_loc),
                     "moments": l_mom.item(), "decorrelation": l_dec.item(), "mean_corr": corr.mean().item()}
                hist.append(r)
                print(f"  deep fold {f} step {step:>5}  loss {r['loss']:.4f}  mmd full {r['mmd_full']:.4f} "
                      f"local {r['mmd_local']:.4f}  moments {r['moments']:.4f}  decor {r['decorrelation']:.4f}  "
                      f"corr {r['mean_corr']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        return {"fold": f, "train_rows": int(len(rows)), "seconds": time.time() - t0, "history": hist}

    @torch.no_grad()
    def sample(self, seed: int) -> tuple[np.ndarray, dict]:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        E = torch.randn(self.Z.shape, generator=gen, device=self.device)
        Zk = torch.empty_like(self.Z)
        for f in range(2):
            r = self.fold == f
            Zk[r] = self._knock(self.nets[f], self.Z[r], E[r])
        zk = Zk.double().cpu().numpy()
        xk = zk * self.sd_t.double().cpu().numpy() + self.mu_t.double().cpu().numpy()
        info = {"zero_mass_error_max": float(np.abs(self.zero_mass - (np.abs(xk) < 1e-6).mean(0)).max()),
                "zero_mass_error_median": float(np.median(np.abs(self.zero_mass - (np.abs(xk) < 1e-6).mean(0))))}
        return zk, info


# ---------------------------------------------------------------------------
# shared diagnostics: second- and third-order moment errors of a knockoff draw
# ---------------------------------------------------------------------------

def _offdiag3(k: int, device, pair: bool) -> torch.Tensor:
    """Mask over (i, a, b) of a k x k x k moment tensor. pair=False: a != i and b != i (swap of i alone).
    pair=True: additionally a != b (swap of the pair {i, a}, third latent b outside it)."""
    i = torch.arange(k, device=device)
    m = (i[:, None, None] != i[None, :, None]) & (i[:, None, None] != i[None, None, :])
    if pair:
        m = m & (i[None, :, None] != i[None, None, :])
    return m


def third_order_loss(Z: torch.Tensor, Zk: torch.Tensor, neigh: torch.Tensor, anchors) -> torch.Tensor:
    """Relative error of the third cross-moments a valid knockoff must reproduce, on neighbourhoods of correlated
    latents (an anchor and its most correlated latents). Pairwise exchangeability implies, for i, a, b in a
    neighbourhood: E[X~_i X_a X_b] = E[X_i X_a X_b] (swap of i, with a, b != i) and E[X~_i X~_a X_b] =
    E[X_i X_a X_b] (swap of {i, a}, b outside it); and, for every latent, E[X_i^2 X~_i] = E[X~_i^2 X_i] and
    E[X~_i^3] = E[X_i^3]. Labels that depend on products of latents (interaction cells) are what a knockoff
    lacking these moments gets wrong. Inputs are standardised; moments are of centred values."""
    n = Z.shape[0]
    Zc, Kc = Z - Z.mean(0), Zk - Zk.mean(0)
    num, den = Z.new_zeros(()), Z.new_zeros(())
    for a in anchors:
        c = neigh[a]
        R, K = Zc[:, c], Kc[:, c]
        k = R.shape[1]
        RR = (R[:, :, None] * R[:, None, :]).reshape(n, k * k)
        KR = (K[:, :, None] * R[:, None, :]).reshape(n, k * k)
        t_rrr = (R.T @ RR).reshape(k, k, k) / n
        t_krr = (K.T @ RR).reshape(k, k, k) / n
        t_kkr = (K.T @ KR).reshape(k, k, k) / n
        m1, m2 = _offdiag3(k, Z.device, False), _offdiag3(k, Z.device, True)
        num = num + ((t_krr - t_rrr)[m1] ** 2).sum() + ((t_kkr - t_rrr)[m2] ** 2).sum()
        den = den + 2 * (t_rrr[m1] ** 2).sum()
    z3 = (Zc ** 3).mean(0)
    num = num + (((Zc ** 2 * Kc).mean(0) - (Kc ** 2 * Zc).mean(0)) ** 2).sum() + (((Kc ** 3).mean(0) - z3) ** 2).sum()
    den = den + 2 * (z3 ** 2).sum()
    return num / den.clamp(min=1e-12)


def second_order_loss(Z: torch.Tensor, Zk: torch.Tensor) -> torch.Tensor:
    """Relative error of the second moments a knockoff must reproduce: mean, cov(X~) = cov(X), and
    cov(X, X~) = cov(X) off the diagonal."""
    p = Z.shape[1]
    Czz = cov(Z, Z)
    nz = (Czz ** 2).sum()
    off = ~torch.eye(p, dtype=torch.bool, device=Z.device)
    return (((Zk.mean(0) - Z.mean(0)) ** 2).mean() + ((cov(Zk, Zk) - Czz) ** 2).sum() / nz
            + (((cov(Z, Zk) - Czz) * off) ** 2).sum() / nz)


@torch.no_grad()
def moment_errors(Z: torch.Tensor, Zk: torch.Tensor, neigh: torch.Tensor, anchors) -> dict:
    return {"second_order": float(second_order_loss(Z, Zk)), "third_order": float(third_order_loss(Z, Zk, neigh, anchors))}


@torch.no_grad()
def generalisation_gap(smp: "DeepKnockoffSampler", seed: int, n_anchors: int = 64) -> dict:
    """For each v1 generator: moment errors of its knockoffs on its own training rows against the held-out rows it is
    applied to, with the block-MVR Gaussian knockoff on the same rows (same noise) as the reference. A training error
    far below the held-out error means the generator fits its training half and does not generalise."""
    gen = torch.Generator(device=smp.device)
    gen.manual_seed(int(seed))
    n, p = smp.Z.shape
    anchors = torch.randint(p, (n_anchors,), generator=gen, device=smp.device)
    E = torch.randn((n, p), generator=gen, device=smp.device)
    b = smp.base
    out = {}
    for f in range(2):
        res = {}
        for part, rows in (("train", smp.fold != f), ("heldout", smp.fold == f)):
            Zr, Er = smp.Z[rows], E[rows]
            deep = smp._knock(smp.nets[f], Zr, Er)
            gauss = b.mu + (Zr - b.mu) @ b.A + Er @ b.Lt
            res[part] = {"deep": moment_errors(Zr, deep, smp.neigh, anchors),
                         "gauss_mvr": moment_errors(Zr, gauss, smp.neigh, anchors), "rows": int(rows.sum())}
        res["heldout_over_train"] = {k: res["heldout"]["deep"][k] / max(res["train"]["deep"][k], 1e-12)
                                     for k in ("second_order", "third_order")}
        out[f"fold{f}"] = res
    return out


# ---------------------------------------------------------------------------
# v2: second-order-exact deep knockoffs (config/preregistration.yaml saebench_amendment_7 revision_1)
# ---------------------------------------------------------------------------

class ExactDeepKnockoffSampler:
    """X~ = mu + (X - mu) A + N, the block-MVR Gaussian knockoff with its Gaussian noise E V^(1/2) replaced by a learned
    noise N = E Lt + f(X, E). Every pair (X, X~) with E[N] = 0, cov(N, X) = 0 and cov(N) = V has exactly the
    Gaussian knockoff's second moments, whatever the law of N, so N can carry dependence on X beyond second order
    (what the Gaussian knockoff lacks: it fails only where labels depend on products of latents) while the
    covariance structure that keeps the Gaussian knockoff valid in linear cells is exact.

    Training (per fold, on the other fold's rows; 15% of them held out for early stopping): swap MMD on the full
    vector and on neighbourhoods (as v1), the third-order exchangeability moments (third_order_loss), and a
    second-order penalty on N. f's last layer is zero-initialised, so training starts at the block-MVR knockoff.
    Draws: N is computed for every row, then centred, its linear regression on X removed and its covariance mapped
    to V exactly, on the whole sample (a sample-level linear map, like the covariance estimate the Gaussian
    knockoff already uses; it does not involve the labels). If f learns nothing, a draw is a block-MVR knockoff
    with exact in-sample second moments. No zero floor: v1 matched the zero atom (1{x = 0} rule at chance) yet its
    null coin was 0.58-0.61 in linear cells, where block MVR's is 0.49, i.e. the second moments it gave up mattered
    more than the zero atom it gained (saebench_amendment_7 revision_1).
    """

    def __init__(self, Z: np.ndarray, base, dk: dict, device, seed: int, cache: Path | None = None):
        n, p = Z.shape
        self.dk, self.device, self.base = dk, device, base
        self.Z = torch.from_numpy(np.asarray(Z, dtype=np.float32)).to(device)
        Lt = base.Lt.double().cpu().numpy()
        V = Lt.T @ Lt
        ev, U = np.linalg.eigh(0.5 * (V + V.T))
        self.V_t = torch.from_numpy(V.astype(np.float32)).to(device)
        self.V_half = (U * np.sqrt(np.clip(ev, 0.0, None))) @ U.T          # symmetric square root of V
        perm = np.random.default_rng([seed, 0]).permutation(n)
        fold = np.zeros(n, dtype=np.int64)
        fold[perm[n // 2:]] = 1
        self.fold = torch.from_numpy(fold).to(device)
        C = (self.Z.T @ self.Z / n).abs()
        self.neigh = C.topk(dk["local_size"], dim=1).indices
        self.nets = [build_net(p, dk).to(device) for _ in range(2)]
        self.info: dict = {"cache": str(cache) if cache else None}
        if cache is not None and cache.exists():
            st = torch.load(cache, map_location=device)
            for net, sd_ in zip(self.nets, st["nets"]):
                net.load_state_dict(sd_)
            self.info.update(st["info"], loaded_from_cache=True)
            print(f"  deep v2: generators loaded from {cache}", flush=True)
        else:
            t = time.time()
            self.info["folds"] = [self._train(f, seed + 1 + f) for f in range(2)]
            self.info["train_seconds"] = time.time() - t
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"nets": [net.state_dict() for net in self.nets],
                            "info": {k: v for k, v in self.info.items() if k != "cache"}}, cache)
        for net in self.nets:
            net.eval()

    def _noise(self, net: nn.Module, Zb: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        return E @ self.base.Lt + net(torch.cat([Zb, E], dim=1))

    def _knock(self, net: nn.Module, Zb: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """Training-time knockoff (second moments matched by penalty, not yet exactly)."""
        b = self.base
        return b.mu + (Zb - b.mu) @ b.A + self._noise(net, Zb, E)

    def _losses(self, net, Zb, E, mask, anchors, masks_loc):
        dk, p = self.dk, Zb.shape[1]
        h = Zb.shape[0] // 2
        G = E @ self.base.Lt                                        # the Gaussian noise of the same draw
        F = net(torch.cat([Zb, E], dim=1))                          # the learned part
        N = G + F
        Zk = self.base.mu + (Zb - self.base.mu) @ self.base.A + N
        ones = torch.ones(p, dtype=torch.bool, device=self.device)
        P1 = torch.cat([Zb[:h], Zk[:h]], dim=1)
        l_full = mmd(P1, swap(Zb[h:2 * h], Zk[h:2 * h], mask), dk["mmd_scales"]) \
            + mmd(P1, swap(Zb[h:2 * h], Zk[h:2 * h], ones), dk["mmd_scales"])
        l_loc = 0.0
        for a, m in zip(anchors, masks_loc):
            c = self.neigh[a]
            l_loc = l_loc + mmd(torch.cat([Zb[:h, c], Zk[:h, c]], dim=1), swap(Zb[h:2 * h, c], Zk[h:2 * h, c], m),
                                dk["mmd_scales"])
        l_loc = l_loc / len(anchors)
        l_third = third_order_loss(Zb, Zk, self.neigh, anchors)
        # second-order penalty on the learned part, against the same draw's Gaussian noise rather than V itself: a
        # batch of 2,048 rows estimates a 2,048 x 2,048 covariance with relative error near 1, and that sampling
        # error is shared by N and G, so it cancels here (the term is exactly 0 when f = 0)
        Czz = cov(Zb, Zb)
        l_noise = ((F.mean(0) ** 2).mean() + (cov(F, Zb) ** 2).sum() / (Czz ** 2).sum()
                   + ((cov(N, N) - cov(G, G)) ** 2).sum() / (self.V_t ** 2).sum())
        loss = dk["gamma_mmd"] * (l_full + l_loc) + dk["lambda_third"] * l_third + dk["lambda_moments"] * l_noise
        val = lambda t: float(t.detach()) if torch.is_tensor(t) else float(t)
        parts = {"mmd_full": val(l_full), "mmd_local": val(l_loc), "third": val(l_third), "noise_moments": val(l_noise)}
        return loss, parts

    def _draws(self, n_rows: int, p: int, gen):
        E = torch.randn((n_rows, p), generator=gen, device=self.device)
        mask = torch.rand(p, generator=gen, device=self.device) < 0.5
        anchors = torch.randint(p, (self.dk["local_blocks"],), generator=gen, device=self.device)
        masks_loc = [torch.rand(self.dk["local_size"], generator=gen, device=self.device) < 0.5 for _ in anchors]
        return E, mask, anchors, masks_loc

    def _train(self, f: int, seed: int) -> dict:
        """Train generator f on the rows of the other fold, early-stopped on a held-out part of them."""
        dk, net = self.dk, self.nets[f]
        p = self.Z.shape[1]
        rows = torch.nonzero(self.fold != f).squeeze(1)
        order = torch.from_numpy(np.random.default_rng([seed, 1]).permutation(len(rows))).to(self.device)
        n_val = int(len(rows) * dk["val_fraction"]) // 2 * 2
        val, tr = rows[order[:n_val]], rows[order[n_val:]]
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        torch.manual_seed(int(seed))
        net.load_state_dict(build_net(p, dk).to(self.device).state_dict())
        opt = torch.optim.AdamW(net.parameters(), lr=dk["lr"], weight_decay=dk["weight_decay"])
        vgen = torch.Generator(device=self.device)
        vgen.manual_seed(int(seed) + 7919)
        vdraw = self._draws(n_val, p, vgen)                        # fixed noise: validation losses are comparable
        best, best_state, bad, hist = float("inf"), None, 0, []
        t0 = time.time()
        for step in range(dk["steps"]):
            idx = tr[torch.randint(len(tr), (dk["batch"],), generator=gen, device=self.device)]
            loss, parts = self._losses(net, self.Z[idx], *self._draws(len(idx), p, gen))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % dk["eval_every"] == 0 or step == dk["steps"] - 1:
                net.eval()
                with torch.no_grad():
                    vloss, vparts = self._losses(net, self.Z[val], *vdraw)
                net.train()
                vl = float(vloss)
                hist.append({"step": step, "loss": loss.item(), "val_loss": vl, **parts,
                             **{f"val_{k}": v for k, v in vparts.items()}})
                print(f"  deep2 fold {f} step {step:>5}  loss {loss.item():.4f}  val {vl:.4f}  mmd {parts['mmd_full']:.4f}/"
                      f"{parts['mmd_local']:.4f}  third {parts['third']:.4f} (val {vparts['third']:.4f})  "
                      f"noise {parts['noise_moments']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
                if vl < best - dk["min_delta"]:
                    best, bad = vl, 0
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= dk["patience"]:
                        print(f"  deep2 fold {f}: early stop at step {step} (best val {best:.4f})", flush=True)
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        return {"fold": f, "train_rows": int(len(tr)), "val_rows": int(n_val), "best_val_loss": best,
                "stopped_at": hist[-1]["step"], "seconds": time.time() - t0, "history": hist}

    def _exact(self, N: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Sample-level map of N to E[N] = 0, cov(N, X) = 0 and cov(N) = V exactly (float64): centre, remove the
        least-squares regression on X, then N1 C^(-1/2) V^(1/2) with symmetric roots (the identity when C = V)."""
        Nc = N - N.mean(0)
        Zc = Z - Z.mean(0)
        beta = np.linalg.lstsq(Zc, Nc, rcond=None)[0]
        N1 = Nc - Zc @ beta
        C = N1.T @ N1 / (len(N1) - 1)
        ev, U = np.linalg.eigh(0.5 * (C + C.T))
        inv_half = (U / np.sqrt(np.clip(ev, ev.max() * 1e-10, None))) @ U.T
        return N1 @ inv_half @ self.V_half

    @torch.no_grad()
    def sample(self, seed: int, learned: bool = True) -> tuple[np.ndarray, dict]:
        """learned=False: the ablation arm deep2_zero, f = 0 (the block-MVR knockoff with the exact map only)."""
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        n, p = self.Z.shape
        E = torch.randn((n, p), generator=gen, device=self.device)
        N = E @ self.base.Lt
        if learned:
            for f in range(2):
                r = self.fold == f
                N[r] = self._noise(self.nets[f], self.Z[r], E[r])
        Zd = self.Z.double().cpu().numpy()
        Nraw = N.double().cpu().numpy()
        Nx = self._exact(Nraw, Zd)
        b = self.base
        mu = b.mu.double().cpu().numpy()
        zk = mu + (Zd - mu) @ b.A.double().cpu().numpy() + Nx
        # diagnostics (DK8): how far the exact map moved N, and the moment errors against a Gaussian draw, same noise
        Zk_t = torch.from_numpy(zk.astype(np.float32)).to(self.device)
        g = b.mu + (self.Z - b.mu) @ b.A + E @ b.Lt
        anchors = torch.randint(p, (32,), generator=gen, device=self.device)
        rank = lambda A: np.argsort(np.argsort(A, axis=0), axis=0).astype(np.float64)
        sel = np.random.default_rng(int(seed)).choice(p, size=min(p, 256), replace=False)
        rz, rk = rank(Zd[:, sel]), rank(zk[:, sel])
        spearman = [float(np.corrcoef(rz[:, j], rk[:, j])[0, 1]) for j in range(len(sel))]
        info = {"exact_map_relative_change": float(np.linalg.norm(Nx - Nraw) / max(np.linalg.norm(Nraw), 1e-12)),
                "learned_part_relative_norm": float(np.linalg.norm(Nraw - (E @ b.Lt).double().cpu().numpy())
                                                    / max(np.linalg.norm(Nraw), 1e-12)),
                "third_order_error": float(third_order_loss(self.Z, Zk_t, self.neigh, anchors)),
                "third_order_error_gauss_mvr": float(third_order_loss(self.Z, g, self.neigh, anchors)),
                "spearman_self_corr_median": float(np.nanmedian(spearman))}
        return zk, info
