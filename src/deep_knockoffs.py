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
# v2: gated deep knockoffs (config/preregistration.yaml saebench_amendment_7)
# ---------------------------------------------------------------------------

class GatedNet(nn.Module):
    """Shared trunk with two zero-initialised heads: a value correction f and a gate correction g (both p)."""

    def __init__(self, p: int, dk: dict):
        super().__init__()
        layers, d = [], 2 * p
        for _ in range(dk["layers"]):
            layers += [nn.Linear(d, dk["hidden"]), nn.LayerNorm(dk["hidden"]), nn.GELU()]
            d = dk["hidden"]
        self.trunk = nn.Sequential(*layers)
        self.value, self.gate = nn.Linear(d, p), nn.Linear(d, p)
        for head in (self.value, self.gate):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.value(h), self.gate(h)


class GatedDeepKnockoffSampler:
    """v2 of DeepKnockoffSampler, changing the three likely causes of v1's failure (saebench_amendment_7):

    1. Zeros are learned, not cut. A latent with exact zeros in the data gets a fire / no-fire gate:
       X~_j = on_j * max(value_j, smallest positive activation of j), on_j ~ Bernoulli(sigmoid(logit0_j + g_j)),
       with logit0_j the latent's firing rate (training starts at the right zero share) and g_j learned from
       (X, noise). The draw is exact Bernoulli (logistic noise, threshold at 0); in training its gradient passes
       through the relaxed gate (straight-through, temperature tau). A firing-rate loss matches E[on_j] to
       Pr(X_j > 0). In v1 a value cut to 0 had no gradient, so the zero share could not be learned.
       Latents with no zeros (always-on ReLU latents, Gaussian data) have no gate and are clamped at their
       smallest observed value, as v1 after revision_2.
    2. Overfitting: a smaller trunk, weight decay, and early stopping on a held-out part of the training fold
       (the swap MMD, moments and firing-rate terms at fixed noise); the best state is kept.
    3. Drift towards copies: a larger decorrelation weight (v1's self-correlation rose above the block-MVR target).
    Everything else is v1's: residual on the block-MVR Gaussian knockoff, full and neighbourhood swap MMD,
    second-moment matching, cross-fitting over two row folds.
    """

    def __init__(self, X: np.ndarray, Z: np.ndarray, mu: np.ndarray, sd: np.ndarray, base, s: np.ndarray,
                 dk: dict, device, seed: int, cache: Path | None = None):
        n, p = Z.shape
        self.dk, self.device, self.base = dk, device, base
        f32 = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device)
        self.Z = f32(Z)
        self.mu_t, self.sd_t = f32(mu), f32(sd)
        has_zero = (X == 0).any(0)
        pos_min = np.where(X > 0, X, np.inf).min(0)
        floor = np.where(has_zero, np.where(np.isfinite(pos_min), pos_min, 0.0), X.min(0))
        self.floor_t = f32(floor)
        self.gated = torch.from_numpy(has_zero).to(device)
        self.fire = (X > 0).mean(0)
        fire = np.clip(self.fire, 1e-4, 1 - 1e-4)
        self.fire_t = f32(self.fire)
        self.logit0 = f32(np.log(fire) - np.log1p(-fire))
        self.zero_mass = (X == 0).mean(0)
        self.target = f32(np.clip(1.0 - s, 0.0, 1.0))
        perm = np.random.default_rng([seed, 0]).permutation(n)
        fold = np.zeros(n, dtype=np.int64)
        fold[perm[n // 2:]] = 1
        self.fold = torch.from_numpy(fold).to(device)
        C = (self.Z.T @ self.Z / n).abs()
        self.neigh = C.topk(dk["local_size"], dim=1).indices
        self.nets = [GatedNet(p, dk).to(device) for _ in range(2)]
        self.info: dict = {"cache": str(cache) if cache else None, "n_gated": int(has_zero.sum())}
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

    def _knock(self, net: nn.Module, Zb: torch.Tensor, E: torch.Tensor, U: torch.Tensor, relaxed: bool):
        """Knockoff rows on the standardised scale, and the firing probabilities. relaxed: straight-through gate."""
        b = self.base
        f, g = net(torch.cat([Zb, E], dim=1))
        t = self.mu_t + self.sd_t * (b.mu + (Zb - b.mu) @ b.A + E @ b.Lt + f)
        value = torch.maximum(t, self.floor_t)
        logit = self.logit0 + g
        noise = torch.log(U) - torch.log1p(-U)                        # logistic: 1{logit + noise > 0} ~ Bernoulli(sigmoid(logit))
        hard = (logit + noise > 0).float()
        if relaxed:
            soft = torch.sigmoid((logit + noise) / self.dk["gate_temperature"])
            on = hard + soft - soft.detach()
        else:
            on = hard
        on = torch.where(self.gated, on, torch.ones_like(on))
        return (on * value - self.mu_t) / self.sd_t, torch.sigmoid(logit)

    def _losses(self, net, Zb, E, U, mask, anchors, masks_loc):
        dk, p = self.dk, Zb.shape[1]
        h = Zb.shape[0] // 2
        Zk, prob = self._knock(net, Zb, E, U, relaxed=True)
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
        Czz = cov(Zb, Zb)
        nz = (Czz ** 2).sum()
        off = ~torch.eye(p, dtype=torch.bool, device=self.device)
        l_mom = (((Zk.mean(0) - Zb.mean(0)) ** 2).mean() + ((cov(Zk, Zk) - Czz) ** 2).sum() / nz
                 + (((cov(Zb, Zk) - Czz) * off) ** 2).sum() / nz)
        corr = cov(Zb, Zk).diagonal() / (Zb.std(0) * Zk.std(0)).clamp(min=1e-6)
        l_dec = ((corr - self.target) ** 2).mean()
        l_fire = (((prob.mean(0) - self.fire_t) ** 2)[self.gated]).mean() if bool(self.gated.any()) else Zb.new_zeros(())
        loss = (dk["gamma_mmd"] * (l_full + l_loc) + dk["lambda_moments"] * l_mom
                + dk["delta_decorrelation"] * l_dec + dk["lambda_firing"] * l_fire)
        parts = {"mmd_full": float(l_full), "mmd_local": float(l_loc), "moments": float(l_mom),
                 "decorrelation": float(l_dec), "firing": float(l_fire), "mean_corr": float(corr.mean())}
        return loss, parts

    def _draws(self, n_rows: int, p: int, gen):
        E = torch.randn((n_rows, p), generator=gen, device=self.device)
        U = torch.rand((n_rows, p), generator=gen, device=self.device).clamp(1e-6, 1 - 1e-6)
        mask = torch.rand(p, generator=gen, device=self.device) < 0.5
        anchors = torch.randint(p, (self.dk["local_blocks"],), generator=gen, device=self.device)
        masks_loc = [torch.rand(self.dk["local_size"], generator=gen, device=self.device) < 0.5 for _ in anchors]
        return E, U, mask, anchors, masks_loc

    def _train(self, f: int, seed: int) -> dict:
        """Train generator f on the rows of the other fold, with early stopping on a held-out part of them."""
        dk, net = self.dk, self.nets[f]
        p = self.Z.shape[1]
        rows = torch.nonzero(self.fold != f).squeeze(1)
        order = torch.from_numpy(np.random.default_rng([seed, 1]).permutation(len(rows))).to(self.device)
        n_val = int(len(rows) * dk["val_fraction"]) // 2 * 2
        val, tr = rows[order[:n_val]], rows[order[n_val:]]
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        torch.manual_seed(int(seed))
        net.load_state_dict(GatedNet(p, dk).to(self.device).state_dict())
        opt = torch.optim.AdamW(net.parameters(), lr=dk["lr"], weight_decay=dk["weight_decay"])
        vgen = torch.Generator(device=self.device)
        vgen.manual_seed(int(seed) + 7919)
        vdraw = self._draws(n_val, p, vgen)                       # fixed noise: validation losses are comparable
        best, best_state, bad, hist = float("inf"), None, 0, []
        t0 = time.time()
        for step in range(dk["steps"]):
            idx = tr[torch.randint(len(tr), (dk["batch"],), generator=gen, device=self.device)]
            Zb = self.Z[idx]
            loss, parts = self._losses(net, Zb, *self._draws(len(idx), p, gen))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % dk["eval_every"] == 0 or step == dk["steps"] - 1:
                net.eval()
                with torch.no_grad():
                    vloss, vparts = self._losses(net, self.Z[val], *vdraw)
                net.train()
                vl = float(vloss)
                r = {"step": step, "loss": float(loss), "val_loss": vl, **parts,
                     **{f"val_{k}": v for k, v in vparts.items()}}
                hist.append(r)
                print(f"  deep2 fold {f} step {step:>5}  loss {r['loss']:.4f}  val {vl:.4f}  mmd {parts['mmd_full']:.4f}/"
                      f"{parts['mmd_local']:.4f}  mom {parts['moments']:.4f}  decor {parts['decorrelation']:.4f}  "
                      f"fire {parts['firing']:.5f}  corr {parts['mean_corr']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
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

    @torch.no_grad()
    def sample(self, seed: int) -> tuple[np.ndarray, dict]:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        n, p = self.Z.shape
        E = torch.randn((n, p), generator=gen, device=self.device)
        U = torch.rand((n, p), generator=gen, device=self.device).clamp(1e-6, 1 - 1e-6)
        Zk = torch.empty_like(self.Z)
        for f in range(2):
            r = self.fold == f
            Zk[r] = self._knock(self.nets[f], self.Z[r], E[r], U[r], relaxed=False)[0]
        zk = Zk.double().cpu().numpy()
        xk = zk * self.sd_t.double().cpu().numpy() + self.mu_t.double().cpu().numpy()
        err = np.abs(self.zero_mass - (np.abs(xk) < 1e-6).mean(0))
        return zk, {"zero_mass_error_max": float(err.max()), "zero_mass_error_median": float(np.median(err))}
