"""Stage 5 — interventional FDR control for SAE latents (config/preregistration.yaml stage5_amendment_1).

Hypothesis for latent j:  H_j: tau_j <= delta,  tau_j = E_x[Delta_j(x)],  Delta_j(x) = m(x) - m(x; intervene on j),
with m(x) the model's probability of the correct class among the task's label tokens at the end of a few-shot
prompt. m is in [0, 1], so Delta_j(x) is in [-1, 1] and the betting e-values need no clipping. Delta_j > 0 means the
latent supports the correct answer; a latent whose removal helps is not claimed (one-sided by design). An
intervention changes latent j's activation on the query's tokens at the SAE layer and adds (new - old) times j's
decoder vector to the residual stream, so the SAE's reconstruction error is kept. The fixed prompt prefix (the
few-shot demonstrations) is never intervened on.

Stages (each resumes; outputs in <results_dir>/stage5/<task>/):
  pilot      replay-vs-full-forward equivalence check, prompted accuracy (gate), delta from the SD of m, timing
  screen     split A: attribution-patching score of every latent (first-order effect of zero ablation), the K
             candidates, sparse query-token activations (for the resampling sampler and mean ablation)
  baselines  attribution top-k (Marks et al.); on the pooled-latent caches: Gaussian equicorrelated knockoffs
             (Enkhbayar 2025) and observational e-value splitting, one-vs-rest per class
  delta      Delta_j(x) for every latent in U = candidates + baseline selections on the split-B and split-O inputs
             (primary intervention), and for the candidates under the sensitivity interventions
  evaluate   betting e-values and e-BH on subsamples of split B (n = 100..2,000, 200 replicates); FDR and power
             against the oracle tau_j from split O; the baselines against the same oracle; sensitivity analyses
  summary    one JSON over the tasks of a model config

Usage: python src/stage5_interventional.py --config config/default.yaml --task sst2 --stage all
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))   # src/
from common import ROOT, cache_path, load_config, results_dir, stream_rng   # noqa: E402

STAGES = ("pilot", "screen", "baselines", "delta", "evaluate")
INTERVENTIONS = ("resample", "zero", "mean")


# ---------------------------------------------------------------------------
# tasks, splits, prompts
# ---------------------------------------------------------------------------

def task_texts(cfg: dict, tname: str):
    """(texts, y, classes) with y the index into classes (the config's label order). Text i is row i of the task's
    pooled-latent cache (SST-2: dataset order, as cache_activations; SAEBench: sbutil.pooled_rows order)."""
    tc = cfg["stage5"]["tasks"][tname]
    if tc["source"] == "sst2":
        from datasets import load_dataset
        dc = cfg["data"]
        ds = load_dataset(dc["dataset"], dc["dataset_config"], split=dc["split"])
        texts = list(ds[dc["text_column"]])
        codes = [str(int(v)) for v in ds[dc["label_column"]]]
    else:
        sys.path.insert(0, str(ROOT / "src" / "saebench"))
        from sbutil import pooled_rows, read_json, tasks_path
        tj = read_json(tasks_path(cfg, tc["dataset"]))
        texts, cls, _ = pooled_rows(tj)
        codes = [str(tj["classes"][int(c)]) for c in cls]
    classes = [str(c) for c in tc["labels"]]
    missing = sorted(set(codes) - set(classes))
    if missing:
        raise SystemExit(f"{tname}: class codes {missing} have no label word in stage5.tasks.{tname}.labels")
    idx = {c: i for i, c in enumerate(classes)}
    return texts, np.asarray([idx[c] for c in codes], dtype=np.int64), classes


def make_splits(cfg: dict, tname: str, y: np.ndarray, limit: int | None = None) -> dict:
    """Disjoint index sets from one seeded shuffle: demo (shots per class), pilot, A, B, O, and obs (split A plus
    every unused text, capped) for the observational baselines. limit (debug) caps every split."""
    s5 = cfg["stage5"]
    ti = list(s5["tasks"]).index(tname)
    order = stream_rng(cfg, "s5_split", ti).permutation(len(y))
    demo = []
    for c in range(int(y.max()) + 1):
        demo += [int(i) for i in order[y[order] == c][: s5["shots_per_class"]]]
    demo = [int(i) for i in order if i in set(demo)]               # demonstrations in shuffled order
    rest = order[~np.isin(order, demo)]
    sz = {k: s5["splits"][k] for k in ("pilot", "screen", "test_pool", "oracle")}
    if limit:
        sz = {k: min(v, limit) for k, v in sz.items()}
    need = sum(sz.values())
    if need > len(rest):
        raise SystemExit(f"{tname}: {len(rest)} texts after the demonstrations, {need} needed by stage5.splits")
    out, at = {"demo": np.asarray(demo)}, 0
    for k, key in (("pilot", "pilot"), ("screen", "A"), ("test_pool", "B"), ("oracle", "O")):
        out[key] = rest[at: at + sz[k]]
        at += sz[k]
    out["obs"] = np.concatenate([out["A"], rest[at:]])[: s5["splits"]["obs_max_rows"]]
    return out


class Prompts:
    """[BOS] instruction, demonstrations (head + text + tail + label), then the query head + text + tail. Pieces are
    tokenised separately and concatenated, so the label tokens a demonstration shows are exactly the scored ones."""

    def __init__(self, tok, tc: dict, classes: list, demo_texts: list, demo_y, max_demo: int):
        self.tok = tok
        enc = lambda s: list(tok(s, add_special_tokens=False)["input_ids"])
        bos = tok.bos_token_id
        self.bos = [bos] if bos is not None and tok("a")["input_ids"][0] == bos else []
        self.head, self.tail = enc(tc["head"]), enc(tc["tail"])
        self.words = [str(tc["labels"][c]) for c in classes]
        firsts = [enc(" " + w) for w in self.words]
        self.label_ids = [f[0] for f in firsts]
        if len(set(self.label_ids)) != len(self.label_ids):
            raise SystemExit(f"label words {self.words} share a first token: {self.label_ids}")
        nl = enc("\n\n")
        prefix = enc(tc["instruction"]) + nl
        for t, c in zip(demo_texts, demo_y):
            prefix += self.head + enc(t)[:max_demo] + self.tail + [self.label_ids[int(c)]] + nl
        self.prefix = self.bos + prefix
        self.max_query = int(tc["max_query_tokens"])

    def build(self, text: str):
        """(token ids, q0): the query's tokens are positions q0..end."""
        q = self.head + list(self.tok(text, add_special_tokens=False)["input_ids"])[: self.max_query] + self.tail
        return torch.tensor(self.prefix + q, dtype=torch.long), len(self.prefix)


# ---------------------------------------------------------------------------
# model: base pass and replay of the layers after the SAE layer
# ---------------------------------------------------------------------------

def _expand(v, K: int):
    """Captured layer arguments for a batch of K copies of one input: tensors with batch dim 1 are expanded."""
    if torch.is_tensor(v):
        return v.expand(K, *v.shape[1:]) if v.dim() >= 2 and v.shape[0] == 1 else v
    if isinstance(v, tuple):
        return tuple(_expand(x, K) for x in v)
    if isinstance(v, list):
        return [_expand(x, K) for x in v]
    if isinstance(v, dict):
        return {k: _expand(x, K) for k, x in v.items()}
    return v


class Runner:
    """The decoder's residual stream after layer L (the SAE's hook point, resid_post) for one input, with every
    later layer's call arguments (masks, rotary embeddings, positions) captured from that same forward pass; and the
    replay of the later layers on K edited copies of it, down to the label-token logits (final norm, the label rows
    of the output embedding, the model's final logit soft-capping). An intervention at layer L cannot change
    anything before it, so replaying from L reproduces the full forward pass at a fraction of the cost; the pilot
    stage checks this numerically against model(...) with and without an intervention."""

    def __init__(self, model, sae, layer: int, label_ids: list):
        self.model, self.sae, self.L = model, sae, int(layer)
        self.dec = model.model if hasattr(model, "model") else model.base_model
        self.later = list(self.dec.layers)[self.L + 1:]
        self.norm = self.dec.norm
        self.W_lab = model.get_output_embeddings().weight[label_ids].detach()
        self.cap = getattr(model.config, "final_logit_softcapping", None)
        self.W_dec = sae.W_dec.detach()
        self.label_ids = list(label_ids)
        self.device = next(model.parameters()).device
        model.requires_grad_(False)
        sae.requires_grad_(False)

    @torch.no_grad()
    def base(self, ids: torch.Tensor):
        caps, hooks = [None] * len(self.later), []
        for i, m in enumerate(self.later):
            def pre(mod, args, kwargs, i=i):
                caps[i] = (args[1:], {k: v for k, v in kwargs.items() if k != "hidden_states"})
            hooks.append(m.register_forward_pre_hook(pre, with_kwargs=True))
        try:
            out = self.dec(input_ids=ids[None].to(self.device), output_hidden_states=True, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        return out.hidden_states[self.L + 1][0], caps

    def replay(self, H: torch.Tensor, caps) -> torch.Tensor:
        """Label logits (K, C) for K edited copies H (K, T, d) of the layer-L residual stream."""
        K, x = H.shape[0], H
        for m, (args, kw) in zip(self.later, caps):
            out = m(x, *_expand(args, K), **_expand(kw, K))
            x = out[0] if isinstance(out, (tuple, list)) else out
        z = self.norm(x[:, -1])
        logits = z.float() @ self.W_lab.float().T
        if self.cap:
            logits = torch.tanh(logits / self.cap) * self.cap
        return logits

    @torch.no_grad()
    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return self.sae.encode(h.to(self.W_dec.dtype)).float()

    @torch.no_grad()
    def full_probs(self, ids: torch.Tensor, delta: torch.Tensor | None = None) -> torch.Tensor:
        """Class probabilities from the unmodified model call (logits soft-capped by the model itself), optionally
        with delta (T, d) added to layer L's output by a forward hook: the reference for the replay."""
        hook = None
        if delta is not None:
            def add(mod, inp, out):
                if isinstance(out, (tuple, list)):
                    return (out[0] + delta[None].to(out[0].dtype),) + tuple(out[1:])
                return out + delta[None].to(out.dtype)
            hook = self.dec.layers[self.L].register_forward_hook(add)
        try:
            logits = self.model(input_ids=ids[None].to(self.device), use_cache=False).logits[0, -1, self.label_ids]
        finally:
            if hook is not None:
                hook.remove()
        return torch.softmax(logits.float(), -1)


def correct_prob(logits: torch.Tensor, y: int) -> torch.Tensor:
    return torch.softmax(logits.float(), -1)[:, int(y)]


# ---------------------------------------------------------------------------
# resampling intervention: a token-level hurdle sampler of a_j | a_-j
# ---------------------------------------------------------------------------

def _logistic_newton(X: np.ndarray, y: np.ndarray, ridge: float, iters: int = 30):
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    r = min(max(y.mean(), 1e-4), 1 - 1e-4)
    w[-1] = math.log(r / (1 - r))
    R = ridge * np.eye(d + 1)
    R[-1, -1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) + R @ w
        H = Xb.T @ (Xb * (p * (1 - p))[:, None]) + R + 1e-9 * np.eye(d + 1)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w[:-1], float(w[-1])


class ResampleBank:
    """Per latent j in a list: covariates = log1p activations of the M latents most co-active with j on split-A query
    tokens (|phi| of the firing indicators), standardised; Pr(a_j > 0 | covariates) by ridge logistic regression and
    log a_j | a_j > 0 by ridge linear regression with Gaussian residuals (log-normal size, clipped to the observed
    range); a marginal hurdle if j has fewer than min_positive positive (or zero) tokens. Also j's token mean (the
    mean-ablation value). Draws are per input and per token, conditional on the input's own other activations."""

    def __init__(self, tokens: dict, latents: np.ndarray, d_sae: int, rc: dict):
        import scipy.sparse as sp
        t0 = time.time()
        S = sp.csr_matrix((tokens["val"], tokens["idx"], tokens["indptr"]), shape=(len(tokens["indptr"]) - 1, d_sae))
        n = S.shape[0]
        B = S.copy()
        B.data[:] = 1.0
        f = np.asarray(B.sum(0)).ravel()
        Bc, Sc = B.tocsc(), S.tocsc()
        M = int(rc["neighbours"])
        U = len(latents)
        self.latents = np.asarray(latents, dtype=np.int64)
        self.nbr = np.zeros((U, M), dtype=np.int64)
        self.mu, self.sd = np.zeros((U, M)), np.ones((U, M))
        self.beta, self.b = np.zeros((U, M)), np.zeros(U)
        self.gamma, self.c, self.sigma = np.zeros((U, M)), np.zeros(U), np.zeros(U)
        self.lo, self.hi = np.zeros(U), np.zeros(U)
        self.mean = np.zeros(U)
        self.marginal = np.zeros(U, dtype=bool)
        self.fire_rate = np.zeros(U)
        BT = B.T.tocsr()
        for u, j in enumerate(self.latents):
            aj = np.asarray(Sc[:, j].todense()).ravel()
            pos = aj > 0
            npos = int(pos.sum())
            self.mean[u] = aj.mean() if n else 0.0
            self.fire_rate[u] = npos / max(n, 1)
            bj = np.asarray(Bc[:, j].todense()).ravel()
            co = BT @ bj
            num = n * co - f[j] * f
            den = np.sqrt(np.maximum(f[j] * (n - f[j]) * f * (n - f), 1e-12))
            phi = np.where(f > 0, num / den, 0.0)
            phi[j] = 0.0
            nb = np.argsort(-np.abs(phi), kind="stable")[:M]
            self.nbr[u] = nb
            logs = np.log(aj[pos]) if npos else np.zeros(0)
            self.lo[u], self.hi[u] = (logs.min(), logs.max()) if npos else (0.0, 0.0)
            if npos < rc["min_positive"] or n - npos < rc["min_positive"]:
                self.marginal[u] = True
                r = min(max(npos / max(n, 1), 1e-6), 1 - 1e-6)
                self.b[u] = math.log(r / (1 - r)) if npos else -30.0
                self.c[u] = logs.mean() if npos else 0.0
                self.sigma[u] = logs.std() if npos > 1 else 0.0
                continue
            F = np.log1p(np.asarray(Sc[:, nb].todense()))
            mu, sd = F.mean(0), F.std(0)
            sd = np.where(sd > 1e-8, sd, 1.0)
            Fs = (F - mu) / sd
            self.mu[u], self.sd[u] = mu, sd
            self.beta[u], self.b[u] = _logistic_newton(Fs, pos.astype(np.float64), rc["logistic_ridge"])
            Fp = Fs[pos]
            Xb = np.hstack([Fp, np.ones((npos, 1))])
            R = rc["size_ridge"] * np.eye(M + 1)
            R[-1, -1] = 0.0
            w = np.linalg.solve(Xb.T @ Xb + R, Xb.T @ logs)
            self.gamma[u], self.c[u] = w[:-1], w[-1]
            self.sigma[u] = float(np.std(logs - Xb @ w))
        self.seconds = time.time() - t0

    def to(self, device):
        f = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
        self.t = {"nbr": f(self.nbr, torch.long), "mu": f(self.mu), "sd": f(self.sd), "beta": f(self.beta), "b": f(self.b),
                  "gamma": f(self.gamma), "c": f(self.c), "sigma": f(self.sigma), "lo": f(self.lo), "hi": f(self.hi),
                  "mean": f(self.mean)}
        return self

    @torch.no_grad()
    def draw(self, Aq: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """New activations (U, Q) for every latent of the bank on one input's Q query tokens; Aq (Q, d_sae)."""
        t = self.t
        U, M = t["nbr"].shape
        Q = Aq.shape[0]
        F = torch.log1p(Aq[:, t["nbr"].reshape(-1)].reshape(Q, U, M).permute(1, 0, 2))
        F = (F - t["mu"][:, None, :]) / t["sd"][:, None, :]
        p = torch.sigmoid(torch.einsum("uqm,um->uq", F, t["beta"]) + t["b"][:, None])
        u = torch.rand((U, Q), generator=gen, device=Aq.device)
        z = torch.randn((U, Q), generator=gen, device=Aq.device)
        logsz = torch.einsum("uqm,um->uq", F, t["gamma"]) + t["c"][:, None] + t["sigma"][:, None] * z
        size = torch.exp(torch.minimum(torch.maximum(logsz, t["lo"][:, None]), t["hi"][:, None]))
        return torch.where(u < p, size, torch.zeros_like(size))

    def summary(self) -> dict:
        return {"latents": int(len(self.latents)), "marginal": int(self.marginal.sum()),
                "median_firing_rate": float(np.median(self.fire_rate)), "fit_seconds": round(self.seconds, 1)}


# ---------------------------------------------------------------------------
# e-values and e-BH
# ---------------------------------------------------------------------------

def betting_log_evalues(X: np.ndarray, delta: float, cap: float, prior_var: float) -> np.ndarray:
    """Log wealth after the last observation of the bet against H: E[x] <= delta, for x in [-1, 1] (axis -2 is time,
    axis -1 the hypotheses, any leading axes are independent replicates). Wealth_t = prod_s (1 + lambda_s (x_s -
    delta)) with lambda_s >= 0 a function of x_1..x_{s-1} only (aGRAPA: the growth-rate-optimal bet for the running
    mean and variance, shrunk towards delta, capped at cap / (1 + delta) so a factor is never below 1 - cap).
    Under H the wealth is a nonnegative supermartingale, so its value at the fixed n is an e-value
    (Waudby-Smith & Ramdas 2024)."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[-2]
    shape = X.shape[:-2] + X.shape[-1:]
    logw, sx, sxx = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    lam_max = cap / (1.0 + delta)
    for t in range(n):
        mu = (delta + sx) / (t + 1)
        var = (prior_var + sxx - (sx * sx / t if t else 0.0)) / (t + 1)
        d = mu - delta
        lam = np.clip(d / (np.maximum(var, 1e-12) + d * d), 0.0, lam_max)
        x = X[..., t, :]
        logw += np.log1p(lam * (x - delta))
        sx += x
        sxx += x * x
    return logw


def ebh(e: np.ndarray, alpha: float) -> np.ndarray:
    """e-BH (Wang & Ramdas 2022) over the last axis: reject the k* largest e-values, k* = max{k : e_(k) >= K / (alpha
    k)}. FDR <= alpha under any dependence between the e-values. Returns a boolean mask."""
    e = np.asarray(e, dtype=np.float64)
    K = e.shape[-1]
    order = np.argsort(-e, axis=-1, kind="stable")
    es = np.take_along_axis(e, order, axis=-1)
    ok = es >= K / (alpha * np.arange(1, K + 1))
    kstar = np.where(ok.any(-1), K - np.argmax(ok[..., ::-1], axis=-1), 0)       # largest k with e_(k) >= K/(alpha k)
    rank = np.argsort(order, axis=-1)                                             # rank of each hypothesis
    return rank < np.asarray(kstar)[..., None]


def fdp_tpr(sel: np.ndarray, true: np.ndarray, keep: np.ndarray | None = None):
    """FDP and TPR of selections sel (..., K) against truth (K,); keep (K,) drops hypotheses (oracle band)."""
    if keep is not None:
        sel, true = sel[..., keep], true[keep]
    ns = sel.sum(-1)
    fd = (sel & ~true).sum(-1)
    td = (sel & true).sum(-1)
    return fd / np.maximum(ns, 1), td / max(int(true.sum()), 1), ns


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------

class Ctx:
    def __init__(self, cfg: dict, tname: str, limit: int | None = None):
        self.cfg, self.task = cfg, tname
        self.s5 = cfg["stage5"]
        self.tc = self.s5["tasks"][tname]
        self.ti = list(self.s5["tasks"]).index(tname)
        self.out = results_dir(cfg) / "stage5" / tname
        self.out.mkdir(parents=True, exist_ok=True)
        self.texts, self.y, self.classes = task_texts(cfg, tname)
        self.splits = make_splits(cfg, tname, self.y, limit)
        self.limit = limit
        self.runner = self.prompts = None

    def attach(self, tok, model, sae):
        d = self.splits["demo"]
        self.prompts = Prompts(tok, self.tc, self.classes, [self.texts[i] for i in d], self.y[d], self.s5["max_demo_tokens"])
        self.runner = Runner(model, sae, self.cfg["model"]["layer"], self.prompts.label_ids)
        return self

    def path(self, name: str) -> Path:
        return self.out / name

    def load_json(self, name: str) -> dict:
        p = self.path(name)
        if not p.exists():
            raise SystemExit(f"{p} is missing: run the earlier stage first")
        return json.loads(p.read_text())

    def write_json(self, name: str, obj) -> None:
        self.path(name).write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def require_gate(ctx: Ctx) -> dict:
    pilot = ctx.load_json("pilot.json")
    if not pilot["gate"]["passed"]:
        raise SystemExit(f"{ctx.task}: the pilot gate failed ({pilot['gate']}); this task-model pair is not run")
    return pilot


# ---------------------------------------------------------------------------
# stage: pilot
# ---------------------------------------------------------------------------

def stage_pilot(ctx: Ctx) -> dict:
    s5, R = ctx.s5, ctx.runner
    t0 = time.time()
    # (1) replay equivalence, with no intervention and with a zero ablation of the query's most active latent
    eq = []
    for i in ctx.splits["pilot"][:3]:
        ids, q0 = ctx.prompts.build(ctx.texts[i])
        h, caps = R.base(ids)
        with torch.no_grad():
            p_rep = torch.softmax(R.replay(h[None], caps), -1)[0]
        p_full = R.full_probs(ids)
        A = R.encode(h)
        j = int(A[q0:].sum(0).argmax())
        delta = torch.zeros_like(h)
        delta[q0:] = -A[q0:, j:j + 1].to(h.dtype) * R.W_dec[j][None].to(h.dtype)
        with torch.no_grad():
            p_rep2 = torch.softmax(R.replay((h + delta)[None], caps), -1)[0]
        p_full2 = R.full_probs(ids, delta)
        eq.append({"input": int(i), "latent": j, "no_intervention": float((p_rep - p_full).abs().max()),
                   "zero_ablation": float((p_rep2 - p_full2).abs().max()),
                   "ablation_effect": float((p_full - p_full2).abs().max())})
    eq_max = max(max(e["no_intervention"], e["zero_ablation"]) for e in eq)
    eq_ok = eq_max <= s5["gate"]["equivalence_atol"]
    print(f"  equivalence (replay vs model): max |dp| = {eq_max:.2e} -> {'ok' if eq_ok else 'FAILED'}", flush=True)
    # (2) prompted accuracy and the metric's spread; (3) timing
    m, pred, t_base, t_rep = [], [], [], []
    C = s5["replay_batch"]
    for k, i in enumerate(ctx.splits["pilot"]):
        ids, q0 = ctx.prompts.build(ctx.texts[i])
        ta = time.time()
        h, caps = R.base(ids)
        _sync(R.device)
        tb = time.time()
        with torch.no_grad():
            lg = R.replay(h[None].expand(C, -1, -1).contiguous(), caps)[:1]
        _sync(R.device)
        t_base.append(tb - ta)
        t_rep.append(time.time() - tb)
        m.append(float(correct_prob(lg, ctx.y[i])[0]))
        pred.append(int(lg[0].argmax()))
    m, pred = np.asarray(m), np.asarray(pred)
    yp = ctx.y[ctx.splits["pilot"]]
    recall = {ctx.classes[c]: float((pred[yp == c] == c).mean()) for c in np.unique(yp)}
    bal_acc = float(np.mean(list(recall.values())))
    sd = float(m.std(ddof=1))
    deltas = {str(f): f * sd for f in s5["delta"]["sd_fraction_grid"]}
    delta = s5["delta"]["sd_fraction"] * sd
    gate_ok = bool(bal_acc >= s5["gate"]["min_balanced_accuracy"] and eq_ok)
    # projected cost of the delta stage: one base pass per input, one replay batch per C latents
    U_est = s5["candidates"] + 2 * s5["baseline_cap"]
    n_inputs = len(ctx.splits["B"]) + len(ctx.splits["O"])
    sens = len(s5["interventions"]["sensitivity"]) * 2 * min(s5["interventions"]["sensitivity_pool"], len(ctx.splits["B"]))
    hours = (n_inputs * (np.median(t_base) + math.ceil(U_est / C) * np.median(t_rep))
             + sens * math.ceil(s5["candidates"] / C) * np.median(t_rep)) / 3600
    ids, q0 = ctx.prompts.build(ctx.texts[ctx.splits["pilot"][0]])
    out = {"task": ctx.task, "model": ctx.cfg["model"]["name"], "n": int(len(m)), "classes": ctx.classes,
           "label_words": ctx.prompts.words, "label_ids": ctx.prompts.label_ids,
           "balanced_accuracy": bal_acc, "recall_by_class": recall, "mean_m": float(m.mean()), "sd_m": sd,
           "delta": delta, "delta_grid": deltas, "equivalence": eq, "equivalence_max": eq_max,
           "gate": {"passed": gate_ok, "balanced_accuracy": bal_acc, "min": s5["gate"]["min_balanced_accuracy"],
                    "equivalence_ok": bool(eq_ok)},
           "seconds_per_base": float(np.median(t_base)), "seconds_per_replay_batch": float(np.median(t_rep)),
           "projected_delta_hours": float(hours), "prompt_tokens_median": None,
           "example_prompt": ctx.prompts.tok.decode(ids), "query_start": int(q0),
           "splits": {k: int(len(v)) for k, v in ctx.splits.items()}, "minutes": round((time.time() - t0) / 60, 2)}
    out["prompt_tokens_median"] = float(np.median([len(ctx.prompts.build(ctx.texts[i])[0]) for i in ctx.splits["pilot"]]))
    ctx.write_json("pilot.json", out)
    print(f"  {ctx.task}: balanced accuracy {bal_acc:.3f} (gate {s5['gate']['min_balanced_accuracy']}), SD(m) {sd:.4f}, "
          f"delta {delta:.5f}; gate {'PASSED' if gate_ok else 'FAILED'}; projected delta stage {hours:.1f} h", flush=True)
    return out


def _sync(device):
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# stage: screen
# ---------------------------------------------------------------------------

def stage_screen(ctx: Ctx) -> dict:
    require_gate(ctx)
    s5, R = ctx.s5, ctx.runner
    d_sae = R.W_dec.shape[0]
    score = torch.zeros(d_sae, dtype=torch.float64, device=R.device)
    fire = torch.zeros(d_sae, dtype=torch.float64, device=R.device)
    indptr, idx, val, ntok = [0], [], [], 0
    m_all = []
    t0 = time.time()
    rows = ctx.splits["A"]
    for k, i in enumerate(rows):
        ids, q0 = ctx.prompts.build(ctx.texts[i])
        h, caps = R.base(ids)
        hg = h.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            mm = correct_prob(R.replay(hg[None], caps), ctx.y[i])[0]
            mm.backward()
        g = hg.grad[q0:].float()
        A = R.encode(h)[q0:]                                       # (Q, d_sae)
        proj = g @ R.W_dec.float().T                               # (Q, d_sae): d m / d a_{t,j} at fixed error term
        score += (A * proj).sum(0).double()                        # first-order drop of m when a_j is zeroed
        fire += (A > 0).sum(0).double()
        nz = torch.nonzero(A > 0)                                  # sparse query-token activations, row-major
        for t in range(A.shape[0]):
            cols = nz[nz[:, 0] == t, 1]
            idx.append(cols.cpu().numpy().astype(np.int32))
            val.append(A[t, cols].cpu().numpy().astype(np.float32))
            indptr.append(indptr[-1] + len(cols))
        ntok += A.shape[0]
        m_all.append(float(mm.detach()))
        if (k + 1) % 100 == 0:
            print(f"  screen {k + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    score = (score / len(rows)).cpu().numpy()
    K = s5["candidates"]
    order = np.argsort(-score, kind="stable")
    cand = order[:K][score[order[:K]] > 0]
    np.savez(ctx.path("screen.npz"), score=score, candidates=cand, fire_tokens=fire.cpu().numpy(),
             indptr=np.asarray(indptr, dtype=np.int64), idx=np.concatenate(idx) if idx else np.zeros(0, np.int32),
             val=np.concatenate(val) if val else np.zeros(0, np.float32), n_tokens=ntok, m=np.asarray(m_all))
    out = {"n_inputs": int(len(rows)), "n_tokens": int(ntok), "candidates": int(len(cand)),
           "positive_scores": int((score > 0).sum()), "top_scores": score[order[:10]].tolist(),
           "minutes": round((time.time() - t0) / 60, 2)}
    ctx.write_json("screen.json", out)
    print(f"  screen: {len(cand)} candidates from {len(rows)} inputs / {ntok} query tokens ({out['minutes']} min)", flush=True)
    return out


# ---------------------------------------------------------------------------
# stage: baselines
# ---------------------------------------------------------------------------

def _obs_cache(ctx: Ctx):
    """(X, retained SAE ids) of the task's pooled-latent cache, or None if this model has no cache for it."""
    if ctx.tc["source"] == "sst2":
        p = cache_path(ctx.cfg)
        if not p.exists():
            return None
        d = np.load(p, allow_pickle=True)
        X = d[f"X_{ctx.cfg['aggregation']['primary']}"]
    else:
        sys.path.insert(0, str(ROOT / "src" / "saebench"))
        from sbutil import cache_path as sb_cache_path
        p = sb_cache_path(ctx.cfg, ctx.tc["dataset"])
        if not p.exists():
            return None
        d = np.load(p, allow_pickle=True)
        X = d["X"]
    if X.shape[0] != len(ctx.texts):
        raise SystemExit(f"{p}: {X.shape[0]} rows, the task has {len(ctx.texts)} texts")
    return np.asarray(X, dtype=np.float32), np.asarray(d["retained_idx"], dtype=np.int64), str(p)


def stage_baselines(ctx: Ctx) -> dict:
    require_gate(ctx)
    s5 = ctx.s5
    scr = np.load(ctx.path("screen.npz"))
    score, cand = scr["score"], scr["candidates"]
    order = np.argsort(-score, kind="stable")
    out = {"atp_topk": {str(k): order[:k][score[order[:k]] > 0].tolist() for k in s5["atp_topk"]}, "observational": {}}
    alphas = s5["alphas"]
    ob = _obs_cache(ctx)
    if ob is None:
        out["observational_note"] = "no pooled-latent cache for this model and task: observational baselines not run"
    else:
        X, retained, cpath = ob
        rows = ctx.splits["obs"]
        X = X[rows].astype(np.float64)
        yv = ctx.y[rows]
        mu, sd = X.mean(0), X.std(0)
        keep = sd > 1e-8
        Z = (X[:, keep] - mu[keep]) / sd[keep]
        ids = retained[keep]
        classes = range(len(ctx.classes)) if len(ctx.classes) > 2 else [1]
        sel = {"knockoff": {str(q): set() for q in alphas}, "evalue_obs": {str(q): set() for q in alphas}}
        from sklearn.linear_model import LogisticRegression
        from knockpy.knockoffs import GaussianSampler
        from planted_fdr import compute_knockoff_plus_threshold
        from stage4_repairs import evalue_discoveries
        ev, lasso = ctx.cfg["stage4_repairs"]["evalue"], ctx.cfg["stage4_repairs"]["lasso"]
        dev = ctx.runner.device if ctx.runner is not None else torch.device("cpu")
        Zt = torch.as_tensor(Z, dtype=torch.float32, device=dev)
        t0 = time.time()
        for ci, c in enumerate(classes):
            yb = (yv == c).astype(np.float64)
            np.random.seed(int(stream_rng(ctx.cfg, "s5_obs", ctx.ti, 0, ci).integers(2**31)))
            Zk = GaussianSampler(X=Z, method="equicorrelated").sample_knockoffs()
            fit = LogisticRegression(penalty="l1", C=s5["observational"]["knockoff_C"], solver="liblinear", max_iter=2000)
            fit.fit(np.hstack([Z, Zk]), yb)
            w = np.abs(fit.coef_[0])
            W = w[: Z.shape[1]] - w[Z.shape[1]:]
            for q in alphas:
                tau = compute_knockoff_plus_threshold(W, q)
                if np.isfinite(tau):
                    sel["knockoff"][str(q)] |= set(ids[W >= tau].tolist())
            split = stream_rng(ctx.cfg, "s5_obs", ctx.ti, 1, ci).permutation(len(yb))
            disc = evalue_discoveries(Zt, torch.as_tensor(yb, dtype=torch.float32, device=dev), split, ev, lasso, alphas)
            for q in alphas:
                sel["evalue_obs"][str(q)] |= set(ids[sorted(disc[q])].tolist())
        cap = s5["baseline_cap"]
        for name, by_q in sel.items():
            union = sorted(set().union(*by_q.values()))
            sub = union
            if len(union) > cap:
                r = stream_rng(ctx.cfg, "s5_obs", ctx.ti, 2, list(sel).index(name))
                sub = sorted(r.choice(union, size=cap, replace=False).tolist())
            out["observational"][name] = {"selected": {q: sorted(v) for q, v in by_q.items()},
                                          "n_selected": {q: len(v) for q, v in by_q.items()},
                                          "evaluated": sub, "subsampled": len(union) > cap}
        out["observational_cache"] = cpath
        out["observational_rows"] = int(len(rows))
        out["observational_minutes"] = round((time.time() - t0) / 60, 2)
    U = set(cand.tolist())
    for v in out["atp_topk"].values():
        U |= set(v)
    for v in out["observational"].values():
        U |= set(v["evaluated"])
    cand_list = cand.tolist()
    out["U"] = cand_list + sorted(U - set(cand_list))              # candidates first, in screening order
    out["n_candidates"] = len(cand_list)
    ctx.write_json("baselines.json", out)
    print(f"  baselines: U = {len(out['U'])} latents ({len(cand_list)} candidates); "
          + "; ".join(f"{k} {v['n_selected']}" for k, v in out["observational"].items()), flush=True)
    return out


# ---------------------------------------------------------------------------
# stage: delta
# ---------------------------------------------------------------------------

def _new_values(kind: str, sub: torch.Tensor, a_cur: torch.Tensor, Aq: torch.Tensor, bank: ResampleBank, gen):
    """New activations (len(sub), Q) of the latents at bank positions sub."""
    if kind == "zero":
        return torch.zeros_like(a_cur)
    if kind == "mean":
        return bank.t["mean"][sub][:, None].expand_as(a_cur).clone()
    if kind == "resample":
        return bank.draw(Aq, gen)[sub]
    raise ValueError(kind)


def stage_delta(ctx: Ctx) -> dict:
    require_gate(ctx)
    s5, R = ctx.s5, ctx.runner
    bl = ctx.load_json("baselines.json")
    U = np.asarray(bl["U"], dtype=np.int64)
    Kc = bl["n_candidates"]
    scr = np.load(ctx.path("screen.npz"))
    bank = ResampleBank({k: scr[k] for k in ("indptr", "idx", "val")}, U, R.W_dec.shape[0], s5["resample"]).to(R.device)
    print(f"  resampling sampler: {bank.summary()}", flush=True)
    W_U = R.W_dec[torch.as_tensor(U, device=R.device)]
    prim = s5["interventions"]["primary"]
    sens = list(s5["interventions"]["sensitivity"])
    n_sens = s5["interventions"]["sensitivity_pool"]
    C = s5["replay_batch"]
    pools = [("B", ctx.splits["B"]), ("O", ctx.splits["O"])]
    ck = ctx.path("delta_checkpoint.npz")
    res = {}
    for pname, rows in pools:
        res[pname] = {"m0": np.full(len(rows), np.nan), "pred": np.full(len(rows), -1), "y": ctx.y[rows],
                      prim: np.full((len(rows), len(U)), np.nan, dtype=np.float32)}
        for kind in sens:
            res[pname][kind] = np.full((min(n_sens, len(rows)), Kc), np.nan, dtype=np.float32)
    done = {"B": 0, "O": 0}
    if ck.exists():
        z = np.load(ck, allow_pickle=True)
        if list(z["U"]) == list(U):
            for pname in res:
                for k in res[pname]:
                    res[pname][k] = z[f"{pname}__{k}"]
            done = json.loads(str(z["done"]))
            print(f"  resuming: {done}", flush=True)
    t0 = time.time()
    last_ck = time.time()
    for pi, (pname, rows) in enumerate(pools):
        for k in range(done[pname], len(rows)):
            i = int(rows[k])
            ids, q0 = ctx.prompts.build(ctx.texts[i])
            h, caps = R.base(ids)
            A = R.encode(h)
            Aq = A[q0:]
            a_cur = Aq[:, torch.as_tensor(U, device=R.device)].T.contiguous()        # (U, Q)
            with torch.no_grad():
                lg0 = R.replay(h[None], caps)
            m0 = float(correct_prob(lg0, ctx.y[i])[0])
            res[pname]["m0"][k], res[pname]["pred"][k] = m0, int(lg0[0].argmax())
            gen = torch.Generator(device=R.device)
            gen.manual_seed(int(stream_rng(ctx.cfg, "s5_resample", ctx.ti, pi, k).integers(2**62)))
            kinds = [(prim, np.arange(len(U)))] + ([(s, np.arange(Kc)) for s in sens] if k < n_sens else [])
            for kind, sub in kinds:
                sub_t = torch.as_tensor(sub, device=R.device)
                new = _new_values(kind, sub_t, a_cur[sub_t], Aq, bank, gen)
                d = (new - a_cur[sub_t]).to(h.dtype)                               # (S, Q)
                out = np.empty(len(sub), dtype=np.float32)
                for s0 in range(0, len(sub), C):
                    dd = d[s0: s0 + C]
                    H = h[None].repeat(len(dd), 1, 1)
                    H[:, q0:, :] += dd[:, :, None] * W_U[sub_t[s0: s0 + C]][:, None, :].to(h.dtype)
                    with torch.no_grad():
                        mm = correct_prob(R.replay(H, caps), ctx.y[i])
                    out[s0: s0 + len(dd)] = (m0 - mm).cpu().numpy()
                res[pname][kind][k, : len(sub)] = out
            done[pname] = k + 1
            if time.time() - last_ck > 600 or k + 1 == len(rows):
                _save_ck(ck, res, U, done)
                last_ck = time.time()
                el = time.time() - t0
                print(f"  delta {pname} {k + 1}/{len(rows)}  ({el / 60:.1f} min)", flush=True)
    np.savez(ctx.path("delta.npz"), U=U, n_candidates=Kc,
             **{f"{p}__{k}": v for p in res for k, v in res[p].items()})
    ck.unlink(missing_ok=True)
    out = {"U": int(len(U)), "n_candidates": int(Kc), "pools": {p: int(len(r)) for p, r in pools},
           "primary": prim, "sensitivity": sens, "sensitivity_pool": n_sens, "resample": bank.summary(),
           "accuracy": {p: float((res[p]["pred"] == res[p]["y"]).mean()) for p in res},
           "minutes": round((time.time() - t0) / 60, 1)}
    ctx.write_json("delta.json", out)
    print(f"  delta: done ({out['minutes']} min)", flush=True)
    return out


def _save_ck(path: Path, res: dict, U, done: dict) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, U=np.asarray(U), done=np.asarray(json.dumps(done)),
             **{f"{p}__{k}": v for p in res for k, v in res[p].items()})
    tmp.replace(path)


# ---------------------------------------------------------------------------
# stage: evaluate
# ---------------------------------------------------------------------------

def _procedure(DB: np.ndarray, delta: float, ns, R: int, alphas, ev: dict, rng_of, chunk: int = 25):
    """Rerun the procedure: for each n, R subsamples of the split-B rows (random order), betting e-values against
    tau_j <= delta for every column, e-BH at each alpha. Returns {n: {alpha: selections (R, K) bool}}."""
    out = {}
    for n in ns:
        if n > DB.shape[0]:
            continue
        sels = {a: [] for a in alphas}
        for r0 in range(0, R, chunk):
            idx = np.stack([rng_of(n, r).choice(DB.shape[0], size=n, replace=False) for r in range(r0, min(R, r0 + chunk))])
            le = betting_log_evalues(DB[idx], delta, ev["bet_cap"], ev["prior_variance"])      # (r, K)
            e = np.exp(np.minimum(le, 700.0))
            for a in alphas:
                sels[a].append(ebh(e, a))
        out[n] = {a: np.concatenate(v) for a, v in sels.items()}
    return out


def _summ(sel: np.ndarray, true: np.ndarray, keep=None, alpha: float | None = None) -> dict:
    fdp, tpr, ns = fdp_tpr(sel, true, keep)
    R = len(fdp)
    fdr, se = float(fdp.mean()), float(fdp.std(ddof=1) / math.sqrt(R)) if R > 1 else 0.0
    d = {"fdr": fdr, "fdr_se": se, "power": float(tpr.mean()), "mean_selected": float(ns.mean()),
         "p_any_false": float((fdp > 0).mean())}
    if alpha is not None:
        d["breach"] = bool(fdr - alpha > 2 * se)
    return d


def stage_evaluate(ctx: Ctx) -> dict:
    pilot = require_gate(ctx)
    s5 = ctx.s5
    bl = ctx.load_json("baselines.json")
    z = np.load(ctx.path("delta.npz"))
    U = z["U"]
    Kc = int(z["n_candidates"])
    prim = s5["interventions"]["primary"]
    alphas, ns, R, ev = s5["alphas"], s5["test_sizes"], s5["replicates"], s5["evalue"]
    rng_of = lambda tag: (lambda n, r: stream_rng(ctx.cfg, "s5_order", ctx.ti, tag, n, r))
    delta0 = pilot["delta"]
    out = {"task": ctx.task, "model": ctx.cfg["model"]["name"], "delta": delta0, "U": int(len(U)), "candidates": Kc,
           "alphas": alphas, "test_sizes": ns, "replicates": R, "interventions": {}}
    pos = {int(j): k for k, j in enumerate(U)}
    for ii, kind in enumerate([prim] + list(s5["interventions"]["sensitivity"])):
        DB, DO = z[f"B__{kind}"], z[f"O__{kind}"]
        cols = np.arange(DB.shape[1])                       # primary: U (candidates first); sensitivity: candidates
        tau, se = DO.mean(0), DO.std(0, ddof=1) / math.sqrt(DO.shape[0])
        res = {"n_B": int(DB.shape[0]), "n_O": int(DO.shape[0]), "oracle": {
            "tau": tau.tolist(), "se": se.tolist(), "n_true_candidates": int((tau[:Kc] > delta0).sum()),
            "n_true_U": int((tau > delta0).sum()), "max_tau": float(tau.max()), "median_se": float(np.median(se))}}
        # the procedure at the primary delta and over the delta grid; the zero-null baseline (delta = 0)
        grid = {"primary": delta0, **{f"sd_{k}": v for k, v in pilot["delta_grid"].items()}, "zero_null": 0.0}
        for gname, dl in grid.items():
            sels = _procedure(DB[:, :Kc], dl, ns, R, alphas, ev, rng_of(10 * ii + list(grid).index(gname)))
            true = tau[:Kc] > delta0 if gname == "zero_null" else tau[:Kc] > dl
            band = np.abs(tau[:Kc] - (delta0 if gname == "zero_null" else dl)) >= s5["oracle_band_se"] * se[:Kc]
            r = {}
            for n, by_a in sels.items():
                r[str(n)] = {str(a): {"vs_oracle": _summ(s, true, alpha=a if gname != "zero_null" else None),
                                      "vs_oracle_band": _summ(s, true, band, alpha=a if gname != "zero_null" else None)}
                             for a, s in by_a.items()}
                if gname == "zero_null":                 # against its own estimand (tau_j > 0)
                    for a, s in by_a.items():
                        r[str(n)][str(a)]["vs_tau_positive"] = _summ(s, tau[:Kc] > 0, alpha=a)
            res[gname] = r
        if ii == 0:
            # fixed baselines against the same oracle (primary intervention only: their latents are in U)
            true_U = tau > delta0
            base = {}
            for k, lat in bl["atp_topk"].items():
                m = np.zeros(len(U), dtype=bool)
                m[[pos[int(j)] for j in lat]] = True
                base[f"atp_top{k}"] = _summ(m[None], true_U)
            for name, v in bl.get("observational", {}).items():
                ev_set = {int(j) for j in v["evaluated"]}
                for q, lat in v["selected"].items():
                    lat = [int(j) for j in lat if int(j) in ev_set]
                    m = np.zeros(len(U), dtype=bool)
                    m[[pos[j] for j in lat]] = True
                    s = _summ(m[None], true_U)
                    s.update({"n_selected": v["n_selected"][q], "evaluated_subsample": v["subsampled"],
                              "fdp_vs_tau_positive": _summ(m[None], tau > 0)["fdr"]})
                    base[f"{name}_q{q}"] = s
            res["baselines"] = base
            # overlap with the candidates' selections at the largest n, alpha 0.10, kept for the sensitivity comparison
        out["interventions"][kind] = res
    # agreement across interventions: the share of replicates' selections in common at the largest common n, alpha 0.1
    ctx.write_json("evaluate.json", out)
    ctx.write_json("evaluate_summary.json", summarise(out, alphas))
    print(json.dumps(summarise(out, alphas), indent=1)[:3000], flush=True)
    return out


def summarise(ev: dict, alphas) -> dict:
    """The headline numbers: per intervention, realised FDR and power at alpha 0.10 for every n, breaches, the
    zero-null baseline's FDR against the minimum-effect truth, and the fixed baselines."""
    s = {"task": ev["task"], "model": ev["model"], "delta": ev["delta"]}
    a = "0.1" if 0.1 in alphas else str(alphas[0])
    for kind, r in ev["interventions"].items():
        d = {"n_true_candidates": r["oracle"]["n_true_candidates"], "n_true_U": r["oracle"]["n_true_U"],
             "primary": {n: {k: round(v[a]["vs_oracle"][k], 4) for k in ("fdr", "fdr_se", "power", "mean_selected")}
                         for n, v in r["primary"].items()},
             "breaches_primary": [f"n{n}_a{al}" for n, v in r["primary"].items() for al, w in v.items()
                                  if w["vs_oracle"].get("breach")],
             "zero_null": {n: {"fdr_vs_min_effect": round(v[a]["vs_oracle"]["fdr"], 4),
                               "fdr_vs_tau_positive": round(v[a]["vs_tau_positive"]["fdr"], 4),
                               "mean_selected": round(v[a]["vs_oracle"]["mean_selected"], 1)} for n, v in r["zero_null"].items()}}
        if "baselines" in r:
            d["baselines"] = {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                              for k, v in r["baselines"].items()}
        s[kind] = d
    return s


# ---------------------------------------------------------------------------
# stage: summary over tasks
# ---------------------------------------------------------------------------

def stage_summary(cfg: dict) -> dict:
    base = results_dir(cfg) / "stage5"
    out = {"model": cfg["model"]["name"], "tasks": {}}
    for t in cfg["stage5"]["tasks"]:
        d = base / t
        if (d / "evaluate_summary.json").exists():
            out["tasks"][t] = json.loads((d / "evaluate_summary.json").read_text())
        elif (d / "pilot.json").exists():
            p = json.loads((d / "pilot.json").read_text())
            out["tasks"][t] = {"gate": p["gate"], "status": "gate failed" if not p["gate"]["passed"] else "incomplete"}
    (base / "stage5_summary.json").write_text(json.dumps(out, indent=2, default=_json_default))
    print(f"wrote {base / 'stage5_summary.json'}")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 5: interventional FDR control for SAE latents")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--task", default=None, help="one of stage5.tasks (not needed for --stage summary)")
    ap.add_argument("--stage", default="all", choices=list(STAGES) + ["all", "summary"])
    ap.add_argument("--limit", type=int, default=None, help="debug: cap every split at N inputs (results not valid)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.stage == "summary":
        stage_summary(cfg)
        return
    if args.task not in cfg["stage5"]["tasks"]:
        raise SystemExit(f"--task must be one of {list(cfg['stage5']['tasks'])}")
    torch.manual_seed(int(stream_rng(cfg, "s5_resample", 999).integers(2**31)))
    ctx = Ctx(cfg, args.task, args.limit)
    if args.limit:
        ctx.out = ctx.out.parent / f"{args.task}__limit{args.limit}"
        ctx.out.mkdir(parents=True, exist_ok=True)
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    done_file = {"pilot": "pilot.json", "screen": "screen.json", "baselines": "baselines.json",
                 "delta": "delta.json", "evaluate": "evaluate.json"}
    todo = [s for s in stages if args.stage != "all" or not ctx.path(done_file[s]).exists()]
    need_model = [s for s in todo if s in ("pilot", "screen", "baselines", "delta")]
    if need_model:
        from cache_activations import load_model_and_sae
        tok, model, sae = load_model_and_sae(cfg)
        ctx.attach(tok, model, sae)
    print(f"stage 5 | {cfg['model']['name']} | task {args.task} | stages {todo} | out {ctx.out}", flush=True)
    fns = {"pilot": stage_pilot, "screen": stage_screen, "baselines": stage_baselines, "delta": stage_delta,
           "evaluate": stage_evaluate}
    for s in todo:
        print(f"\n=== {s} ===", flush=True)
        r = fns[s](ctx)
        if s == "pilot" and not r["gate"]["passed"]:
            print(f"  gate failed: {args.task} stops here (reported as not run)", flush=True)
            return


if __name__ == "__main__":
    main()
