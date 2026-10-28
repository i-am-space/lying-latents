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

stage5_amendment_2 (new outputs; the stages above and their files are unchanged):
  baselines2 observational baselines over several knockoff draws / e-value splits, each class at alpha / C
  delta2     Delta_j(x) (primary intervention) for the latents baselines2 selects that are not in U
  evaluate2  CPU only: replicates re-split B + O into test and oracle inputs; the pre-registered specification
             (K = 250, bet cap 0.5) and v2 (K = 50, bet cap 0.9); a boundary stress test (every latent's Delta
             shifted to mean delta, so every rejection is false); tau > 2 SE counts; baselines2 against the oracle
  --stage v2 runs baselines2, delta2, evaluate2

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
V2_STAGES = ("baselines2", "delta2", "evaluate2")              # stage5_amendment_2
V3_STAGES = ("baselines3", "delta3", "evaluate3")              # stage5_amendment_3
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
    def __init__(self, cfg: dict, tname: str, limit: int | None = None, light: bool = False):
        """light: no texts or splits (evaluate / evaluate2 read only the saved stage outputs)."""
        self.cfg, self.task = cfg, tname
        self.s5 = cfg["stage5"]
        self.tc = self.s5["tasks"][tname]
        self.ti = list(self.s5["tasks"]).index(tname)
        self.out = results_dir(cfg) / "stage5" / tname
        self.out.mkdir(parents=True, exist_ok=True)
        if light:
            self.texts = self.y = self.classes = self.splits = None
        else:
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
# stage5_amendment_2: baselines2, delta2, evaluate2
# ---------------------------------------------------------------------------

def stage_baselines2(ctx: Ctx) -> dict:
    """The observational baselines over several independent knockoff draws and e-value splits, each class at
    alpha / C (the union's FDP is at most the sum of the per-class FDPs, so its FDR is <= alpha). Latents selected
    in any draw are evaluated (at most baseline_cap per baseline, a random subsample if more); those not already
    in U are listed in "extra" for delta2."""
    require_gate(ctx)
    s5, oc = ctx.s5, ctx.s5["observational"]
    U = {int(j) for j in ctx.load_json("baselines.json")["U"]}
    ob = _obs_cache(ctx)
    if ob is None:
        out = {"observational_note": "no pooled-latent cache for this model and task: observational baselines not run",
               "extra": []}
        ctx.write_json("baselines2.json", out)
        print(f"  baselines2: {out['observational_note']}", flush=True)
        return out
    X, retained, cpath = ob
    rows = ctx.splits["obs"]
    X = X[rows].astype(np.float64)
    yv = ctx.y[rows]
    mu, sd = X.mean(0), X.std(0)
    keep = sd > 1e-8
    Z = (X[:, keep] - mu[keep]) / sd[keep]
    ids = retained[keep]
    p = Z.shape[1]
    classes = range(len(ctx.classes)) if len(ctx.classes) > 2 else [1]
    C = len(classes)
    alphas = s5["alphas"]
    qc = {q: (q / C if oc.get("per_class_alpha", True) else q) for q in alphas}
    D = int(oc["draws"])
    from sklearn.linear_model import LogisticRegression
    from knockpy.knockoffs import GaussianSampler
    from planted_fdr import compute_knockoff_plus_threshold
    from stage4_repairs import evalue_discoveries
    ev, lasso = ctx.cfg["stage4_repairs"]["evalue"], ctx.cfg["stage4_repairs"]["lasso"]
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    Zt = torch.as_tensor(Z, dtype=torch.float32, device=dev)
    sampler = GaussianSampler(X=Z, method="equicorrelated")       # knockoffs do not depend on y: one sampler
    draws = {"knockoff": [], "evalue_obs": []}
    t0 = time.time()
    for d in range(D):
        per = {name: {str(q): set() for q in alphas} for name in draws}
        for ci, c in enumerate(classes):
            yb = (yv == c).astype(np.float64)
            np.random.seed(int(stream_rng(ctx.cfg, "s5_obs", ctx.ti, 3, d, ci).integers(2**31)))
            Zk = sampler.sample_knockoffs()
            fit = LogisticRegression(penalty="l1", C=oc["knockoff_C"], solver="liblinear", max_iter=2000)
            fit.fit(np.hstack([Z, Zk]), yb)
            w = np.abs(fit.coef_[0])
            W = w[:p] - w[p:]
            for q in alphas:
                tau = compute_knockoff_plus_threshold(W, qc[q])
                if np.isfinite(tau):
                    per["knockoff"][str(q)] |= set(ids[W >= tau].tolist())
            split = stream_rng(ctx.cfg, "s5_obs", ctx.ti, 4, d, ci).permutation(len(yb))
            disc = evalue_discoveries(Zt, torch.as_tensor(yb, dtype=torch.float32, device=dev), split, ev, lasso,
                                      [qc[q] for q in alphas])
            for q in alphas:
                per["evalue_obs"][str(q)] |= set(ids[sorted(disc[qc[q]])].tolist())
        for name in draws:
            draws[name].append({q: sorted(int(j) for j in v) for q, v in per[name].items()})
        print(f"  baselines2 draw {d + 1}/{D} ({time.time() - t0:.0f}s): "
              + "; ".join(f"{k} {[len(v) for v in per[k].values()]}" for k in per), flush=True)
    cap = oc["baseline_cap"]
    out = {"classes": C, "alpha_per_class": {str(q): qc[q] for q in alphas}, "draws": D, "baselines": {}}
    for bi, (name, dl) in enumerate(draws.items()):
        union = sorted(set().union(*[set(v) for dsel in dl for v in dsel.values()]))
        sub = union
        if len(union) > cap:
            sub = sorted(stream_rng(ctx.cfg, "s5_obs", ctx.ti, 5, bi).choice(union, size=cap, replace=False).tolist())
        out["baselines"][name] = {"draws": dl, "evaluated": [int(j) for j in sub], "n_union": len(union),
                                  "subsampled": len(union) > cap}
    out["extra"] = sorted(set().union(*[set(v["evaluated"]) for v in out["baselines"].values()]) - U)
    out.update({"observational_cache": cpath, "observational_rows": int(len(rows)),
                "minutes": round((time.time() - t0) / 60, 2)})
    ctx.write_json("baselines2.json", out)
    print(f"  baselines2: {len(out['extra'])} latents not in U go to delta2 ({out['minutes']} min)", flush=True)
    return out


def stage_delta2(ctx: Ctx) -> dict:
    """Delta_j(x) under the primary intervention for baselines2's extra latents, on the same split-B and split-O
    inputs as delta (same row order), with its own resampling draws (stream s5_resample, pool index 2 / 3)."""
    require_gate(ctx)
    s5, R = ctx.s5, ctx.runner
    E = np.asarray(ctx.load_json("baselines2.json")["extra"], dtype=np.int64)
    prim = s5["interventions"]["primary"]
    pools = [("B", ctx.splits["B"]), ("O", ctx.splits["O"])]
    res = {p: np.full((len(r), len(E)), np.nan, dtype=np.float32) for p, r in pools}
    t0 = time.time()
    if len(E):
        scr = np.load(ctx.path("screen.npz"))
        bank = ResampleBank({k: scr[k] for k in ("indptr", "idx", "val")}, E, R.W_dec.shape[0], s5["resample"]).to(R.device)
        print(f"  resampling sampler: {bank.summary()}", flush=True)
        Et = torch.as_tensor(E, device=R.device)
        W_E = R.W_dec[Et]
        C = s5["replay_batch"]
        ck = ctx.path("delta2_checkpoint.npz")
        done = {"B": 0, "O": 0}
        if ck.exists():
            zc = np.load(ck, allow_pickle=True)
            if list(zc["E"]) == list(E):
                res = {p: zc[p] for p in res}
                done = json.loads(str(zc["done"]))
                print(f"  resuming: {done}", flush=True)
        last_ck = time.time()
        for pi, (pname, rows) in enumerate(pools):
            for k in range(done[pname], len(rows)):
                i = int(rows[k])
                ids, q0 = ctx.prompts.build(ctx.texts[i])
                h, caps = R.base(ids)
                Aq = R.encode(h)[q0:]
                a_cur = Aq[:, Et].T.contiguous()                                       # (E, Q)
                with torch.no_grad():
                    m0 = float(correct_prob(R.replay(h[None], caps), ctx.y[i])[0])
                gen = torch.Generator(device=R.device)
                gen.manual_seed(int(stream_rng(ctx.cfg, "s5_resample", ctx.ti, pi + 2, k).integers(2**62)))
                d = (_new_values(prim, torch.arange(len(E), device=R.device), a_cur, Aq, bank, gen) - a_cur).to(h.dtype)
                outv = np.empty(len(E), dtype=np.float32)
                for s0 in range(0, len(E), C):
                    dd = d[s0: s0 + C]
                    H = h[None].repeat(len(dd), 1, 1)
                    H[:, q0:, :] += dd[:, :, None] * W_E[s0: s0 + C][:, None, :].to(h.dtype)
                    with torch.no_grad():
                        mm = correct_prob(R.replay(H, caps), ctx.y[i])
                    outv[s0: s0 + len(dd)] = (m0 - mm).cpu().numpy()
                res[pname][k] = outv
                done[pname] = k + 1
                if time.time() - last_ck > 600 or k + 1 == len(rows):
                    tmp = ck.with_suffix(".tmp.npz")
                    np.savez(tmp, E=E, done=np.asarray(json.dumps(done)), **res)
                    tmp.replace(ck)
                    last_ck = time.time()
                    print(f"  delta2 {pname} {k + 1}/{len(rows)}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        ck.unlink(missing_ok=True)
    np.savez(ctx.path("delta2.npz"), E=E, **{f"{p}__{prim}": v for p, v in res.items()})
    out = {"extra": int(len(E)), "primary": prim, "minutes": round((time.time() - t0) / 60, 1)}
    ctx.write_json("delta2.json", out)
    print(f"  delta2: {len(E)} latents ({out['minutes']} min)", flush=True)
    return out


def _selections(X: np.ndarray, delta: float, cap: float, ev: dict, alphas) -> dict:
    """e-BH selections at each alpha from betting e-values on X (..., n, K)."""
    e = np.exp(np.minimum(betting_log_evalues(X, delta, cap, ev["prior_variance"]), 700.0))
    return {a: ebh(e, a) for a in alphas}


def _summ_rep(sel: np.ndarray, true: np.ndarray, keep: np.ndarray | None = None, alpha: float | None = None) -> dict:
    """As _summ, with a truth (and band) per replicate: sel, true, keep are (R, K). Power is averaged over the
    replicates whose oracle has at least one true hypothesis."""
    if keep is not None:
        sel, true = sel & keep, true & keep
    ns, nt = sel.sum(-1), true.sum(-1)
    fdp = (sel & ~true).sum(-1) / np.maximum(ns, 1)
    tpr = (sel & true).sum(-1) / np.maximum(nt, 1)
    R = len(fdp)
    fdr, se = float(fdp.mean()), float(fdp.std(ddof=1) / math.sqrt(R)) if R > 1 else 0.0
    d = {"fdr": fdr, "fdr_se": se, "power": float(tpr[nt > 0].mean()) if (nt > 0).any() else 0.0,
         "mean_selected": float(ns.mean()), "mean_true": float(nt.mean()), "p_any_false": float((fdp > 0).mean())}
    if alpha is not None:
        d["breach"] = bool(fdr - alpha > 2 * se)
    return d


def _eval_baselines2(ctx: Ctx, P: np.ndarray, U, delta0: float, k_se: float) -> dict:
    """baselines2's selections, draw by draw, against the oracle from all B + O inputs (the baselines never see
    them). A draw that selects nothing has FDP 0; a draw whose selections all fall outside the evaluated subsample
    is left out."""
    p2 = ctx.path("baselines2.json")
    if not p2.exists():
        return {"note": "baselines2 not run"}
    b2 = json.loads(p2.read_text())
    if "baselines" not in b2:
        return {"note": b2.get("observational_note", "")}
    prim = ctx.s5["interventions"]["primary"]
    N = P.shape[0]
    T = {int(j): (float(t), float(s)) for j, t, s in zip(U, P.mean(0), P.std(0, ddof=1) / math.sqrt(N))}
    if b2["extra"]:
        if not ctx.path("delta2.npz").exists():
            return {"note": "delta2 not run: baselines2's extra latents have no Delta yet"}
        z2 = np.load(ctx.path("delta2.npz"))
        P2 = np.vstack([z2[f"B__{prim}"], z2[f"O__{prim}"]]).astype(np.float64)
        T.update({int(j): (float(t), float(s)) for j, t, s in
                  zip(z2["E"], P2.mean(0), P2.std(0, ddof=1) / math.sqrt(len(P2)))})
    out = {}
    for name, v in b2["baselines"].items():
        ev_set = set(v["evaluated"])
        for q in v["draws"][0]:
            f_min, f_2se, f_pos, ntrue, nsel = [], [], [], [], []
            for dsel in v["draws"]:
                nsel.append(len(dsel[q]))
                lat = [j for j in dsel[q] if j in ev_set]
                if dsel[q] and not lat:
                    continue
                if not lat:                              # nothing selected: FDP 0
                    for lst in (f_min, f_2se, f_pos, ntrue):
                        lst.append(0)
                    continue
                t = np.array([T[j][0] for j in lat])
                s = np.array([T[j][1] for j in lat])
                f_min.append(float((t <= delta0).mean()))
                f_2se.append(float((t <= k_se * s).mean()))
                f_pos.append(float((t <= 0).mean()))
                ntrue.append(int((t > delta0).sum()))
            m = len(f_min)
            out[f"{name}_q{q}"] = {
                "fdr_vs_min_effect": float(np.mean(f_min)) if m else None,
                "fdr_se": float(np.std(f_min, ddof=1) / math.sqrt(m)) if m > 1 else 0.0,
                "fdr_vs_tau_above_2se": float(np.mean(f_2se)) if m else None,
                "fdr_vs_tau_positive": float(np.mean(f_pos)) if m else None,
                "mean_true_selected": float(np.mean(ntrue)) if m else None,
                "mean_selected": float(np.mean(nsel)), "share_draws_selecting": float(np.mean(np.asarray(nsel) > 0)),
                "draws_evaluated": m, "draws": len(v["draws"]), "alpha_per_class": b2["alpha_per_class"][q],
                "evaluated_subsample": v["subsampled"]}
    return out


def stage_evaluate2(ctx: Ctx) -> dict:
    """CPU only, from delta.npz (and delta2.npz). Per intervention:
    - re-split replicates: B and O pooled; replicate r at size n tests on n random inputs and takes its oracle tau_j
      (and SE) from the other N - n, so replicates differ in both test data and truth (they still resample one
      finite pool: the SE describes that resampling, not fresh draws of texts);
    - two specifications: prereg (K = 250 candidates, bet cap 0.5) and v2 (the top K = 50, bet cap 0.9);
    - a boundary stress test: each latent's Delta shifted to mean exactly delta over the pool, inputs drawn with
      replacement, so every hypothesis is null and every rejection is false; P(any rejection) is the FDR;
    - counts of tau > 0, tau > 2 SE and tau > delta from all N inputs; baselines2 against the same oracle."""
    pilot = require_gate(ctx)
    s5, v2 = ctx.s5, ctx.s5["v2"]
    z = np.load(ctx.path("delta.npz"))
    U, Kc = z["U"], int(z["n_candidates"])
    prim = s5["interventions"]["primary"]
    alphas, ns, R, ev = s5["alphas"], s5["test_sizes"], s5["replicates"], s5["evalue"]
    specs = {"prereg": (Kc, float(ev["bet_cap"])), "v2": (min(int(v2["candidates"]), Kc), float(v2["bet_cap"]))}
    delta0, k_se, Rs, chunk = pilot["delta"], float(v2["se_mult"]), int(v2["stress_replicates"]), 25
    grid = {"primary": delta0, **{f"sd_{k}": v for k, v in pilot["delta_grid"].items()}, "zero_null": 0.0}
    out = {"task": ctx.task, "model": ctx.cfg["model"]["name"], "delta": delta0, "U": int(len(U)), "candidates": Kc,
           "specs": {k: {"K": a, "bet_cap": b} for k, (a, b) in specs.items()}, "alphas": alphas, "test_sizes": ns,
           "replicates": R, "stress_replicates": Rs, "interventions": {}}
    t0 = time.time()
    for ii, kind in enumerate([prim] + list(s5["interventions"]["sensitivity"])):
        P = np.vstack([z[f"B__{kind}"], z[f"O__{kind}"]]).astype(np.float64)
        if np.isnan(P).any():
            raise SystemExit(f"{ctx.task}: delta.npz {kind} has missing values")
        N = P.shape[0]
        tau_all, se_all = P.mean(0)[:Kc], P.std(0, ddof=1)[:Kc] / math.sqrt(N)
        res = {"N": N, "counts": {"candidates": Kc, "tau_positive": int((tau_all > 0).sum()),
                                  "tau_above_2se": int((tau_all > k_se * se_all).sum()),
                                  "tau_above_delta": int((tau_all > delta0).sum()),
                                  "tau_above_delta_in_top_v2": int((tau_all[:specs["v2"][0]] > delta0).sum())}}
        rs = {sp: {g: {} for g in grid} for sp in specs}
        for n in ns:
            if n > N - v2["min_oracle"]:
                continue
            acc = {(sp, g): {a: [] for a in alphas} for sp in specs for g in grid}
            tau_r, se_r = [], []
            for r0 in range(0, R, chunk):
                perms = [stream_rng(ctx.cfg, "s5_order", ctx.ti, 1000 + ii, n, r).permutation(N)
                         for r in range(r0, min(R, r0 + chunk))]
                tau_r += [P[pm[n:], :Kc].mean(0) for pm in perms]
                se_r += [P[pm[n:], :Kc].std(0, ddof=1) / math.sqrt(N - n) for pm in perms]
                Xte = P[np.stack([pm[:n] for pm in perms])][:, :, :Kc]                 # (chunk, n, Kc)
                for sp, (Kk, cap) in specs.items():
                    for g, dl in grid.items():
                        for a, s in _selections(Xte[:, :, :Kk], dl, cap, ev, alphas).items():
                            acc[(sp, g)][a].append(s)
            tau_r, se_r = np.stack(tau_r), np.stack(se_r)
            for sp, (Kk, cap) in specs.items():
                t_, s_ = tau_r[:, :Kk], se_r[:, :Kk]
                for g, dl in grid.items():
                    ref = delta0 if g == "zero_null" else dl
                    true, band = t_ > ref, np.abs(t_ - ref) >= s5["oracle_band_se"] * s_
                    rec = {}
                    for a in alphas:
                        sel = np.concatenate(acc[(sp, g)][a])
                        am = None if g == "zero_null" else a
                        rec[str(a)] = {"vs_oracle": _summ_rep(sel, true, alpha=am),
                                       "vs_oracle_band": _summ_rep(sel, true, band, alpha=am)}
                        if g == "zero_null":             # against its own estimand, counted with and without noise
                            rec[str(a)]["vs_tau_above_2se"] = _summ_rep(sel, t_ > k_se * s_, alpha=a)
                            rec[str(a)]["vs_tau_positive"] = _summ_rep(sel, t_ > 0, alpha=a)
                    rs[sp][g][str(n)] = rec
        res["resplit"] = rs
        st = {}
        for si, (sp, (Kk, cap)) in enumerate(specs.items()):
            Xn = P[:, :Kk] - P[:, :Kk].mean(0) + delta0
            st[sp] = {}
            for n in v2["stress_sizes"]:
                anyr, nrej = {a: [] for a in alphas}, {a: [] for a in alphas}
                for r0 in range(0, Rs, chunk):
                    c = min(Rs, r0 + chunk) - r0
                    idx = stream_rng(ctx.cfg, "s5_order", ctx.ti, 2000 + ii, si, n, r0).integers(0, N, (c, n))
                    for a, s in _selections(Xn[idx], delta0, cap, ev, alphas).items():
                        anyr[a].append(s.any(-1))
                        nrej[a].append(s.sum(-1))
                st[sp][str(n)] = {}
                for a in alphas:
                    x = np.concatenate(anyr[a]).astype(float)
                    st[sp][str(n)][str(a)] = {"p_any_rejection": float(x.mean()), "se": float(x.std(ddof=1) / math.sqrt(len(x))),
                                              "mean_rejections": float(np.concatenate(nrej[a]).mean())}
        res["stress"] = st
        if ii == 0:
            res["baselines2"] = _eval_baselines2(ctx, P, U, delta0, k_se)
        out["interventions"][kind] = res
        print(f"  evaluate2 {kind}: done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    out["minutes"] = round((time.time() - t0) / 60, 1)
    ctx.write_json("evaluate2.json", out)
    ctx.write_json("evaluate2_summary.json", summarise2(out))
    print(json.dumps(summarise2(out), indent=1)[:4000], flush=True)
    return out


def summarise2(ev: dict) -> dict:
    """evaluate2's headline numbers at alpha 0.10: per intervention and specification, realised FDR and power by n
    (primary delta), breaches anywhere in the grid, the zero-null baseline, the stress test, and baselines2."""
    s = {"task": ev["task"], "model": ev["model"], "delta": ev["delta"], "specs": ev["specs"]}
    a = "0.1" if 0.1 in ev["alphas"] else str(ev["alphas"][0])
    rnd = lambda x: round(x, 4) if isinstance(x, float) else x
    for kind, r in ev["interventions"].items():
        d = {"counts": r["counts"]}
        for sp, gr in r["resplit"].items():
            d[sp] = {
                "primary": {n: {k: rnd(v[a]["vs_oracle"][k]) for k in ("fdr", "fdr_se", "power", "mean_selected", "mean_true")}
                            for n, v in gr["primary"].items()},
                "breaches": [f"{g}_n{n}_a{al}" for g, gv in gr.items() for n, nv in gv.items() for al, w in nv.items()
                             if w["vs_oracle"].get("breach")],
                "zero_null": {n: {"fdr_vs_min_effect": rnd(v[a]["vs_oracle"]["fdr"]),
                                  "fdr_vs_tau_above_2se": rnd(v[a]["vs_tau_above_2se"]["fdr"]),
                                  "mean_selected": rnd(v[a]["vs_oracle"]["mean_selected"])} for n, v in gr["zero_null"].items()},
                "stress_p_any_rejection": {n: rnd(v[a]["p_any_rejection"]) for n, v in r["stress"][sp].items()}}
        if "baselines2" in r:
            d["baselines2"] = {k: ({kk: rnd(vv) for kk, vv in v.items()} if isinstance(v, dict) else v)
                               for k, v in r["baselines2"].items()}
        s[kind] = d
    return s


# ---------------------------------------------------------------------------
# stage5_amendment_3: baselines3, delta3, evaluate3 (does a better knockoff sampler pick latents the model uses?)
# ---------------------------------------------------------------------------

def stage_baselines3(ctx: Ctx) -> dict:
    """Knockoff selections on the task's real labels from three samplers, paired draw by draw: Gaussian
    equicorrelated (Enkhbayar 2025), block-MVR Gaussian (our best Gaussian) and hurdle SCIP (stage4_repairs). Every
    sampler sees the same rows; one knockoff draw per sampler and draw (knockoffs do not depend on y) serves every
    class; each class is tested at alpha / C (one-vs-rest) and the union over classes is the draw's selection. Two
    statistics: the fixed L1-logistic penalty of Enkhbayar (lambda = 1 / (C n), FISTA) and the 5-fold CV lambda of
    Stage 4 (stage3_followups.cv_lasso, folds from stream s5_obs). W = |b_j| - |b~_j|, Knockoff+ at alpha / C."""
    require_gate(ctx)
    s5, b3 = ctx.s5, ctx.s5["baselines3"]
    ob = _obs_cache(ctx)
    if ob is None:
        out = {"note": "no pooled-latent cache for this model and task: baselines3 not run", "extra": []}
        ctx.write_json("baselines3.json", out)
        print(f"  baselines3: {out['note']}", flush=True)
        return out
    from knockpy import smatrix
    from knockpy.knockoffs import GaussianSampler
    from knockoff_audit import estimate_cov
    from planted_fdr import compute_knockoff_plus_threshold
    from planted_fdr_controls import fit_lasso_checked
    from stage3_followups import GaussKnockoffGPU, cv_lasso
    from stage4_repairs import SCIPSampler
    X, retained, cpath = ob
    rows = ctx.splits["obs"]
    X = X[rows].astype(np.float64)
    yv = ctx.y[rows]
    mu, sd = X.mean(0), X.std(0)
    keep = sd > 1e-8
    Xr = X[:, keep]
    Z = (Xr - mu[keep]) / sd[keep]
    ids = retained[keep]
    n, p = Z.shape
    classes = range(len(ctx.classes)) if len(ctx.classes) > 2 else [1]
    C = len(classes)
    alphas = s5["alphas"]
    qc = {q: q / C for q in alphas}
    D = int(b3["draws"])
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    Zt = torch.as_tensor(Z, dtype=torch.float32, device=dev)
    t0 = time.time()
    # samplers, built once per task
    equi = GaussianSampler(X=Z, method="equicorrelated")
    Sigma = estimate_cov(Z, ctx.cfg["stage4_repairs"]["covariance"])
    np.random.seed(int(stream_rng(ctx.cfg, "s5_obs", ctx.ti, 30).integers(2**31)))   # knockpy's MVR solver uses numpy's global state
    S = np.asarray(smatrix.compute_smatrix(Sigma, method="mvr", how_approx="blockdiag", max_block=b3["mvr_max_block"]))
    mvr = GaussKnockoffGPU(Zt, Sigma, S)
    scip = SCIPSampler(Xr, "hurdle", ctx.cfg["stage4_repairs"]["scip"], dev)
    print(f"  baselines3: n {n}, p {p}, classes {C}; samplers built ({time.time() - t0:.0f}s)", flush=True)

    def draw(name: str, seed: int) -> np.ndarray:
        if name == "gauss_equi":
            np.random.seed(seed)
            return equi.sample_knockoffs()
        if name == "gauss_mvr":
            return mvr.sample(seed).double().cpu().numpy()
        Xk, _ = scip.sample(seed)
        return (Xk - mu[keep]) / sd[keep]

    samplers, stats = list(b3["samplers"]), list(b3["statistics"])
    lam_fixed = 1.0 / (float(b3["knockoff_C"]) * n)
    cvsec = {"lambda_ratios": ctx.cfg["stage4_cv"]["cv"]["lambda_ratios"], "folds": ctx.cfg["stage4_cv"]["cv"]["folds"],
             "lasso": ctx.cfg["stage4_cv"]["cv"]["lasso"]}
    lasso = ctx.cfg["stage4_cv"]["cv"]["lasso"]
    sel = {f"{s}__{st}": [] for s in samplers for st in stats}
    corr = {s: [] for s in samplers}
    for d in range(D):
        for si, s in enumerate(samplers):
            Zk = draw(s, int(stream_rng(ctx.cfg, "s5_obs", ctx.ti, 31, d, si).integers(2**31)))
            corr[s].append(float(np.mean(np.sum((Z - Z.mean(0)) * (Zk - Zk.mean(0)), 0)
                                         / np.maximum(np.sqrt(np.sum((Z - Z.mean(0)) ** 2, 0) * np.sum((Zk - Zk.mean(0)) ** 2, 0)), 1e-12))))
            Phi = torch.as_tensor(np.hstack([Z, Zk]), dtype=torch.float32, device=dev)
            per = {st: {str(q): set() for q in alphas} for st in stats}
            for ci, c in enumerate(classes):
                yt = torch.as_tensor((yv == c).astype(np.float32), device=dev)
                for st in stats:
                    if st == "fixed":
                        w = fit_lasso_checked(Phi, yt, lam_fixed, lasso["max_iter"], lasso["tol"])[0]
                    else:   # the same folds for every sampler: paired
                        w = cv_lasso(Phi, yt, cvsec, stream_rng(ctx.cfg, "s5_obs", ctx.ti, 32, d, ci))[0]
                    wa = w.abs().cpu().numpy()
                    W = wa[:p] - wa[p:]
                    for q in alphas:
                        tau = compute_knockoff_plus_threshold(W, qc[q])
                        if np.isfinite(tau):
                            per[st][str(q)] |= set(ids[W >= tau].tolist())
            for st in stats:
                sel[f"{s}__{st}"].append({q: sorted(int(j) for j in v) for q, v in per[st].items()})
            del Phi
        print(f"  baselines3 draw {d + 1}/{D} ({(time.time() - t0) / 60:.1f} min): "
              + "; ".join(f"{k} {[len(v) for v in sel[k][-1].values()]}" for k in sel), flush=True)
    cap = int(b3["cap"])
    U = {int(j) for j in ctx.load_json("baselines.json")["U"]}
    out = {"classes": C, "alpha_per_class": {str(q): qc[q] for q in alphas}, "draws": D, "rows": int(n),
           "lambda_fixed": lam_fixed, "self_correlation": {s: float(np.mean(v)) for s, v in corr.items()},
           "observational_cache": cpath, "baselines": {}}
    for bi, (name, dl) in enumerate(sel.items()):
        union = sorted(set().union(*[set(v) for dsel in dl for v in dsel.values()]))
        sub = union
        if len(union) > cap:
            sub = sorted(stream_rng(ctx.cfg, "s5_obs", ctx.ti, 33, bi).choice(union, size=cap, replace=False).tolist())
        out["baselines"][name] = {"draws": dl, "evaluated": [int(j) for j in sub], "n_union": len(union),
                                  "subsampled": len(union) > cap}
    out["extra"] = sorted(set().union(*[set(v["evaluated"]) for v in out["baselines"].values()]) - U)
    out["minutes"] = round((time.time() - t0) / 60, 1)
    ctx.write_json("baselines3.json", out)
    print(f"  baselines3: {len(out['extra'])} latents not in U go to delta3 ({out['minutes']} min)", flush=True)
    return out


def stage_delta3(ctx: Ctx) -> dict:
    """Delta_j(x) under the primary intervention for baselines3's latents not in U, on the split-O inputs (the oracle
    rows; delta's row order), with its own resampling draws (stream s5_resample, pool index 4)."""
    require_gate(ctx)
    s5, R = ctx.s5, ctx.runner
    b3 = ctx.load_json("baselines3.json")
    E = np.asarray(b3.get("extra", []), dtype=np.int64)
    prim = s5["interventions"]["primary"]
    rows = ctx.splits["O"]
    res = np.full((len(rows), len(E)), np.nan, dtype=np.float32)
    t0 = time.time()
    if len(E):
        scr = np.load(ctx.path("screen.npz"))
        bank = ResampleBank({k: scr[k] for k in ("indptr", "idx", "val")}, E, R.W_dec.shape[0], s5["resample"]).to(R.device)
        print(f"  resampling sampler: {bank.summary()}", flush=True)
        Et = torch.as_tensor(E, device=R.device)
        W_E = R.W_dec[Et]
        C = s5["replay_batch"]
        ck = ctx.path("delta3_checkpoint.npz")
        start = 0
        if ck.exists():
            zc = np.load(ck)
            if list(zc["E"]) == list(E):
                res, start = zc["res"], int(zc["done"])
                print(f"  resuming at {start}/{len(rows)}", flush=True)
        last_ck = time.time()
        for k in range(start, len(rows)):
            i = int(rows[k])
            ids, q0 = ctx.prompts.build(ctx.texts[i])
            h, caps = R.base(ids)
            Aq = R.encode(h)[q0:]
            a_cur = Aq[:, Et].T.contiguous()
            with torch.no_grad():
                m0 = float(correct_prob(R.replay(h[None], caps), ctx.y[i])[0])
            gen = torch.Generator(device=R.device)
            gen.manual_seed(int(stream_rng(ctx.cfg, "s5_resample", ctx.ti, 4, k).integers(2**62)))
            d = (_new_values(prim, torch.arange(len(E), device=R.device), a_cur, Aq, bank, gen) - a_cur).to(h.dtype)
            outv = np.empty(len(E), dtype=np.float32)
            for s0 in range(0, len(E), C):
                dd = d[s0: s0 + C]
                H = h[None].repeat(len(dd), 1, 1)
                H[:, q0:, :] += dd[:, :, None] * W_E[s0: s0 + C][:, None, :].to(h.dtype)
                with torch.no_grad():
                    mm = correct_prob(R.replay(H, caps), ctx.y[i])
                outv[s0: s0 + len(dd)] = (m0 - mm).cpu().numpy()
            res[k] = outv
            if time.time() - last_ck > 600 or k + 1 == len(rows):
                tmp = ck.with_suffix(".tmp.npz")
                np.savez(tmp, E=E, res=res, done=k + 1)
                tmp.replace(ck)
                last_ck = time.time()
                print(f"  delta3 O {k + 1}/{len(rows)}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        ck.unlink(missing_ok=True)
    np.savez(ctx.path("delta3.npz"), E=E, **{f"O__{prim}": res})
    out = {"extra": int(len(E)), "rows": int(len(rows)), "primary": prim, "minutes": round((time.time() - t0) / 60, 1)}
    ctx.write_json("delta3.json", out)
    print(f"  delta3: {len(E)} latents ({out['minutes']} min)", flush=True)
    return out


def stage_evaluate3(ctx: Ctx) -> dict:
    """CPU only. Oracle tau_j and SE from the split-O inputs for every latent (delta: U; delta2: baselines2's extra
    latents; delta3: baselines3's). Per sampler and statistic, draw by draw at each alpha: the share of selections
    with tau > delta, tau > 2 SE and tau > 0, the number selected, and recall of U's latents with tau > delta. Paired
    contrasts (same draw, same rows, same folds): hurdle - block MVR (primary) and hurdle - equicorrelated, on the
    share with tau > 2 SE, over the draws where both select (task level), and pooled over tasks in the summary stage."""
    pilot = require_gate(ctx)
    s5, b3c = ctx.s5, ctx.s5["baselines3"]
    b3 = ctx.load_json("baselines3.json")
    if "baselines" not in b3:
        ctx.write_json("evaluate3.json", {"task": ctx.task, "note": b3.get("note", "")})
        return {}
    prim = s5["interventions"]["primary"]
    delta0, k_se = pilot["delta"], float(s5["v2"]["se_mult"])
    T = {}
    z = np.load(ctx.path("delta.npz"))
    for src_ids, P in [(z["U"], z[f"O__{prim}"])]:
        P = P.astype(np.float64)
        T.update({int(j): (float(t), float(s)) for j, t, s in zip(src_ids, P.mean(0), P.std(0, ddof=1) / math.sqrt(len(P)))})
    if ctx.path("delta2.npz").exists():
        z2 = np.load(ctx.path("delta2.npz"))
        if len(z2["E"]):
            P = z2[f"O__{prim}"].astype(np.float64)
            T.update({int(j): (float(t), float(s)) for j, t, s in zip(z2["E"], P.mean(0), P.std(0, ddof=1) / math.sqrt(len(P)))})
    z3 = np.load(ctx.path("delta3.npz"))
    if len(z3["E"]):
        P = z3[f"O__{prim}"].astype(np.float64)
        T.update({int(j): (float(t), float(s)) for j, t, s in zip(z3["E"], P.mean(0), P.std(0, ddof=1) / math.sqrt(len(P)))})
    U = [int(j) for j in z["U"]]
    true_U = {j for j in U if T[j][0] > delta0}
    rec = {}
    for name, v in b3["baselines"].items():
        ev_set = set(v["evaluated"])
        rec[name] = {}
        for q in v["draws"][0]:
            rows = []
            for dsel in v["draws"]:
                lat = [j for j in dsel[q] if j in ev_set]
                t = np.array([T[j][0] for j in lat]) if lat else np.zeros(0)
                s = np.array([T[j][1] for j in lat]) if lat else np.zeros(0)
                rows.append({"n_selected": len(dsel[q]), "n_evaluated": len(lat),
                             "share_above_delta": float((t > delta0).mean()) if lat else None,
                             "share_above_2se": float((t > k_se * s).mean()) if lat else None,
                             "share_positive": float((t > 0).mean()) if lat else None,
                             "recall_above_delta": float(len(set(dsel[q]) & true_U) / max(len(true_U), 1))})
            rec[name][q] = rows
    def agg(rows):
        f = lambda k: [r[k] for r in rows if r[k] is not None]
        m = lambda x: float(np.mean(x)) if x else None
        return {"mean_selected": m([r["n_selected"] for r in rows]), "share_draws_selecting": m([r["n_selected"] > 0 for r in rows]),
                "share_above_delta": m(f("share_above_delta")), "share_above_2se": m(f("share_above_2se")),
                "share_positive": m(f("share_positive")), "recall_above_delta": m([r["recall_above_delta"] for r in rows]),
                "draws_with_selection": len(f("share_above_2se"))}
    summary = {name: {q: agg(rows) for q, rows in by_q.items()} for name, by_q in rec.items()}
    contrasts = {}
    for st in b3c["statistics"]:
        for other in ("gauss_mvr", "gauss_equi"):
            a, b = f"hurdle__{st}", f"{other}__{st}"
            if a not in rec or b not in rec:
                continue
            for q in rec[a]:
                diffs = [ra["share_above_2se"] - rb["share_above_2se"] for ra, rb in zip(rec[a][q], rec[b][q])
                         if ra["share_above_2se"] is not None and rb["share_above_2se"] is not None]
                dsel = [ra["n_selected"] - rb["n_selected"] for ra, rb in zip(rec[a][q], rec[b][q])]
                drec = [ra["recall_above_delta"] - rb["recall_above_delta"] for ra, rb in zip(rec[a][q], rec[b][q])]
                contrasts[f"hurdle_minus_{other}__{st}__q{q}"] = {
                    "share_above_2se_diff": float(np.mean(diffs)) if diffs else None,
                    "share_above_2se_diffs": diffs, "n_paired_draws": len(diffs),
                    "n_selected_diff": float(np.mean(dsel)), "recall_above_delta_diff": float(np.mean(drec))}
    out = {"task": ctx.task, "model": ctx.cfg["model"]["name"], "delta": delta0, "oracle_rows": int(z[f"O__{prim}"].shape[0]),
           "n_true_U": len(true_U), "self_correlation": b3["self_correlation"], "draws": rec, "summary": summary,
           "contrasts": contrasts}
    ctx.write_json("evaluate3.json", out)
    a = "0.1"
    print(f"  evaluate3 {ctx.task} (q = 0.1): " + "; ".join(
        f"{k} sel {v[a]['mean_selected']:.1f}, >2SE {v[a]['share_above_2se']}, >delta {v[a]['share_above_delta']}, "
        f"recall {v[a]['recall_above_delta']:.2f}" for k, v in summary.items() if a in v), flush=True)
    return out


def stage_summary3(cfg_paths: list) -> dict:
    """stage5_amendment_3 rule B3-1, over every model and task: the paired hurdle - block-MVR difference in the share
    of selections with tau > 2 SE (CV statistic, q = 0.1), pooled over (task, draw) pairs where both select; a
    one-sample t-test of the pooled differences against 0 (two-sided), and per task."""
    from scipy import stats as sstats
    rows, per_task, out = [], {}, {"tasks": {}}
    for cp in cfg_paths:
        cfg = load_config(cp)
        for t in cfg["stage5"]["tasks"]:
            f = results_dir(cfg) / "stage5" / t / "evaluate3.json"
            if not f.exists():
                continue
            e = json.loads(f.read_text())
            key = f"{cfg['model']['name']}/{t}"
            out["tasks"][key] = {k: e.get(k) for k in ("summary", "contrasts", "n_true_U", "self_correlation", "note")}
            for name, c in (e.get("contrasts") or {}).items():
                if name.startswith("hurdle_minus_gauss_mvr__cv__q0.1"):
                    rows += c["share_above_2se_diffs"]
                    per_task[key] = c["share_above_2se_diff"]
    if rows:
        tt = sstats.ttest_1samp(rows, 0.0)
        out["B3_1"] = {"pooled_mean_diff": float(np.mean(rows)), "n_pairs": len(rows),
                       "p_two_sided": float(tt.pvalue) if len(rows) > 1 else None, "per_task": per_task,
                       "hurdle_better": bool(np.mean(rows) > 0 and len(rows) > 1 and tt.pvalue < 0.05)}
    else:
        out["B3_1"] = {"note": "no (task, draw) pair where both samplers select: the contrast is undefined", "per_task": per_task}
    dest = ROOT / "results" / "stage5_baselines3_summary.json"
    dest.write_text(json.dumps(out, indent=2, default=_json_default))
    print(json.dumps(out.get("B3_1"), indent=1, default=_json_default))
    print(f"wrote {dest}")
    return out


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
        if (d / "evaluate2_summary.json").exists():                  # stage5_amendment_2
            out.setdefault("tasks_v2", {})[t] = json.loads((d / "evaluate2_summary.json").read_text())
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
    ap.add_argument("--stage", default="all", choices=list(STAGES) + list(V2_STAGES) + list(V3_STAGES) + ["all", "v2", "v3", "summary", "summary3"],
                    help="all: the stage5_amendment_1 stages; v2: baselines2, delta2, evaluate2 (stage5_amendment_2); "
                         "v3: baselines3, delta3, evaluate3 (stage5_amendment_3); summary3: rule B3-1 over both models")
    ap.add_argument("--limit", type=int, default=None, help="debug: cap every split at N inputs (results not valid)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.stage == "summary":
        stage_summary(cfg)
        return
    if args.stage == "summary3":
        stage_summary3([ROOT / "config" / "default.yaml", ROOT / "config" / "models" / "gemma3_1b.yaml"])
        return
    if args.task not in cfg["stage5"]["tasks"]:
        raise SystemExit(f"--task must be one of {list(cfg['stage5']['tasks'])}")
    torch.manual_seed(int(stream_rng(cfg, "s5_resample", 999).integers(2**31)))
    stages = {"all": list(STAGES), "v2": list(V2_STAGES), "v3": list(V3_STAGES)}.get(args.stage, [args.stage])
    done_file = {"pilot": "pilot.json", "screen": "screen.json", "baselines": "baselines.json",
                 "delta": "delta.json", "evaluate": "evaluate.json", "baselines2": "baselines2.json",
                 "delta2": "delta2.json", "evaluate2": "evaluate2.json", "baselines3": "baselines3.json",
                 "delta3": "delta3.json", "evaluate3": "evaluate3.json"}
    out_dir = results_dir(cfg) / "stage5" / (f"{args.task}__limit{args.limit}" if args.limit else args.task)
    todo = [s for s in stages if args.stage not in ("all", "v2", "v3") or not (out_dir / done_file[s]).exists()]
    ctx = Ctx(cfg, args.task, args.limit, light=all(s in ("evaluate", "evaluate2", "evaluate3") for s in todo))
    if args.limit:
        ctx.out = out_dir
        ctx.out.mkdir(parents=True, exist_ok=True)
    need_model = [s for s in todo if s in ("pilot", "screen", "baselines", "delta", "delta2", "delta3")]
    if need_model:
        from cache_activations import load_model_and_sae
        tok, model, sae = load_model_and_sae(cfg)
        ctx.attach(tok, model, sae)
    print(f"stage 5 | {cfg['model']['name']} | task {args.task} | stages {todo} | out {ctx.out}", flush=True)
    fns = {"pilot": stage_pilot, "screen": stage_screen, "baselines": stage_baselines, "delta": stage_delta,
           "evaluate": stage_evaluate, "baselines2": stage_baselines2, "delta2": stage_delta2,
           "evaluate2": stage_evaluate2, "baselines3": stage_baselines3, "delta3": stage_delta3,
           "evaluate3": stage_evaluate3}
    for s in todo:
        print(f"\n=== {s} ===", flush=True)
        r = fns[s](ctx)
        if s == "pilot" and not r["gate"]["passed"]:
            print(f"  gate failed: {args.task} stops here (reported as not run)", flush=True)
            return


if __name__ == "__main__":
    main()
