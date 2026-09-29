"""Stage 4, Component 1: Distribution-Aware Hurdle Knockoff Sampler.

Constructs knockoffs for zero-inflated, continuous non-negative features
(e.g., Sparse Autoencoder JumpReLU activations) using a Gaussian Copula Hurdle model.

Key Mathematical Properties:
1. Zero-Mass Preservation: The knockoffs tilde{X}_j match the empirical zero mass
   p_{0,j} = Pr(X_j = 0) up to finite-sample sampling fluctuations.
2. Trivial Classifier Mitigation: Because Pr(tilde{X}_j = 0) approximately equals Pr(X_j = 0),
   the trivial zero-indicator classifier that exploited Gaussian knockoffs drops
   from ~94% accuracy towards chance (~50%).
3. Quantile Inversion: The positive tail uses empirical quantile mapping, preserving
   empirical skewness and heavy tails without imposing rigid parametric forms.
4. Latent Second-Order Structure: Second-order exchangeability is enforced in the latent Gaussian
   copula space via an S-matrix (MVR, SDP, or Equicorrelated). Note that while
   second-order exchangeability holds in the latent Gaussian domain, joint Model-X
   exchangeability after non-linear hurdle thresholding is an empirical copula
   approximation that must be evaluated empirically rather than assumed analytically.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch


class GaussianCopulaHurdleSampler:
    """Gaussian Copula Hurdle Knockoff Sampler for zero-inflated activations."""

    def __init__(
        self,
        s_method: str = "mvr",
        covariance_estimator: str = "ledoit_wolf",
        positive_tail: str = "empirical",
    ):
        """
        Args:
            s_method: Knockoff S-matrix construction ('mvr', 'equicorrelated', 'sdp').
            covariance_estimator: 'ledoit_wolf' or 'sample'.
            positive_tail: 'empirical' (rank quantile mapping) or 'lognormal'.
        """
        self.s_method = s_method
        self.covariance_estimator = covariance_estimator
        self.positive_tail = positive_tail

        # Fitted parameters
        self.p0: np.ndarray | None = None  # Zero-mass probabilities (p,)
        self.thresholds: np.ndarray | None = None  # Normal thresholds Phi^{-1}(p0)
        self.pos_values: list[np.ndarray] | None = None  # Empirical positive sorted values
        self.lognormal_params: list[tuple[float, float]] | None = None
        self.Sigma: np.ndarray | None = None  # Latent Gaussian correlation matrix
        self.S: np.ndarray | None = None  # Diagonal knockoff S-matrix
        self.Z_latent: np.ndarray | None = None  # Latent Gaussian representations of X
        self.p: int = 0
        self.n: int = 0

    def fit(self, X: np.ndarray, rng: np.random.Generator | None = None) -> GaussianCopulaHurdleSampler:
        """Fit the Hurdle Copula model on activations X (n, p).
        
        Note on Randomized PIT:
            The Probability Integral Transform (PIT) uses randomized tie-breaking
            u_zero ~ Uniform(0, p_{0,j}) for exact zero entries to continuously embed
            the point mass into latent Gaussian space. The fitted Z_latent represents
            one fixed realization of this transformation governed by `rng`, ensuring
            deterministic reproducibility across replicates when X is fixed.
        """
        if rng is None:
            rng = np.random.default_rng(20260904)

        self.n, self.p = X.shape
        self.p0 = np.zeros(self.p, dtype=np.float64)
        self.thresholds = np.zeros(self.p, dtype=np.float64)
        self.pos_values = []
        self.lognormal_params = []

        Z_latent = np.zeros((self.n, self.p), dtype=np.float64)

        for j in range(self.p):
            col = X[:, j]
            zero_mask = (col == 0.0)
            n_zero = np.sum(zero_mask)
            p0_j = float(n_zero) / float(self.n)
            
            # Constrain p0 away from 0 and 1 for numerical stability
            p0_j = float(np.clip(p0_j, 1e-5, 1.0 - 1e-5))
            self.p0[j] = p0_j
            self.thresholds[j] = stats.norm.ppf(p0_j)

            # Store positive tail
            pos = col[~zero_mask]
            if len(pos) == 0:
                pos = np.array([1e-3], dtype=np.float64)
            self.pos_values.append(np.sort(pos))

            if self.positive_tail == "lognormal":
                log_pos = np.log(np.maximum(pos, 1e-8))
                self.lognormal_params.append((float(np.mean(log_pos)), float(np.std(log_pos) + 1e-6)))

            # Probability Integral Transform (PIT) with randomized tie-breaking on zero atom
            # For zero entries: draw Uniform(0, p0_j)
            u_zero = rng.uniform(low=1e-8, high=p0_j, size=n_zero)
            
            # For positive entries: rank-based Uniform(p0_j, 1.0)
            n_pos = self.n - n_zero
            if n_pos > 0:
                ranks = stats.rankdata(pos, method="average")
                u_pos = p0_j + (1.0 - p0_j) * ((ranks - 0.5) / float(n_pos))
            else:
                u_pos = np.empty(0, dtype=np.float64)

            # Map uniforms to latent standard normal
            u_full = np.zeros(self.n, dtype=np.float64)
            u_full[zero_mask] = u_zero
            u_full[~zero_mask] = u_pos
            u_full = np.clip(u_full, 1e-7, 1.0 - 1e-7)

            Z_latent[:, j] = stats.norm.ppf(u_full)

        self.Z_latent = Z_latent

        # Estimate latent Gaussian covariance / correlation
        if self.covariance_estimator == "ledoit_wolf":
            cov_est = LedoitWolf(assume_centered=False).fit(Z_latent).covariance_
        else:
            cov_est = np.cov(Z_latent, rowvar=False)

        # Standardize covariance to correlation matrix
        d = np.sqrt(np.diag(cov_est))
        d = np.where(d > 1e-8, d, 1.0)
        Sigma = cov_est / np.outer(d, d)
        
        # Ensure positive definiteness
        min_eig = np.min(np.linalg.eigvalsh(Sigma))
        if min_eig < 1e-4:
            Sigma += (1e-4 - min_eig) * np.eye(self.p)
            d = np.sqrt(np.diag(Sigma))
            Sigma = Sigma / np.outer(d, d)

        self.Sigma = Sigma

        # Compute knockoff S-matrix in latent Gaussian space
        if self.s_method == "equicorrelated":
            # Exact closed-form equicorrelated construction: S = s * I
            ev = np.linalg.eigvalsh(self.Sigma)
            min_eig = max(float(ev[0]), 1e-4)
            s_val = min(2.0 * min_eig, 0.99)
            S_raw = np.diag(np.full(self.p, s_val))
        elif self.s_method in ("mvr", "sdp"):
            try:
                import knockpy.smatrix
                S_raw = np.asarray(knockpy.smatrix.compute_smatrix(self.Sigma, method=self.s_method))
            except ImportError as e:
                raise RuntimeError(
                    f"The '{self.s_method}' construction requires knockpy. "
                    "Install knockpy or use s_method='equicorrelated'."
                ) from e
        else:
            raise ValueError(f"Unknown s_method: {self.s_method!r}. Choose 'mvr', 'equicorrelated', or 'sdp'.")

        self.S = S_raw

        # Precompute knockoff generation matrices C and L in latent space
        inv_Sigma = np.linalg.pinv(self.Sigma)
        self.C = np.eye(self.p) - inv_Sigma @ self.S

        V_k = 2.0 * self.S - self.S @ inv_Sigma @ self.S
        V_k = 0.5 * (V_k + V_k.T)
        eigvals, eigvecs = np.linalg.eigh(V_k)
        eigvals = np.maximum(eigvals, 1e-8)
        self.L = eigvecs @ np.diag(np.sqrt(eigvals))

        return self

    def sample_knockoffs(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample Hurdle knockoffs tilde{X} matching the empirical zero mass and positive tail."""
        if self.C is None or self.L is None or self.Z_latent is None:
            raise RuntimeError("Sampler must be fitted via .fit(X) before sampling knockoffs.")

        if rng is None:
            rng = np.random.default_rng()

        p = self.p
        n = self.n

        # Fast latent knockoff draw: tilde{Z} = Z @ C + noise @ L^T
        mu_k = self.Z_latent @ self.C
        noise = rng.standard_normal((n, p))
        Z_tilde = mu_k + noise @ self.L.T

        # Quantile Inversion: map latent Gaussian knockoffs back to the Hurdle domain
        X_tilde = np.zeros((n, p), dtype=np.float64)

        for j in range(p):
            p0_j = self.p0[j]
            thresh_j = self.thresholds[j]
            z_col = Z_tilde[:, j]

            # Condition 1: If z_col <= thresh_j, knockoff activation is EXACT ZERO
            zero_idx = (z_col <= thresh_j)
            X_tilde[zero_idx, j] = 0.0

            # Condition 2: If z_col > thresh_j, invert into positive tail
            pos_idx = ~zero_idx
            if np.any(pos_idx):
                u_pos = stats.norm.cdf(z_col[pos_idx])
                # Scale uniform quantile to (0, 1) conditional on being in positive tail
                q_pos = (u_pos - p0_j) / (1.0 - p0_j)
                q_pos = np.clip(q_pos, 1e-6, 1.0 - 1e-6)

                if self.positive_tail == "empirical":
                    # Continuous piece-wise linear quantile mapping
                    sorted_vals = self.pos_values[j]
                    if len(sorted_vals) > 1:
                        grid = np.linspace(0.0, 1.0, len(sorted_vals))
                        X_tilde[pos_idx, j] = np.interp(q_pos, grid, sorted_vals)
                    else:
                        X_tilde[pos_idx, j] = sorted_vals[0]
                elif self.positive_tail == "lognormal":
                    mu_log, std_log = self.lognormal_params[j]
                    X_tilde[pos_idx, j] = np.exp(stats.norm.ppf(q_pos, loc=mu_log, scale=std_log))

        return X_tilde


# ---------------------------------------------------------------------------
# Diagnostics & Benchmark Evaluation
# ---------------------------------------------------------------------------

def evaluate_hurdle_diagnostics(X: np.ndarray, X_tilde: np.ndarray) -> dict:
    """Evaluate zero-mass preservation and trivial classifier accuracy."""
    p_zero_real = (X == 0.0).mean(axis=0)
    p_zero_knockoff = (X_tilde == 0.0).mean(axis=0)
    
    # Trivial zero-indicator classifier accuracy:
    # 0.5 * (Pr(predict real | real) + Pr(predict knockoff | knockoff))
    acc = 0.5 * (p_zero_real + (1.0 - p_zero_knockoff))
    zero_diff = np.abs(p_zero_real - p_zero_knockoff)
    
    return {
        "median_p_zero_real": float(np.median(p_zero_real)),
        "median_p_zero_knockoff": float(np.median(p_zero_knockoff)),
        "max_zero_mass_error": float(np.max(zero_diff)),
        "mean_zero_mass_error": float(np.mean(zero_diff)),
        "median_classifier_accuracy": float(np.median(acc)),
        "p95_classifier_accuracy": float(np.quantile(acc, 0.95)),
    }


def evaluate_swap_exchangeability(
    X: np.ndarray,
    X_tilde: np.ndarray,
    rng: np.random.Generator | None = None,
    swap_fraction: float = 0.5,
    holdout: float = 0.3,
) -> dict:
    """Empirical Model-X exchangeability swap two-sample test.

    Under exact Model-X exchangeability, for any coordinate subset S:
        (X, X_tilde)_swap(S) \stackrel{d}{=} (X, X_tilde)

    We construct:
      - Group 0: Unswapped original pairs [X_A, X_tilde_A]
      - Group 1: Swapped pairs [X_B, X_tilde_B]_swap(S)

    A classifier is trained to separate Group 0 vs Group 1 on a training split,
    and evaluated on a held-out test split. Under exchangeability, the held-out
    ROC-AUC should be approximately 0.50 (chance).
    """
    if rng is None:
        rng = np.random.default_rng(20260904)

    n, p = X.shape
    perm = rng.permutation(n)
    n_A = n // 2
    idx_A, idx_B = perm[:n_A], perm[n_A:]

    # Paired standardization to avoid trivial scale artifacts
    mu_X = X.mean(axis=0)
    sd_X = X.std(axis=0)
    sd_X = np.where(sd_X > 1e-8, sd_X, 1.0)
    Xs = (X - mu_X) / sd_X
    Xts = (X_tilde - mu_X) / sd_X

    # Group 0: unswapped
    Z0 = np.hstack([Xs[idx_A], Xts[idx_A]])

    def _test_swap(S_idx: np.ndarray) -> float:
        X1 = Xs[idx_B].copy()
        Xt1 = Xts[idx_B].copy()
        if len(S_idx) > 0:
            X1[:, S_idx], Xt1[:, S_idx] = Xt1[:, S_idx], X1[:, S_idx]
        Z1 = np.hstack([X1, Xt1])

        # Combine into labeled dataset
        Z_all = np.vstack([Z0, Z1]).astype(np.float32)
        y_all = np.concatenate([np.zeros(len(Z0)), np.ones(len(Z1))])

        perm_sub = rng.permutation(len(y_all))
        Z_all, y_all = Z_all[perm_sub], y_all[perm_sub]

        n_train = int(len(y_all) * (1.0 - holdout))
        clf = LogisticRegression(max_iter=500, C=1.0, random_state=int(rng.integers(2**31)))
        clf.fit(Z_all[:n_train], y_all[:n_train])
        probs = clf.predict_proba(Z_all[n_train:])[:, 1]
        return float(roc_auc_score(y_all[n_train:], probs))

    # 1. Null control: |S| = 0 (no swap; should be ~0.50)
    auc_null = _test_swap(np.array([], dtype=int))

    # 2. Subset swap: random subset of features (default 50%)
    k_swap = max(1, int(p * swap_fraction))
    S_subset = np.sort(rng.choice(p, size=k_swap, replace=False))
    auc_subset = _test_swap(S_subset)

    # 3. Full swap: all coordinates S = {0, ..., p-1}
    auc_full = _test_swap(np.arange(p))

    return {
        "auc_null": auc_null,
        "auc_subset_swap": auc_subset,
        "auc_full_swap": auc_full,
        "swap_subset_size": int(k_swap),
    }


def run_hurdle_trial(
    X: np.ndarray,
    S_idx: np.ndarray,
    form: str,
    amplitude: float,
    q: float,
    sampler: GaussianCopulaHurdleSampler,
    rng: np.random.Generator,
    lam: float = 0.02,
    device: str = "cpu",
) -> tuple[float, float, int]:
    """Run a single planted-signal trial with Hurdle knockoffs."""
    from planted_fdr import (
        compute_knockoff_plus_threshold,
        evaluate_discoveries,
        fit_lasso,
        generate_planted_labels,
    )

    n, p = X.shape
    # Standardize X for signal generation so all k planted latents have unit variance
    mu_X = X.mean(axis=0)
    sd_X = X.std(axis=0)
    sd_X = np.where(sd_X > 1e-8, sd_X, 1.0)
    X_std_sig = (X - mu_X) / sd_X

    y_np, _ = generate_planted_labels(X_std_sig, S_idx, form, amplitude, rng)
    X_tilde = sampler.sample_knockoffs(rng=rng)

    # Paired standardization: scale both X and X_tilde using identical (mu_X, sd_X)
    # to avoid artificial asymmetry between real and knockoff pairs in Lasso
    X_std = (X - mu_X) / sd_X
    X_tilde_std = (X_tilde - mu_X) / sd_X
    Z_std = np.hstack([X_std, X_tilde_std])

    # Fit L1-logistic Lasso via FISTA
    Z_t = torch.tensor(Z_std, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_np, dtype=torch.float32, device=device)
    w = fit_lasso(Z_t, y_t, lam=lam, lr=0.033, max_iter=800)
    w_cpu = w.detach().cpu().numpy()

    # Contrast statistic W_j = |w_j| - |w_{j+p}|
    W = np.abs(w_cpu[:p]) - np.abs(w_cpu[p:])
    tau = compute_knockoff_plus_threshold(W, q)

    if np.isinf(tau):
        discovered = set()
    else:
        discovered = set(np.where(W >= tau)[0].tolist())

    fdr, power = evaluate_discoveries(discovered, set(S_idx.tolist()))
    return fdr, power, len(discovered)


def run_stage4_hurdle_benchmark(
    X: np.ndarray,
    signal_sizes: list[int] | None = None,
    amplitudes: list[float] | None = None,
    forms: list[str] | None = None,
    fdr_targets: list[float] | None = None,
    replicates: int = 5,
    s_method: str = "mvr",
    device: str = "cpu",
    output_path: str | None = None,
) -> dict:
    """Run the Stage 4 benchmark across conditions using the Hurdle knockoff sampler."""
    if signal_sizes is None:
        signal_sizes = [10, 20]
    if amplitudes is None:
        amplitudes = [0.5, 1.0, 2.0, 5.0]
    if forms is None:
        forms = ["linear", "interaction"]
    if fdr_targets is None:
        fdr_targets = [0.10]

    n, p = X.shape
    print(f"Initializing Stage 4 Hurdle Benchmark: n={n}, p={p}")
    print(f"Conditions: k={signal_sizes}, amps={amplitudes}, forms={forms}, q={fdr_targets}, reps={replicates}")

    rng = np.random.default_rng(20260904)

    # Fit the Hurdle sampler once on X
    sampler = GaussianCopulaHurdleSampler(s_method=s_method)
    sampler.fit(X, rng=rng)

    results_grid = []

    for form_idx, form in enumerate(forms):
        for k in signal_sizes:
            for amp in amplitudes:
                for q in fdr_targets:
                    fdrs, powers, n_discs = [], [], []
                    for rep in range(replicates):
                        # Explicit disjoint seed per condition to avoid cross-cell RNG coupling
                        cell_seed = int(20260904 + form_idx * 100000 + k * 10000 + int(amp * 100) + rep * 10 + int(q * 1000))
                        rep_rng = np.random.default_rng(cell_seed)
                        S_idx = np.sort(rep_rng.choice(p, size=k, replace=False))
                        fdr, power, nd = run_hurdle_trial(
                            X=X,
                            S_idx=S_idx,
                            form=form,
                            amplitude=amp,
                            q=q,
                            sampler=sampler,
                            rng=rep_rng,
                            device=device,
                        )
                        fdrs.append(fdr)
                        powers.append(power)
                        n_discs.append(nd)

                    res = {
                        "form": form,
                        "k": k,
                        "amplitude": amp,
                        "q": q,
                        "fdr_mean": float(np.mean(fdrs)),
                        "fdr_se": float(np.std(fdrs) / np.sqrt(replicates)),
                        "power_mean": float(np.mean(powers)),
                        "power_se": float(np.std(powers) / np.sqrt(replicates)),
                        "mean_discoveries": float(np.mean(n_discs)),
                    }
                    results_grid.append(res)
                    print(f"  [{form:<11}] k={k:<2} amp={amp:<4} q={q:<4} -> FDR: {res['fdr_mean']:.3f} | Power: {res['power_mean']:.3f}")

    out_data = {
        "benchmark": "stage4_hurdle_benchmark",
        "n": n,
        "p": p,
        "s_method": s_method,
        "results": results_grid,
    }

    if output_path is not None:
        import json
        with open(output_path, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"Results saved to {output_path}")

    return out_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 4 Hurdle Sampler & Benchmark")
    parser.add_argument("--s_method", type=str, default="mvr", choices=["mvr", "equicorrelated", "sdp"], help="Knockoff construction method")
    parser.add_argument("--reps", type=int, default=3, help="Replicates per cell for quick check")
    args = parser.parse_args()

    print("=== Stage 4: Gaussian Copula Hurdle Sampler & Benchmark ===")
    rng = np.random.default_rng(20260904)
    
    # Smoke test on synthetic zero-inflated data (89% zero mass, p=50, n=2000)
    # NOTE: This validates sampler numerical mechanics and pipeline flow.
    # Production Stage 4 runs should pass the full 67,349 x 2,048 SAE activation matrix.
    n_test, p_test = 2000, 50
    raw_latent = rng.standard_normal((n_test, p_test))
    X_synthetic = np.where(raw_latent > 1.23, raw_latent - 1.23, 0.0)
    
    print(f"\n1. Testing Copula Fit & Knockoff Generation (smoke test: n={n_test}, p={p_test})...")
    sampler = GaussianCopulaHurdleSampler(s_method=args.s_method)
    sampler.fit(X_synthetic, rng=rng)
    X_tilde = sampler.sample_knockoffs(rng=rng)

    diag = evaluate_hurdle_diagnostics(X_synthetic, X_tilde)
    swap_diag = evaluate_swap_exchangeability(X_synthetic, X_tilde, rng=rng)
    print("   Diagnostics:")
    print(f"   - Median Real Zero Fraction:     {diag['median_p_zero_real']:.4f}")
    print(f"   - Median Knockoff Zero Fraction: {diag['median_p_zero_knockoff']:.4f}")
    print(f"   - Max Zero-Mass Difference:      {diag['max_zero_mass_error']:.4f}")
    print(f"   - Trivial Classifier Accuracy:   {diag['median_classifier_accuracy']:.4f} (Gaussian was ~0.945)")
    print(f"   - Swap Null Control AUC (|S|=0): {swap_diag['auc_null']:.4f} (Expected ~0.50)")
    print(f"   - Swap Test AUC (50% subset):    {swap_diag['auc_subset_swap']:.4f} (Exchangeable ~0.50)")
    print(f"   - Swap Test AUC (full swap):     {swap_diag['auc_full_swap']:.4f} (Exchangeable ~0.50)")

    print("\n2. Executing Stage 4 Mini-Benchmark across Linear and Interaction signals...")
    run_stage4_hurdle_benchmark(
        X=X_synthetic,
        signal_sizes=[10],
        amplitudes=[0.5, 2.0],
        forms=["linear", "interaction"],
        fdr_targets=[0.10],
        replicates=args.reps,
        s_method=args.s_method,
    )
    print("\n=== Stage 4 Verification Complete ===")
