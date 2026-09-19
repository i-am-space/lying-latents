"""Stage 3 — Realised FDR against known ground truth (planted signal).

Evaluates whether the finite-sample FDR guarantee of Gaussian second-order knockoffs
holds empirically, or whether the exchangeability violation demonstrated in Stage 2
inflates realised false discoveries above the nominal target q in {0.05, 0.10, 0.20}.

The real activation matrix X (preserving zero-atom, tails, and feature correlation)
is retained, while synthetic binary labels Y are generated from a planted subset S
of true signals. All latents outside S are conditionally null by construction.

Usage: python src/planted_fdr.py --config config/default.yaml [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import cache_path, load_config, results_dir, rng_for
from knockoff_audit import estimate_cov, make_knockoffs, standardise


# ---------------------------------------------------------------------------
# Planted Signal Label Generator
# ---------------------------------------------------------------------------

def generate_planted_labels(
    X: np.ndarray,
    S: np.ndarray,
    form: str,
    amplitude: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic binary task labels Y = f(X_S) + noise.

    Only latents in S carry signal. Latents outside S are conditionally null.
    """
    n = X.shape[0]
    k = len(S)
    XS = X[:, S]  # (n, k)

    # Random signs for linear weights
    w = rng.choice([-1.0, 1.0], size=k)

    if form == "linear":
        logits = (XS @ w) / np.sqrt(k) * amplitude
    elif form == "interaction":
        # Linear part + pairwise product interactions
        lin_part = (XS @ w) / np.sqrt(k)
        inter_part = np.zeros(n, dtype=np.float64)
        n_pairs = 0
        for i in range(0, k - 1, 2):
            gamma = rng.choice([-1.0, 1.0])
            pair_prod = XS[:, i] * XS[:, i + 1]
            inter_part += gamma * pair_prod
            n_pairs += 1
        if n_pairs > 0:
            sd_inter = np.std(inter_part)
            if sd_inter > 1e-8:
                inter_part = inter_part / sd_inter
            logits = (0.7 * lin_part + 0.3 * inter_part) * amplitude
        else:
            logits = lin_part * amplitude
    else:
        raise ValueError(f"Unknown functional form: {form!r}")

    # Center logits for approximate 50/50 class balance
    logits = logits - np.median(logits)
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
    y = (rng.random(n) < prob).astype(np.float32)
    return y, prob


# ---------------------------------------------------------------------------
# Fast L1-Logistic Regression (Lasso) via FISTA on GPU / CPU
# ---------------------------------------------------------------------------

def fit_lasso(
    Z_t: torch.Tensor,
    y_t: torch.Tensor,
    lam: float = 0.02,
    lr: float = 0.05,
    max_iter: int = 500,
) -> torch.Tensor:
    """Fit L1-logistic regression on [X, Xk] via FISTA (accelerated proximal gradient).

    Z_t: (n, 2p) standardized features
    y_t: (n,) float32 {0, 1}
    Returns: w of shape (2p,)
    """
    n, dim = Z_t.shape
    w = torch.zeros(dim, device=Z_t.device, dtype=Z_t.dtype)
    b = torch.zeros(1, device=Z_t.device, dtype=Z_t.dtype)
    w_prev = w.clone()

    for t in range(1, max_iter + 1):
        # Nesterov momentum
        beta = (t - 1.0) / (t + 2.0)
        v = w + beta * (w - w_prev)
        w_prev.copy_(w)

        # Gradient of logistic loss
        logits = Z_t @ v + b
        p = torch.sigmoid(logits)
        err = (p - y_t) / n
        grad_w = Z_t.t() @ err
        grad_b = err.sum()

        # Proximal step with soft-thresholding
        u = v - lr * grad_w
        b = b - lr * grad_b
        thresh = lr * lam
        w = torch.sign(u) * torch.clamp(u.abs() - thresh, min=0.0)

    return w


# ---------------------------------------------------------------------------
# Knockoff+ Filter
# ---------------------------------------------------------------------------

def compute_knockoff_plus_threshold(W: np.ndarray, q: float) -> float:
    """Compute the Knockoff+ threshold tau for nominal FDR target q.

    tau = min { t > 0 : (1 + #{j : W_j <= -t}) / max(1, #{j : W_j >= t}) <= q }
    """
    pos_W = np.sort(np.abs(W[W != 0]))
    if len(pos_W) == 0:
        return float("inf")

    for t in pos_W:
        num = 1.0 + np.sum(W <= -t)
        den = max(1.0, float(np.sum(W >= t)))
        if num / den <= q:
            return float(t)
    return float("inf")


def evaluate_discoveries(
    discovered: set[int],
    true_set: set[int],
) -> tuple[float, float]:
    """Calculate Realised FDR and Power given discovery set and ground-truth signal set."""
    n_disc = len(discovered)
    n_true_signals = len(true_set)

    if n_disc == 0:
        return 0.0, 0.0

    false_disc = len(discovered - true_set)
    true_disc = len(discovered & true_set)

    fdr = false_disc / n_disc
    power = true_disc / max(1, n_true_signals)
    return float(fdr), float(power)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def make_stage3_figures(
    results_dict: dict,
    rd: Path,
) -> None:
    """Generate Fig 7 (Nominal vs Realised FDR) and Fig 8 (Power vs Signal Size)."""
    records = results_dict["records"]
    signal_sizes = results_dict["signal_sizes"]
    fdr_targets = results_dict["nominal_fdr_targets"]
    forms = results_dict["functional_forms"]

    # --- Fig 7: Nominal vs Realised FDR ---
    fig, axes = plt.subplots(1, len(forms), figsize=(6 * len(forms), 5), sharey=True)
    if len(forms) == 1:
        axes = [axes]

    colors = {10: "#3b6ea5", 20: "#e07b39", 30: "#5b8c5a"}

    for ax, form in zip(axes, forms):
        # Diagonal line: ideal FDR control
        ax.plot([0, 0.25], [0, 0.25], "k--", label="Ideal Control (Realised = Nominal)", alpha=0.7)

        for k in signal_sizes:
            fdrs_mean = []
            fdrs_err = []
            for q in fdr_targets:
                subset = [r for r in records if r["signal_size"] == k and r["form"] == form and r["q"] == q]
                vals = [r["knockoff_fdr"] for r in subset]
                fdrs_mean.append(np.mean(vals))
                fdrs_err.append(np.std(vals) / np.sqrt(len(vals)))

            ax.errorbar(
                fdr_targets,
                fdrs_mean,
                yerr=fdrs_err,
                marker="o",
                label=f"|S| = {k} signals",
                color=colors.get(k, "blue"),
                capsize=4,
                lw=1.5,
            )

        ax.set_title(f"Knockoff FDR Control ({form.capitalize()})")
        ax.set_xlabel("Nominal FDR Target (q)")
        ax.set_ylabel("Realised FDR")
        ax.set_xlim(0, 0.25)
        ax.set_ylim(-0.02, max(0.4, max(r["knockoff_fdr"] for r in records) * 1.15))
        ax.legend(fontsize=9)
        ax.grid(True, ls=":", alpha=0.5)

    fig.suptitle("Stage 3: Realised False Discovery Rate on Semi-Synthetic Ground Truth", y=1.02)
    fig.tight_layout()
    fig.savefig(rd / "fig7_fdr_control.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 8: Statistical Power Comparison ---
    fig, axes = plt.subplots(1, len(forms), figsize=(6 * len(forms), 5), sharey=True)
    if len(forms) == 1:
        axes = [axes]

    for ax, form in zip(axes, forms):
        for q in fdr_targets:
            powers_mean = []
            powers_err = []
            for k in signal_sizes:
                subset = [r for r in records if r["signal_size"] == k and r["form"] == form and r["q"] == q]
                vals = [r["knockoff_power"] for r in subset]
                powers_mean.append(np.mean(vals))
                powers_err.append(np.std(vals) / np.sqrt(len(vals)))

            ax.errorbar(
                signal_sizes,
                powers_mean,
                yerr=powers_err,
                marker="s",
                label=f"q = {q}",
                capsize=4,
                lw=1.5,
            )

        ax.set_title(f"Knockoff Power ({form.capitalize()})")
        ax.set_xlabel("Planted Signal Size (|S|)")
        ax.set_ylabel("Statistical Power (True Positive Rate)")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xticks(signal_sizes)
        ax.legend(fontsize=9)
        ax.grid(True, ls=":", alpha=0.5)

    fig.suptitle("Stage 3: Statistical Power across Signal Sizes", y=1.02)
    fig.tight_layout()
    fig.savefig(rd / "fig8_power_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 — Realised FDR on semi-synthetic ground truth")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config YAML")
    parser.add_argument("--cache", default=None, help="Override path to activation cache")
    parser.add_argument("--device", default=None, help="torch device, e.g. 'cuda' or 'cpu'")
    parser.add_argument("--limit-reps", type=int, default=None, help="Override number of replicates for quick testing")
    args = parser.parse_args()

    # Device selection
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Config & Paths
    cfg = load_config(args.config)
    rd = results_dir(cfg)
    cp = args.cache or cache_path(cfg)
    if not cp.exists() if hasattr(cp, "exists") else True:
        raise SystemExit(f"Cache not found at {cp}. Run src/cache_activations.py first.")

    d = np.load(cp, allow_pickle=True)
    prim = cfg["aggregation"]["primary"]
    X_raw = d[f"X_{prim}"].astype(np.float64)
    n_total, p = X_raw.shape
    print(f"Loaded activations: n={n_total}, p={p} (Aggregator: {prim})")

    cfg_s3 = cfg["stage3"]
    n_samples = min(cfg_s3.get("n_samples", n_total), n_total)
    n_reps = args.limit_reps or cfg_s3["replicates"]
    signal_sizes = cfg_s3["signal_sizes"]
    fdr_targets = cfg_s3["nominal_fdr_targets"]
    forms = cfg_s3["functional_forms"]
    amplitude = float(cfg_s3.get("signal_amplitude", 3.0))

    # Standardize X
    X_sub = X_raw[:n_samples]
    Z, mu, sd = standardise(X_sub)
    print(f"Standardized features: shape={Z.shape}")

    # Step 1: Generate Gaussian Knockoffs (Model-X unsupervised property)
    print("\n=== Generating Gaussian Second-Order Knockoffs (knockpy) ===")
    t0 = time.time()
    Sigma = estimate_cov(Z, "ledoit_wolf")
    seed_k = int(rng_for(cfg, "knockoff_sampler").integers(2**31))
    Zk, S_mat = make_knockoffs(Z, Sigma, cfg["knockoffs"]["s_method"], seed_k)
    print(f"Knockoffs generated in {time.time() - t0:.1f}s | mean s = {np.diag(S_mat).mean():.4f}")

    # Full concatenated design matrix [X, Xk]
    ZZ = np.hstack([Z, Zk]).astype(np.float32)
    ZZ_t = torch.from_numpy(ZZ).to(device)

    # Optional: Stage 1 baseline threshold for comparison if available
    s1_path = rd / "stage1_calibration.json"
    wy_threshold = None
    if s1_path.exists():
        try:
            s1_data = json.loads(s1_path.read_text())
            wy_threshold = s1_data.get("permutation", {}).get("mean_diff", {}).get("max_score_null_q95")
            print(f"Stage 1 Westfall-Young 95th threshold loaded: {wy_threshold:.4f}")
        except Exception:
            pass

    rng_plant = rng_for(cfg, "planted_signal")

    # Replicate Loop
    records = []
    print(f"\n=== Running Stage 3 Benchmark ({n_reps} reps, |S| in {signal_sizes}, forms in {forms}) ===")
    start_all = time.time()

    total_runs = len(forms) * len(signal_sizes) * n_reps
    run_idx = 0

    for form in forms:
        for k in signal_sizes:
            for rep in range(n_reps):
                run_idx += 1
                # 1. Select ground-truth signals S
                S_idx = np.sort(rng_plant.choice(p, size=k, replace=False))
                true_set = set(S_idx.tolist())

                # 2. Generate labels Y
                y_np, _ = generate_planted_labels(Z, S_idx, form, amplitude, rng_plant)
                y_t = torch.from_numpy(y_np).to(device)

                # 3. Fit Lasso on [X, Xk]
                w = fit_lasso(
                    ZZ_t,
                    y_t,
                    lam=cfg_s3.get("lasso_lambda", 0.02),
                    lr=cfg_s3.get("lasso_lr", 0.05),
                    max_iter=cfg_s3.get("lasso_max_iter", 500),
                )
                w_np = w.cpu().numpy()

                # 4. Feature contrast statistics W_j = |beta_j| - |beta_tilde_j|
                w_real = w_np[:p]
                w_knock = w_np[p:]
                W = np.abs(w_real) - np.abs(w_knock)

                # Optional: Mean Difference scores for Westfall-Young comparator
                mask1 = y_np == 1.0
                mean_diff_scores = np.abs(Z[mask1].mean(axis=0) - Z[~mask1].mean(axis=0))

                # 5. Evaluate for each nominal target q
                for q in fdr_targets:
                    tau = compute_knockoff_plus_threshold(W, q)
                    if np.isinf(tau):
                        discovered_ko = set()
                    else:
                        discovered_ko = set(np.where(W >= tau)[0].tolist())

                    fdr_ko, pwr_ko = evaluate_discoveries(discovered_ko, true_set)

                    # Stage 1 Westfall-Young comparison
                    if wy_threshold is not None:
                        discovered_wy = set(np.where(mean_diff_scores >= wy_threshold)[0].tolist())
                        fdr_wy, pwr_wy = evaluate_discoveries(discovered_wy, true_set)
                    else:
                        fdr_wy, pwr_wy = None, None

                    records.append({
                        "form": form,
                        "signal_size": k,
                        "rep": rep,
                        "q": q,
                        "knockoff_tau": float(tau) if not np.isinf(tau) else -1.0,
                        "knockoff_n_discoveries": len(discovered_ko),
                        "knockoff_fdr": fdr_ko,
                        "knockoff_power": pwr_ko,
                        "wy_n_discoveries": len(discovered_wy) if discovered_wy is not None else 0,
                        "wy_fdr": fdr_wy,
                        "wy_power": pwr_wy,
                    })

                if run_idx % 10 == 0 or run_idx == total_runs:
                    elapsed = time.time() - start_all
                    eta = (elapsed / run_idx) * (total_runs - run_idx)
                    print(
                        f"  [{run_idx:>3}/{total_runs}] Form={form:<11} |S|={k:<2} rep={rep:<2} | "
                        f"Elapsed: {elapsed/60:.1f}m ETA: {eta/60:.1f}m",
                        flush=True,
                    )

    total_time = time.time() - start_all
    print(f"\nStage 3 benchmark completed in {total_time/60:.1f} minutes.")

    # Aggregate summaries by condition
    condition_summary = []
    for form in forms:
        for k in signal_sizes:
            for q in fdr_targets:
                subset = [r for r in records if r["form"] == form and r["signal_size"] == k and r["q"] == q]
                fdrs = [r["knockoff_fdr"] for r in subset]
                powers = [r["knockoff_power"] for r in subset]
                discs = [r["knockoff_n_discoveries"] for r in subset]

                fdr_mean = float(np.mean(fdrs))
                fdr_se = float(np.std(fdrs) / np.sqrt(len(fdrs)))
                is_controlled = fdr_mean <= (q + 1.96 * fdr_se)

                condition_summary.append({
                    "form": form,
                    "signal_size": k,
                    "nominal_q": q,
                    "realised_fdr_mean": fdr_mean,
                    "realised_fdr_se": fdr_se,
                    "power_mean": float(np.mean(powers)),
                    "power_se": float(np.std(powers) / np.sqrt(len(powers))),
                    "mean_discoveries": float(np.mean(discs)),
                    "fdr_controlled": is_controlled,
                })

    results_dict = {
        "config_hash": str(d["config_hash"]),
        "master_seed": cfg["master_seed"],
        "n_samples": n_samples,
        "p": p,
        "signal_sizes": signal_sizes,
        "nominal_fdr_targets": fdr_targets,
        "functional_forms": forms,
        "replicates": n_reps,
        "elapsed_minutes": round(total_time / 60, 2),
        "conditions": condition_summary,
        "records": records,
    }

    # Save JSON and generate figures
    out_json_path = rd / "stage3_planted_fdr.json"
    out_json_path.write_text(json.dumps(results_dict, indent=2))
    print(f"Wrote summary results to {out_json_path}")

    make_stage3_figures(results_dict, rd)
    print(f"Generated figures {rd}/fig7_fdr_control.png and {rd}/fig8_power_comparison.png")

    # Print summary table to console
    print("\n" + "=" * 75)
    print(f"{'Form':<12} {'|S|':<4} {'Nominal q':<10} {'Realised FDR':<16} {'Power':<12} {'Controlled?':<12}")
    print("-" * 75)
    for c in condition_summary:
        fdr_str = f"{c['realised_fdr_mean']:.3f} ± {c['realised_fdr_se']:.3f}"
        pwr_str = f"{c['power_mean']:.3f} ± {c['power_se']:.3f}"
        ctrl_str = "YES" if c["fdr_controlled"] else "NO (INFLATED)"
        print(f"{c['form']:<12} {c['signal_size']:<4} {c['nominal_q']:<10} {fdr_str:<16} {pwr_str:<12} {ctrl_str:<12}")
    print("=" * 75)


if __name__ == "__main__":
    main()
