# Proposition — concept dilution and redundancy (Step 1)

Setting. A concept is carried by a group $G$ of $m$ latents. A label depends on the concept through
$\eta = \beta\, \tilde S$, with $S = \sum_{j \in G} X_j$ and $\tilde S = (S - \mathbb{E}S)/\mathrm{sd}(S)$.
On the scale of standardised latents $\tilde X_j$ the coefficient of child $j$ is
$$\beta_j = \beta\,\frac{\mathrm{sd}(X_j)}{\mathrm{sd}(S)} .$$
The group null $H_G : Y \perp X_G \mid X_{-G}$ is false with an effect, $\beta$, that does not depend on $m$.

**(a) Dilution.** The concept fires with probability $\pi$; when it fires, exactly one child fires (chosen
uniformly), with magnitude $M$, $\mu_k = \mathbb{E}M^k$. Then
$$\mathrm{Var}(X_j) = \tfrac{\pi}{m}\mu_2 - \tfrac{\pi^2}{m^2}\mu_1^2,\qquad \mathrm{Var}(S) = \pi\mu_2 - \pi^2\mu_1^2,
\qquad \beta_j^2 = \frac{\beta^2}{m}\,\bigl(1 + O(\pi)\bigr).$$
Every child is conditionally non-null, but its standardised effect falls as $m^{-1/2}$ and its Fisher
information at fixed $n$ as $1/m$. So for fixed $n$ the power of any per-latent conditional test goes to
$0$ as $m$ grows, while the concept's effect is unchanged. **A group test keeps its power only if its
statistic pools within the group.** A statistic that penalises each coordinate separately before
aggregating, e.g. $W_G = \sum_{j\in G}|\hat\beta_j| - \sum_{j\in G}|\hat{\tilde\beta}_j|$ from an $\ell_1$ fit,
inherits the per-child shrinkage.

**(b) Redundancy.** All $m$ children co-fire as near-duplicates, $X_j = A\,e^{\varepsilon_j}$ with
$\varepsilon_j \sim N(0,\sigma^2)$. Then $\mathrm{sd}(S) \approx m\,\mathrm{sd}(X_j)$ and $\beta_j \approx \beta/m$.
As $\sigma \to 0$, each $X_j$ becomes a function of $X_{-j}$, so $H_j : Y \perp X_j \mid X_{-j}$ becomes
true for every child even though $H_G$ is false. Second-order knockoffs follow suit: as
$\mathrm{corr}(X_j, X_k) \to 1$ the per-variable MVR $s_j \to 0$, so $\tilde X_j \to X_j$ and $W_j \to 0$.
At the group level $S$ remains well conditioned and group knockoffs keep power.

**MKF+.** A latent is selected only if it passes every layer, so multilayer power at the latent level is
bounded by per-latent power. MKF+ inherits both failures.

## Numerical check (`results/toy_report.json`, `fig_c1_toy.png`)

$n = 20{,}000$, $p = 1024$, $K = 20$ planted concepts, $\pi = 0.05$, $q = 0.1$, 50 replicates; Gaussian
second-order knockoffs (block MVR).

| m | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| dilution: per-child effect (theory $m^{-1/2}$) | 1.000 | 0.712 (0.707) | 0.504 (0.500) | 0.355 (0.354) | 0.250 (0.250) | 0.174 (0.177) |
| redundancy: per-child effect (theory $m^{-1}$) | 1.000 | 0.501 (0.500) | 0.251 (0.250) | 0.126 (0.125) | 0.063 (0.063) | 0.031 (0.031) |
| redundancy: mean MVR $s$, per-latent / group | 0.998 / 0.998 | 0.089 / 0.373 | 0.040 / 0.257 | 0.028 / 0.205 | 0.020 / 0.153 | 0.017 / 0.152 |

Concept power at amplitude 3:

| m | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| dilution — per-latent | 1.00 | 1.00 | 1.00 | 0.90 | 0.01 | 0.00 |
| dilution — group (lasso $W_G$) | 1.00 | 1.00 | 1.00 | 0.89 | 0.00 | 0.00 |
| dilution — group-sum statistic (exploratory) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| redundancy — per-latent | 1.00 | 1.00 | 0.73 | 0.01 | 0.00 | 0.00 |
| redundancy — group (lasso $W_G$) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| MKF+ (c = 1), both mechanisms | identical to per-latent | | | | | |

FDR was 0.000 at both levels in every zero-inflated cell. The library unit test (rule T1, Gaussian data,
where knockoffs are exactly valid) passed: no BY-significant breach in 288 global-null and 864 planted
cells.

**Status of the pre-registered prediction** ("per-latent power decays in m; group power is flat"):
confirmed for redundancy, **refuted for dilution with the pre-registered group statistic**. The
group-sum statistic restores flat power in both mechanisms, but that comparison is exploratory, and
favourable by construction because toy labels are functions of $S$. Its confirmatory test is the Step 7
benchmark (`concept_amendment_2`).
