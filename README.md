# switchback

A Python package for **designing and analyzing switchback experiments** —
A/B tests where the same unit (a city, a marketplace, a server) is
alternately assigned to treatment and control across consecutive time
periods. They are the workhorse for measuring interventions in systems
with temporal interference, where unit-level randomization would
contaminate control with treatment spillovers.

The package is organized into four pluggable modules:

| module | role |
|---|---|
| `switchback.dgp` | **data generating processes** — given an assignment path of length `T`, return the observed outcomes |
| `switchback.design` | **designs** — produce assignment paths |
| `switchback.estimators` | **estimators** — map `(W, Y)` to a treatment-effect estimate |
| `switchback.decisions` | **decisions** — estimate the variance and construct confidence intervals |

## Install

```bash
pip install -e .
```

## 60-second quickstart

The one-call front door is `decide(design, estimator, W, Y, alpha)` — it
queries the point estimate, dispatches the appropriate design-derived
variance estimator, and returns a confidence interval, all in one shot:

```python
from switchback.design import AdaptiveBlockDesign
from switchback.estimators import IPWEstimator
from switchback.dgp.state_space import LatentStateDGP
from switchback.decisions import decide

design = AdaptiveBlockDesign(B=24, rho=0.5, seed=0)   # 24 seasonal blocks × K days
dgp    = LatentStateDGP(gamma=0.2, seed=0)            # AR(1) latent state DGP
est    = IPWEstimator(design, m=1)                    # paper's primary form

W = design.sample(672)              # length-T assignment path (T = B·K = 24·28)
Y = dgp.generate(W)                 # observed outcomes

result = decide(design, est, W, Y, alpha=0.05)
print(result.estimate, result.variance, result.ci)
# → 0.987..., 0.0272..., (0.664..., 1.310...)
```

For BernoulliDesign / CompleteRandomization the call is identical — `decide`
auto-dispatches to the appropriate variance estimator under the hood.

## Designs (`switchback.design`)

All designs expose `design.sample(T) -> np.ndarray` and a propensity oracle
`design.consecutive_prob(T, t, p, value)` for the IPW estimator.

| class | parameters | what it does |
|---|---|---|
| `BernoulliDesign(l, p, seed)` | `l: int` (time window length), `p ∈ [0, 1]` | Partition `[0, T)` into windows of length `l`; within each window draw one `Bern(p)` coin and apply it to every period. `l=1` is per-period i.i.d. Bernoulli. |
| `CompleteRandomization(l, seed)` | `l: int` (time window length) | Same window partition, but the treated count is **fixed**: exactly half of the `T/l` windows are treated, sampled without replacement. |
| `AdaptiveBlockDesign(B, rho, seed)` | `B: int` (seasonal blocks), `rho ∈ [0.5, 1]` | Ni & Bojinov's model-assisted design. `B` seasonal blocks (e.g. 24 hours of the day) × `K = T/B` days. Block 0 sampled by CR; block `b ≥ 1` sampled adaptively given block `b−1` to maintain exactly `ρ·K` consecutive same-arm pairs. |

## DGPs (`switchback.dgp`)

| class | model |
|---|---|
| `SimpleDGP(mu, tau, sigma)` | `Y_t = μ_t + τ·W_t + ε_t`, iid noise, no carryover. Baseline / sanity-check. |
| `CarryoverDGP(betas, mu, sigma)` | `Y_t = μ_t + Σ_{k=0}^{q} β_k W_{t-k} + ε_t`. Finite-order linear carryover of order `q = len(betas) − 1`. |
| `LatentStateDGP(mu, beta_0, alpha_0, gamma, sigma_y, sigma_h, h0, seed)` | `Y_t = μ_t + h_t + β_0·W_t + ε^Y_t`, `h_t = γ·h_{t-1} + α_0·W_t + ε^h_t`. Geometric ("infinite") carryover through the AR(1) latent state. |

## Estimators (`switchback.estimators`)

All estimators take `(design, m)` where `m` is the **burn-in length** —
how many consecutive same-arm periods are required for a valid
observation. **The constructor enforces `m ≤ design.l`** since
no validity is guaranteed beyond a single window.

| class | what it does |
|---|---|
| `IPWEstimator(design, m)` | Horvitz–Thompson IPW. `m=1` (the paper's primary form) targets the lag-1 contiguous same-arm contrast. |
| `HajekEstimator(design, m)` | Stratified Hájek (normalized IPW). Shift-invariant in `Y`; auto-stratifies when the burn-in window straddles a design-window boundary. |
| `StratifiedHajekEstimator(design, m)` | Hájek within each stratum, aggregated by realised-size weights. |

For `AdaptiveBlockDesign` the natural configuration is `m = l = 1`
(the implicit window length is 1; the design's `ρ` controls the consecutive
same-arm propensity).

## Decisions (`switchback.decisions`)

The decisions module estimates the design-based variance and constructs
confidence intervals from a point estimate. Two design-derived variance
estimators are available, dispatched automatically by the `decide` front
door:

| design | variance estimator | notes |
|---|---|---|
| `AdaptiveBlockDesign` | `block_variance` | Eq. 28 of Ni & Bojinov with block-0 boundary fixes (1/π_b weight + count-variance correction), per-pair γ⁺/γ⁻ matrix from the Class A/B/C structural classification, and factor-2 mirror at forward δ ≤ B/2. Auto-truncates max_delta based on chain decay. |
| `BernoulliDesign`, `CompleteRandomization` | `HACVariance` | Newey-West HAC on per-window influence sequences. Rejects `AdaptiveBlockDesign` at construction. |

Public API:

```python
from switchback.decisions import (
    decide,           # one-call front door (recommended)
    DecisionResult,     # dataclass: .estimate, .variance, .ci, .alpha
    block_variance,      # AdaptiveBlockDesign variance estimator
    HACVariance,         # HAC variance for window-structured designs
    normal_ci,           # (estimate, variance, alpha) -> CI
    confidence_interval, # convenience wrapper around HACVariance
    block_confidence_interval,  # convenience wrapper around block_variance
)
```

### Quick reference

```python
# One-call decide (dispatches by design type)
result = decide(design, estimator, W, Y, alpha=0.05)
result.estimate, result.variance, result.ci, result.alpha

# Step-by-step (if you want the intermediate values)
tau_hat = estimator.fit(W, Y).estimate_
v_hat   = block_variance(design, W, Y)     # for AdaptiveBlockDesign
# or:    HACVariance(design, est).fit(W, Y).variance_  # for window designs
lo, hi  = normal_ci(tau_hat, v_hat, alpha=0.05)
```

## Calibration

Under design-based variance estimation at `B=24, K=28` (the package's standing rule
for `AdaptiveBlockDesign` tests), `block_variance` calibrates to within
~1% of empirical `Var_W(τ̂ | noise)` across:

| DGP regime | mean V̂ / Var ratio |
|---|---|
| LatentStateDGP, γ=0.2, ρ=0.5 | **1.012** |
| LatentStateDGP, γ=0.2, ρ=0.75 | **1.001** |
| LatentStateDGP, γ=0.5, ρ=0.7 | **1.010** |

across 10 noise seeds × 12,000 W draws each.

## Running the tests

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

148 tests covering: DGP sanity, design propensities, estimator
correctness, variance calibration, boundary corrections, CI math, and
the `m ≤ l` and dispatch invariants.

## References

Ni, Tu and Iavor Bojinov. *Enhancing Efficiency and Robustness for
Switchback Experiments: A Model-assisted Design and Analysis*.

## License

MIT.
