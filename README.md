# Silph Scope

*Developed by Silph Co., this device allows Pokemon trainers to uncover the true identities of ghosts.*

A Bayesian framework for regime-aware market allocation. It estimates a three-dimensional HMM across returns, realized volatility, and a conditioning signal using FFBS-within-Gibbs.  Dictate your exposure using probabilistic vol-target or Kelly allocations tied to any percentile from the full MCMC posterior.

---

## Model

Discrete regime $z_t \in \{1, \dots, K\}$ with transition matrix $P$. Trivariate Gaussian emission $y_t \mid z_t = k \sim \mathcal{N}_3(\mu_k, \Sigma_k)$, with a Normal-Inverse-Wishart prior on $(\mu_k, \Sigma_k)$ and a sticky Dirichlet prior on rows of $P$ (extra mass on the diagonal, biasing toward long dwell times).

Inference is block Gibbs: FFBS draws $z_{1:T}$ jointly, NIW conjugate updates yield $(\mu_k, \Sigma_k)$, and Dirichlet conjugate updates yield $P$ row-by-row. See Presse & Sgouralis (2023), Ch. 8.

The Gaussian mixture inherits the usual strengths of a regime model: regime-dependent means recover unconditional skew, regime-dependent variances recover unconditional kurtosis, and the regime-dependent return<->logRV covariance term captures the leverage effect. Within any single regime, returns remain Gaussian - the fat tails are a *between-regime* phenomenon, not a within-regime one.

### Use this for

Regime detection and position sizing in liquid markets, where Gaussian emissions are a reasonable local approximation and the interpretability of the fitted parameters matters for sizing decisions.

### Don't use this for

Tail-risk estimation, stress scenarios, VaR/ES, assets with heavy power-law returns.

---

## Structure

```
.
|-- src/
|   `-- silph_scope/
|       |-- utils.py      # FFBS, NIW sampler, MVN emission, helpers
|       |-- runner.py     # Gibbs + FFBS MCMC sampler
|       |-- backtest.py   # Rolling/expanding-window OOS backtest
|       `-- data.py       # Asset-specific data loaders
`-- tests/
    `-- test_utils.py     # Unit tests: core MCMC primitives in `utils.py`
```

---

## Quickstart

```bash
conda env create -f environment.yml
conda activate silph-scope
```

```python
from silph_scope.data import load_with_rv_lag
from silph_scope.backtest import run_backtest, print_summary
import logging

# Signal defaults to yesterday's log-RV
df = load_with_rv_lag('data/spy.csv')

result, refit_log = run_backtest(
    df,
    K=2,
    min_train=756,
    refit_every=21,
    sizing='vol_target',
    sigma_target=0.15,
)

print_summary(result, K=2, logger=logging.getLogger())
```

The training window expands until it hits `lookback` observations, then rolls forward at fixed width. Refits happen every `refit_every` days; between refits, the stored posterior is used to filter today's regime given observations up to yesterday.

---

## The three HMM channels

Every row fed to the model is $(r_t, \log \text{RV}_t, s_t)$:

| Channel | Meaning |
|---|---|
| `Return` | Log daily return |
| `LogRV` | Log annualised realised variance (Garman-Klass, Yang-Zhang, etc.; loader's choice) |
| `Signal` | Any conditioning variable: lagged RV, VIX level, VRP, credit spread, your own signal |

The emission covariance $\Sigma_k$ is estimated jointly across all three, so the model learns *regime-specific* correlations between returns, vol, and your signal.

---
## Data Loaders

The loaders below are reference implementations showing how to construct a DataFrame in the format `run_backtest` expects: columns `Return`, `LogRV`, `Signal`, `Close`, `Open`, subject to the [causality requirement](#causality) for `Signal`. They cover two common setups (lagged realised vol, VIX-derived), but the model is signal-agnostic — bring your own loader for any stationary, causally-available signal you like. See [BYO data](#byo-data) below.

`RealizedVar` should be daily variance (annualised ×252, then logged to produce `LogRV`). Any intraday-only estimator works — from tick-level realised variance down to a single-candle estimator like Parkinson or Rogers-Satchell off daily OHLC bars — as long as it excludes the overnight return, since the backtest's `f_overnight` gross-up assumes that.


### Causality

To prevent look-ahead bias, the model requires strict causality. The conditional emission (default) evaluates `Signal[t+1]` at each backtest step and treats it as information observable at the close of day `t`, used to size a position applied to day `t+1`'s return.


### Lagged log-RV (no auxiliary data)

Works for any asset with an OHLC + `RealizedVar` CSV. `LogRV` and `Signal` are the same quantity `lag` day(s) apart: the model treats them as independent channels and learns the lag-1 autocorrelation through their covariance.

```python
df = load_with_rv_lag('data/spy.csv', lag=1)
```

### VIX-derived signal

```python
df = load_with_vix(
    'data/spy.csv',
    'data/vix.csv',
    vix_mode='close_lag',   # 'open' | 'close_lag' | 'vrp_lag' | 'open_vrp'
)
```
`close_lag` and `vrp_lag` preserve [causality](#causality). `open` and `open_vrp` do **not** — they carry today's 9:31 AM print, which is not observable at close of the prior day. Use those modes only with `use_marginal_filter=True`, or with an open-rebalanced variant of the backtest (not currently implemented).

### BYO Data

Any DataFrame with columns `Return`, `LogRV`, `Signal`, `Close`, `Open` works directly, subject to the [causality requirement](#causality) above:

```python
result, refit_log = run_backtest(df, signal_col='Signal', ...)
```

---

## MCMC Sampler

To fit the model on a fixed window without running a backtest:

```python
from silph_scope.runner import fit_hmm, save_results
from pathlib import Path

Y = df[['Return', 'LogRV', 'Signal']].values

run_data = fit_hmm(
    Y,
    K=2,
    n_iter=5000,
    burn_in=1000,
    thin=2,
    seed=42,
)

# run_data keys: trace, trace_full, mu_list, Sigma_list, P, regimes, K
run_dir = Path('runs/my_run')
run_dir.mkdir(parents=True, exist_ok=True)
save_results(run_dir, run_data, config={'K': 2})
```
Transaction costs are modelled as a flat one-way bps charge applied to weight changes between rebalances. Slippage, borrow costs on shorts etc are not yet modelled.


`save_results` writes two files to `run_dir`:

| File | Description |
|---|---|
| `triv_hmm_mcmc.npz` | Full trace arrays compressed with `np.savez_compressed` |
| `triv_hmm_meta.json` | Config, K, posterior-final `mu_list`, `Sigma_list`, and `P` |

Load them back with:

```python
import numpy as np, json
from pathlib import Path

run_dir = Path('runs/my_run')
trace   = np.load(run_dir / 'triv_hmm_mcmc.npz')
meta    = json.loads((run_dir / 'triv_hmm_meta.json').read_text())
```

---

## Backtest

```python
result, refit_log = run_backtest(
    df,
    K=2,                      # number of regimes
    signal_col='Signal',      # third HMM column
    min_train=756,            # ~3 years before first refit
    refit_every=21,           # monthly refits
    lookback=2000,            # rolling training window
    sizing='vol_target',      # 'vol_target' or 'kelly'
    sigma_target=0.15,        # annualised vol target
    target_percentile=10.0,   # percentile of posterior weights to use
    allow_short=False,        # allow negative weights (kelly sizing only)
    use_marginal_filter=False, # marginalise signal out of daily filter
    max_weight=1.5,           # position cap
    tc_bps=1.0,               # one-way transaction cost
    n_iter=5000,
    burn_in=1000,
    thin=2,
    seed=42,
    nu0=6,
    kappa0=1.0,
)
```

`result` is the input DataFrame with added columns:

| Column | Description |
|---|---|
| `weight` | Position size set at close of day t |
| `vol_hat` | Forecast annualised vol for day t+1 |
| `strat_ret` | Strategy return on day t (after costs) |
| `prob_<regime>` | Filtered regime probabilities, one column per regime (e.g. `prob_clear_skies`, `prob_thunderstorm` for K=2; `prob_clear_skies`, `prob_sandstorm`, `prob_thunderstorm` for K=3) |

`refit_log` is a dict with dates, fitted parameters, and overnight fraction at each refit.


---

## CLI

```bash
# fit on a fixed window
python3 -m silph_scope.runner \
    --data data/spy-signals.csv \
    --K 2 --iters 5000 --burnin 1000

# full OOS backtest (vol-target, signal-conditioned filter by default)
python3 -m silph_scope.backtest \
    --data data/spy-signals.csv \
    --K 2 --min-train 756 --refit-every 21 \
    --sizing vol_target --sigma-target 0.15 \
    --outdir runs

# full OOS backtest (Kelly, 10th-percentile posterior weight, marginal filter)
python3 -m silph_scope.backtest \
    --data data/spy-signals.csv \
    --K 2 --min-train 756 --refit-every 21 \
    --sizing kelly --target-percentile 10 \
    --use-marginal-filter \
    --outdir runs
```

**Runner** output (`runs/triv_hmm_K{n}_<timestamp>/`):

```
triv_hmm_mcmc.npz   # compressed trace arrays
triv_hmm_meta.json  # K, final mu/Sigma/P, config
triv_hmm.log        # full run log
```

**Backtest** output (`runs/backtest_K{n}_<timestamp>/`):

```
backtest.csv       # full result DataFrame
refit_log.json     # per-refit parameters and f_overnight
config.json        # all CLI args
backtest.log       # full run log
```

---

## MCMC Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `K` | 2 | Number of regimes |
| `n_iter` | 5000 | Total Gibbs iterations |
| `burn_in` | 1000 | Iterations discarded |
| `thin` | 2 | Thinning factor |
| `nu0` | 6 | IW degrees of freedom (>= 4 for a 3D emission; higher means tighter prior on $\Sigma$) |
| `kappa0` | 1.0 | NIW prior strength on mean |
| `sticky_kappa` | 40 | Diagonal bump on the Dirichlet prior for $P$; implies prior expected dwell ~ 42 days at $K=2$ |
| `m0` | data mean | Length-3 prior mean `[mu_r, mu_v, mu_sig]` |

---

## Roadmap

- Port the analysis notebooks into `notebooks/`. Two exist in private repo - one for OOS backtest diagnostics (i.e. rolling Sharpe vs EWMA, parameter trajectories across refits, etc.), one for post-hoc MCMC diagnostics (i.e. posterior densities, pairwise subplots etc). Cleanup, refactor, QC, and commit.
- Deploy to paper trading to see if this (OOS) backtest billionaire Pokemon trainer is ready for the Elite Four, or if the model is just going to hurt itself in its confusion.
- Look into migrating from current "belt and suspenders" Conda/Pip setup to something like Pixi for a fully unified, single-file (`pyproject.toml`) package management architecture.
- Catch 'em all (edge cases, pitfalls, backtest plumbing, the long tail).

---

## References

Presse, S. and Sgouralis, I. (2023). *Data Modeling for the Sciences: Applications, basics, computations*. Cambridge University Press. Ch. 8 for the FFBS-within-Gibbs scheme used here.