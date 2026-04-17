#!/usr/bin/env python3
"""
Rolling/expanding-window OOS backtest for the Trivariate HMM vol-targeting strategy.

The training window expands until it reaches ``lookback`` observations, then
rolls forward at a fixed width.

DataFrame contract:
The input DataFrame must have at minimum:
    Return  — log daily returns
    LogRV   — log annualised realised variance
    Signal  — any conditioning signal (column name is configurable)
    Close   — closing price  (used for overnight fraction estimate)
    Open    — opening price  (used for overnight fraction estimate)

The Signal column sharpens regime identification during MCMC refits but is
excluded from the live daily filter — only (Return, LogRV) are used there.
The vol forecast is derived exclusively from LogRV (column 1).

Usage:
    python3 -m silph_scope.backtest --data data/spy-signals.csv --K 2
    python3 -m silph_scope.backtest --data data/spy-signals.csv --K 3 --sizing kelly
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import hmm_log_emission, forward_filter, _regime_label
from .runner import fit_hmm, setup_logging

_REQUIRED_COLS = {'Return', 'LogRV', 'Close', 'Open'}

def _extract_params(trace: dict, K: int) -> tuple[list, list, np.ndarray]:
    """Posterior-mean mu_list, Sigma_list, P from a trace dict."""
    mu_list, Sigma_list = [], []
    for k in range(K):
        mu_r = float(np.mean(trace[f'mu_r{k}']))
        mu_v = float(np.mean(trace[f'mu_v{k}']))
        mu_s = float(np.mean(trace[f'mu_sig{k}']))
        sr   = float(np.mean(trace[f'sigma_r{k}']))
        sv   = float(np.mean(trace[f'sigma_v{k}']))
        ss   = float(np.mean(trace[f'sigma_sig{k}']))
        r_rv = float(np.mean(trace[f'rho_rv{k}']))
        r_rs = float(np.mean(trace[f'rho_rsig{k}']))
        r_vs = float(np.mean(trace[f'rho_vsig{k}']))
        mu_list.append(np.array([mu_r, mu_v, mu_s]))
        Sigma_list.append(np.array([
            [sr**2,       r_rv*sr*sv, r_rs*sr*ss],
            [r_rv*sr*sv,  sv**2,      r_vs*sv*ss],
            [r_rs*sr*ss,  r_vs*sv*ss, ss**2      ],
        ]))
    P = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i != j:
                P[i, j] = float(np.mean(trace[f'P{i}{j}']))
        P[i, i] = 1.0 - sum(P[i, j] for j in range(K) if j != i)
    return mu_list, Sigma_list, P

def _estimate_f_overnight(df: pd.DataFrame) -> float:

    opens = df['Open'].to_numpy()
    closes = df['Close'].to_numpy()
    dates = df.index.to_numpy()
    
    mask = np.diff(dates) == np.timedelta64(1, 'D')
    
    prev_closes = closes[:-1][mask]
    opens_today = opens[1:][mask]
    closes_today = closes[1:][mask]
    
    o = np.log(opens_today / prev_closes)
    c = np.log(closes_today / prev_closes)
    
    return float(np.var(o) / np.var(c))

def _conditional_emission(
    y_t: np.ndarray,
    s_t: float,
    mu_list: list[np.ndarray],
    Sigma_list: list[np.ndarray],
    K: int,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Log p(r_t, log_rv_t, s_t | S_t=k).
    
    Evaluates the full 3D joint likelihood for regime identification, while
    returning the conditional 2D parameters for downstream volatility forecasting.
    """
    
    # Extract today's observed signal
    
    mu_cond_list  = []
    Sig_cond_list = []
    
    for k in range(K):
        mu    = mu_list[k]
        Sigma = Sigma_list[k]
        
        # 1. Partition the mean
        mu_x = mu[:2]         # [Return, LogRV]
        mu_y = mu[2]          # Scalar: [Signal]
        
        # 2. Partition the covariance
        Sig_xx = Sigma[:2, :2]  # 2x2 matrix
        Sig_xy = Sigma[:2, 2]   # 1D array (length 2)
        Sig_yx = Sigma[2, :2]   # 1D array (length 2)
        Sig_yy = Sigma[2, 2]    # Scalar (variance of Signal)
        
        # 3. Calculate Conditional Parameters
        inv_Sig_yy = 1.0 / Sig_yy
        
        mu_cond  = mu_x + Sig_xy * inv_Sig_yy * (s_t - mu_y)
        Sig_cond = Sig_xx - np.outer(Sig_xy, Sig_yx) * inv_Sig_yy
        
        mu_cond_list.append(mu_cond)
        Sig_cond_list.append(Sig_cond)
        
    # 4. Evaluate the FULL 3D observation against the FULL parameters
    # to properly capture the regime-discriminating power of the signal
    ll = hmm_log_emission(y_t.reshape(1, -1), mu_list, Sigma_list)[0]
    
    return ll, mu_cond_list, Sig_cond_list

def _bivariate_emission(
    y_t: np.ndarray,
    mu_list: list[np.ndarray],
    Sigma_list: list[np.ndarray],
    K: int,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Log p(r_t, log_rv_t | S_t=k) using the (Return, LogRV) marginal.

    The Signal dimension is marginalised out — only the top-left 2×2 block
    of each regime's covariance is used, keeping the live filter causal.

    Returns ll (K,), mu2_list, Sig2_list.
    """
    mu2_list  = [mu_list[k][:2]        for k in range(K)]
    Sig2_list = [Sigma_list[k][:2, :2] for k in range(K)]
    ll        = hmm_log_emission(y_t[:2].reshape(1, -1), mu2_list, Sig2_list)[0]
    return ll, mu2_list, Sig2_list

def _filter_step(alpha: np.ndarray, ll_t: np.ndarray, P: np.ndarray) -> np.ndarray:
    """One incremental forward-filter update."""
    alpha_pred = alpha @ P
    log_joint  = np.log(np.maximum(alpha_pred, 1e-300)) + ll_t
    log_joint -= log_joint.max()
    alpha_new  = np.exp(log_joint)
    return alpha_new / alpha_new.sum()

def _reinit_filter(
    Y: np.ndarray,
    t: int,
    mu_list: list[np.ndarray],
    Sigma_list: list[np.ndarray],
    P: np.ndarray,
    K: int,
) -> np.ndarray:
    """Replay the bivariate filter from t=0 up to (not including) t."""
    if t == 0:
        return np.full(K, 1.0 / K)
    mu2  = [mu_list[k][:2]        for k in range(K)]
    sig2 = [Sigma_list[k][:2, :2] for k in range(K)]
    ll   = hmm_log_emission(Y[:t, :2], mu2, sig2)
    return forward_filter(ll, P)[-1]

def _kelly_trace_percentile(
    trace: dict,
    pred_probs: np.ndarray,
    K: int,
    percentile: float = 10.0,
) -> float:
    """Compute the Kelly weight at a given percentile across posterior samples.

    For each MCMC draw, computes the mixture Kelly weight using that draw's
    per-regime return means and variances, then returns the requested percentile
    over the full posterior.
    """
    # (n_samples, K)
    mu_r_mat  = np.column_stack([trace[f'mu_r{k}']    for k in range(K)])
    sr_sq_mat = np.column_stack([trace[f'sigma_r{k}'] for k in range(K)]) ** 2

    # (n_samples,)
    raw_mu = mu_r_mat @ pred_probs
    var_r  = (sr_sq_mat @ pred_probs
              + ((mu_r_mat - raw_mu[:, None]) ** 2) @ pred_probs)
    raw_w  = raw_mu / np.maximum(var_r, 1e-8)
    return float(np.percentile(raw_w, percentile))

def _vol_target_trace_percentile(
    trace: dict,
    pred_probs: np.ndarray,
    K: int,
    sigma_target: float,
    f_overnight: float,
    percentile: float = 10.0,
) -> float:
    """Compute the vol-target weight at a given percentile across posterior samples.

    For each MCMC draw, computes the mixture vol forecast and derives the
    corresponding vol-target weight, then returns the requested percentile.
    Low percentile = conservative (implied by high vol draws).
    """
    # (n_samples, K)
    mu_v_mat  = np.column_stack([trace[f'mu_v{k}']    for k in range(K)])
    sv_sq_mat = np.column_stack([trace[f'sigma_v{k}'] for k in range(K)]) ** 2

    # (n_samples,)
    forecast_log_rv = mu_v_mat @ pred_probs
    var_within      = sv_sq_mat @ pred_probs
    var_between     = ((mu_v_mat - forecast_log_rv[:, None]) ** 2) @ pred_probs
    jensen          = 0.5 * (var_within + var_between)

    vol_hat = np.sqrt(np.exp(forecast_log_rv + jensen)) / np.sqrt(max(1.0 - f_overnight, 1e-6))
    weights = sigma_target / vol_hat
    return float(np.percentile(weights, percentile))

def run_backtest(
    df: pd.DataFrame,
    K: int = 2,
    signal_col: str = 'Signal',
    min_train: int = 756,
    refit_every: int = 21,
    lookback: int = 2000,
    sigma_target: float = 0.15,
    max_weight: float = 1.5,
    tc_bps: float = 1.0,
    sizing: str = 'vol_target',
    target_percentile: float = 10.0,
    allow_short: bool = False,
    use_marginal_filter: bool = False,
    n_iter: int = 5000,
    burn_in: int = 1000,
    thin: int = 2,
    seed: int = 42,
    nu0: int = 6,
    kappa0: float = 1.0,
    sticky_kappa: float = 10.0,
    m0: Optional[list] = None,
    estimate_f_overnight: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Rolling/expanding-window OOS backtest for the Trivariate HMM.

    The training window expands until it reaches ``lookback`` observations, then
    rolls forward at a fixed width.  weight[t] is determined at close of day t
    and applied to day t+1's return.

    df, K, signal_col, min_train, refit_every, lookback, sigma_target, max_weight,
    tc_bps, sizing, allow_short, n_iter, burn_in, thin, seed, nu0, kappa0, m0: see
    signature. Returns (result, refit_log) where result is the input df with added
    columns weight, vol_hat, strat_ret, prob_<regime>, and refit_log is a dict of
    refit metadata (dates, params, f_overnight).
    """
    logger = logging.getLogger(__name__)
    missing = (_REQUIRED_COLS | {signal_col}) - set(df.columns)
    if missing:
        raise ValueError(f'DataFrame missing columns: {missing}')
    if sizing not in ('vol_target', 'kelly'):
        raise ValueError(f"sizing must be 'vol_target' or 'kelly', got {sizing!r}")

    returns = df['Return'].values
    log_rv  = df['LogRV'].values
    signal  = df[signal_col].values
    T       = len(df)
    Y       = np.column_stack([returns, log_rv, signal])

    refit_set = set(range(min_train, T, refit_every))
    n_refits  = len(refit_set)

    weights    = np.full(T, np.nan)
    vol_hats   = np.full(T, np.nan)
    filt_probs = np.full((T, K), np.nan)
    strat_ret  = np.full(T, np.nan)

    refit_log: dict = {'t': [], 'dates': [], 'mu': [], 'Sigma': [], 'P': [], 'f_overnight': []}

    mu_list: Optional[list]     = None
    Sigma_list: Optional[list]  = None
    P_cur: Optional[np.ndarray] = None
    trace_cur: Optional[dict]   = None
    f_overnight = 0.0
    alpha       = np.full(K, 1.0 / K)
    n_done      = 0
    pbar        = tqdm(total=n_refits, desc='Backtest', unit='refit')

    for t in range(T-1):

        # refit ────────────────────────────────────────────────────────────
        if t in refit_set:
            n_done += 1
            pbar.update(1)
            pbar.set_postfix({'date': str(df.index[t].date())})
            t0 = max(0, t - lookback)
            logger.info(
                    f'  [{n_done}/{n_refits}] Refit t={t} '
                    f'({df.index[t].date()})  n_train={t - t0}'
                )
            run_data = fit_hmm(
                Y[t0:t], K=K,
                n_iter=n_iter, burn_in=burn_in, thin=thin,
                seed=seed + t,  # vary per refit for independent chains
                nu0=nu0, kappa0=kappa0, sticky_kappa=sticky_kappa,
                m0=m0,
            )
            mu_list, Sigma_list, P_cur = _extract_params(run_data['trace'], K)
            trace_cur = run_data['trace']
            alpha       = _reinit_filter(Y, t, mu_list, Sigma_list, P_cur, K)
            f_overnight = _estimate_f_overnight(df.iloc[t0:t]) if estimate_f_overnight else 0.0
            logger.info(f'    f_overnight={f_overnight:.4f}')

            refit_log['t'].append(t)
            refit_log['dates'].append(df.index[t])
            refit_log['mu'].append([m.tolist() for m in mu_list])
            refit_log['Sigma'].append([S.tolist() for S in Sigma_list])
            refit_log['P'].append(P_cur.tolist())
            refit_log['f_overnight'].append(f_overnight)

        if mu_list is None:
            continue

        # daily filter update
        if use_marginal_filter:
            ll_t, mu2_list, Sig2_list = _bivariate_emission(Y[t], mu_list, Sigma_list, K)
        else:
            ll_t, mu2_list, Sig2_list = _conditional_emission(Y[t],Y[t+1][2],mu_list, Sigma_list, K)
        alpha         = _filter_step(alpha, ll_t, P_cur)
        filt_probs[t] = alpha

        # vol forecast for t+1 
        pred_probs = alpha @ P_cur

        mu_v_arr        = np.array([mu2_list[k][1] for k in range(K)])
        forecast_log_rv = float(pred_probs @ mu_v_arr)

        sv_sq_arr   = np.array([Sig2_list[k][1, 1] for k in range(K)])
        var_within  = float(pred_probs @ sv_sq_arr)
        var_between = float(pred_probs @ (mu_v_arr - forecast_log_rv) ** 2)
        jensen      = 0.5 * (var_within + var_between)

        vol_hat     = np.sqrt(np.exp(forecast_log_rv + jensen)) / np.sqrt(max(1.0 - f_overnight, 1e-6))
        vol_hats[t] = vol_hat

        # position sizing 
        if sizing == 'kelly':
            raw_w = _kelly_trace_percentile(
                trace_cur, pred_probs, K, target_percentile,
            )
            lo = -max_weight if allow_short else 0.2
            weights[t] = np.clip(raw_w, lo, max_weight)
        else:
            raw_w = _vol_target_trace_percentile(
                trace_cur, pred_probs, K, sigma_target, f_overnight, target_percentile,
            )
            weights[t] = np.clip(raw_w, 0.0, max_weight)

        dom   = int(np.argmax(alpha))
        p_str = '  '.join(f'P({_regime_label(k, K)})={alpha[k]:.2f}' for k in range(K))
        logger.info(
            f'  {df.index[t].date()}  regime={_regime_label(dom, K)}'
            f'  {p_str}  weight={weights[t]:.3f}  vol_hat={vol_hat:.3f}'
        )

    pbar.close()

    # strategy returns
    for t in range(1, T):
        if np.isnan(weights[t - 1]):
            continue
        r = weights[t - 1] * returns[t]
        if t >= 2 and not np.isnan(weights[t - 2]):
            r -= abs(weights[t - 1] - weights[t - 2]) * tc_bps * 1e-4
        strat_ret[t] = r

    result = df.copy()
    result['weight']    = weights
    result['vol_hat']   = vol_hats
    result['strat_ret'] = strat_ret
    for k in range(K):
        result[f'prob_{_regime_label(k, K)}'] = filt_probs[:, k]
    return result, refit_log

# Performance summary

def _perf(r: pd.Series) -> dict:
    ann_ret = r.mean() * 252
    ann_vol = r.std()  * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    wealth  = (1 + r).cumprod()
    return {
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'sharpe':  sharpe,
        'max_dd':  float((wealth / wealth.cummax() - 1).min()),
    }

def print_summary(result: pd.DataFrame, K: int, logger: logging.Logger) -> None:

    strat = result['strat_ret'].dropna()
    bench = result['Return'].loc[strat.index]
    s, b  = _perf(strat), _perf(bench)    
    def shout(msg):
        logger.log(100, msg)

    shout('=' * 55)
    shout('OOS PERFORMANCE SUMMARY')

    shout('=' * 55)
    shout(f'{"":20s}  {"Strategy":>10s}  {"Buy&Hold":>10s}')
    shout(f'{"Ann. Return":20s}  {s["ann_ret"]:>10.2%}  {b["ann_ret"]:>10.2%}')
    shout(f'{"Ann. Vol":20s}  {s["ann_vol"]:>10.2%}  {b["ann_vol"]:>10.2%}')
    shout(f'{"Sharpe":20s}  {s["sharpe"]:>10.3f}  {b["sharpe"]:>10.3f}')
    shout(f'{"Max Drawdown":20s}  {s["max_dd"]:>10.2%}  {b["max_dd"]:>10.2%}')

    prob_cols = [f'prob_{_regime_label(k, K)}' for k in range(K)]
    prob_df   = result[prob_cols].dropna()
    if len(prob_df) == 0:
        return

    dominant              = prob_df.values.argmax(axis=1)
    runs: dict[int, list] = {k: [] for k in range(K)}
    run_len, run_reg      = 1, dominant[0]
    for d in dominant[1:]:
        if d == run_reg:
            run_len += 1
        else:
            runs[run_reg].append(run_len)
            run_reg, run_len = d, 1
    runs[run_reg].append(run_len)

    shout('')
    shout('REGIME DURATION STATISTICS  (trading days)')
    shout(f'{"Regime":12s}  {"Frac":>7s}  {"Spells":>7s}  {"Mean":>7s}  {"Median":>7s}')
    for k in range(K):
        sl = runs[k]
        shout(
            f'{_regime_label(k, K):12s}  {sum(sl)/len(dominant):>7.1%}'
            f'  {len(sl):>7d}  {np.mean(sl):>7.1f}  {np.median(sl):>7.1f}'
        )

def main() -> None:
    parser = argparse.ArgumentParser(
        description='OOS backtest for the Trivariate HMM vol-targeting strategy. '
                    'Input CSV must have: Return, LogRV, <signal_col>, Close, Open.'
    )
    parser.add_argument('--data',           type=str,   required=True)
    parser.add_argument('--signal-col',     type=str,   default='Signal')
    parser.add_argument('--K',              type=int,   default=2)
    parser.add_argument('--min-train',      type=int,   default=756)
    parser.add_argument('--refit-every',    type=int,   default=21)
    parser.add_argument('--lookback',       type=int,   default=2000)
    parser.add_argument('--refit-iters',    type=int,   default=5000)
    parser.add_argument('--refit-burnin',   type=int,   default=1000)
    parser.add_argument('--thin',           type=int,   default=2)
    parser.add_argument('--sigma-target',   type=float, default=0.15)
    parser.add_argument('--max-weight',     type=float, default=1.5)
    parser.add_argument('--tc-bps',         type=float, default=1.0)
    parser.add_argument('--sizing',         type=str,   default='vol_target',
                        choices=['vol_target', 'kelly'])
    parser.add_argument('--target-percentile', type=float, default=10.0)
    parser.add_argument('--allow-short',        action='store_true')
    parser.add_argument('--use-marginal-filter', action='store_true')
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--nu0',            type=int,   default=6)
    parser.add_argument('--kappa0',         type=float, default=1.0)
    parser.add_argument('--sticky-kappa',   type=float, default=10.0)
    parser.add_argument('--outdir',         type=str,   default='runs')
    parser.add_argument('--verbose',        action='store_true')
    args = parser.parse_args()

    df = pd.read_csv(args.data, index_col='Date', parse_dates=True)

    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.outdir) / f'backtest_K{args.K}_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(str(out_dir / 'backtest.log'), verbose=args.verbose)
    logger.info(f'Command: {" ".join(sys.argv)}')
    logger.info(f'Output: {out_dir}')
    logger.info(
        f'K={args.K}  signal_col={args.signal_col}  '
        f'min_train={args.min_train}  refit_every={args.refit_every}  '
        f'sizing={args.sizing}'
    )
    logger.info(f'Data: {len(df)} rows  {df.index[0].date()} to {df.index[-1].date()}')

    result, refit_log = run_backtest(
        df,
        K=args.K,
        signal_col=args.signal_col,
        min_train=args.min_train,
        refit_every=args.refit_every,
        lookback=args.lookback,
        sigma_target=args.sigma_target,
        max_weight=args.max_weight,
        tc_bps=args.tc_bps,
        sizing=args.sizing,
        target_percentile=args.target_percentile,
        allow_short=args.allow_short,
        use_marginal_filter=args.use_marginal_filter,
        n_iter=args.refit_iters,
        burn_in=args.refit_burnin,
        thin=args.thin,
        seed=args.seed,
        nu0=args.nu0,
        kappa0=args.kappa0,
        sticky_kappa=args.sticky_kappa,
    )

    result.to_csv(out_dir / 'backtest.csv')
    logger.info(f'Saved results → {out_dir}/backtest.csv')

    with open(out_dir / 'refit_log.json', 'w') as f:
        json.dump({
            'dates':       [str(d.date()) for d in refit_log['dates']],
            't':           refit_log['t'],
            'mu':          refit_log['mu'],
            'Sigma':       refit_log['Sigma'],
            'P':           refit_log['P'],
            'f_overnight': refit_log['f_overnight'],
        }, f, indent=2)

    with open(out_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print_summary(result, args.K, logger)
    logger.info(f'Done. Results in {out_dir}/')

if __name__ == '__main__':
    main()
