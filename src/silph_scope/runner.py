#!/usr/bin/env python3
"""
Trivariate HMM — Gibbs + FFBS sampler.

Observation model:
    y_t = [r_t,  log(AnnRV_t),  signal_t]  ~  MVN(mu_s, Sigma_s)  given S_t = s

The third column can be any conditioning signal (log-VIX, lagged RV, momentum,
credit spread, …). It sharpens regime identification during fitting but is not
used in the live forward filter or vol forecast.

Conjugate prior: Normal-Inverse-Wishart
    Sigma_s         ~ IW(Psi0, nu0)
    mu_s | Sigma_s  ~ N(m0, Sigma_s / kappa0)

Gibbs steps:
    A. FFBS      — sample state sequence S_{1:T}
    B. Dirichlet — sample transition matrix P
    C. NIW       — sample (mu_s, Sigma_s) jointly per regime

Regimes are identified by sorting on mu_v (mean log-RV); regime 0 = lowest vol.

Usage:
    python3 -m silph_scope.runner --data data/spy-signals.csv --K 2
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

from .utils import (
    hmm_log_emission,
    init_regimes,
    sample_NIW_s,
    decompose_Sigma_3d,
    forward_filter,
    backward_sample,
    sample_transition_matrix,
    ess,
    _regime_label,
    _post_hoc_relabel
)

_PNAMES = ['mu_r', 'mu_v', 'mu_sig',
           'sigma_r', 'sigma_v', 'sigma_sig',
           'rho_rv', 'rho_rsig', 'rho_vsig']


def _precompute_sigma(Sigma_list):
    return (
        [np.linalg.inv(S) for S in Sigma_list],
        [float(np.linalg.slogdet(S)[1]) for S in Sigma_list],
    )


def _build_prior(Y, K, nu0, m0, sticky_kappa: float = 10.0):
    d = Y.shape[1]
    m0_arr = np.asarray(m0, dtype=float) if m0 is not None else Y.mean(axis=0)
    Psi0 = max(nu0 - d - 1, 1) * np.cov(Y.T)
    dirichlet_prior = np.ones((K, K))
    np.fill_diagonal(dirichlet_prior, sticky_kappa)
    return m0_arr, Psi0, dirichlet_prior


def _record_sample(trace_full, mu_list, Sigma_list, P, K):
    for i in range(K):
        for j in range(K):
            if i != j:
                trace_full[f'P{i}{j}'].append(P[i, j])
    for k in range(K):
        trace_full[f'mu_r{k}'].append(mu_list[k][0])
        trace_full[f'mu_v{k}'].append(mu_list[k][1])
        trace_full[f'mu_sig{k}'].append(mu_list[k][2])
        sr, sv, ss, rho_rv, rho_rs, rho_vs = decompose_Sigma_3d(Sigma_list[k])
        trace_full[f'sigma_r{k}'].append(sr)
        trace_full[f'sigma_v{k}'].append(sv)
        trace_full[f'sigma_sig{k}'].append(ss)
        trace_full[f'rho_rv{k}'].append(rho_rv)
        trace_full[f'rho_rsig{k}'].append(rho_rs)
        trace_full[f'rho_vsig{k}'].append(rho_vs)



                    
def _log_posterior(logger, trace, regimes, K, P_keys):
    logger.info('=' * 70)
    logger.info('POSTERIOR ESTIMATES')
    logger.info('=' * 70)
    T_total = len(regimes)
    for k in range(K):
        n_k = int((regimes == k).sum())
        logger.info(f'  Regime {k} ({_regime_label(k, K)}):  {n_k} days  ({n_k/T_total:.1%})')

    for k in range(K):
        logger.info(f'Regime {k} ({_regime_label(k, K)}):')
        for pn in _PNAMES:
            key = f'{pn}{k}'
            s = trace[key]
            q025, q975 = np.percentile(s, [2.5, 97.5])
            logger.info(
                f'  {key:>12s}:  mean={np.mean(s):>+10.5f}  '
                f'std={np.std(s):>8.5f}  95% CI=[{q025:+.5f}, {q975:+.5f}]'
            )

    logger.info('Transition matrix (posterior means):')
    for i in range(K):
        row = []
        for j in range(K):
            if i != j:
                row.append(f'P{i}{j}={trace[f"P{i}{j}"].mean():.4f}')
            else:
                p_ii = 1.0 - sum(trace[f'P{i}{jj}'].mean() for jj in range(K) if jj != i)
                row.append(f'P{i}{i}={p_ii:.4f}')
        logger.info(f'  {"  ".join(row)}')

    logger.info('Effective Sample Sizes:')
    for key in P_keys + [f'{pn}{k}' for k in range(K) for pn in _PNAMES]:
        logger.info(f'  {key:>14s}: {ess(trace[key]):>5d}')


def fit_hmm(
    Y: np.ndarray,
    K: int = 2,
    n_iter: int = 5000,
    burn_in: int = 1000,
    thin: int = 2,
    seed: int = 42,
    nu0: int = 6,
    kappa0: float = 1.0,
    sticky_kappa: float = 10.0,
    m0: Optional[list] = None,
) -> dict:
    """Gibbs + FFBS sampler for the K-regime Trivariate HMM.
    Returns dict with keys: trace_full, trace, mu_list, Sigma_list, P, regimes, K.
    """
    logger = logging.getLogger(__name__)
    np.random.seed(seed)

    T, d = Y.shape
    if d != 3:
        raise ValueError(f'Y must have 3 columns [returns, log(AnnRV), signal], got {d}')

    m0_arr, Psi0, dirichlet_prior = _build_prior(Y, K, nu0, m0, sticky_kappa)
    regimes, mu_list, Sigma_list  = init_regimes(Y, K=K)
    P, _                          = sample_transition_matrix(regimes, prior_alpha=dirichlet_prior, K=K)
    Sigma_inv_list, log_det_list  = _precompute_sigma(Sigma_list)

    P_keys: list[str] = [f'P{i}{j}' for i in range(K) for j in range(K) if i != j]
    trace_full: dict[str, list] = {pk: [] for pk in P_keys}
    for k in range(K):
        for pn in _PNAMES:
            trace_full[f'{pn}{k}'] = []

    logger.info('=' * 70)
    logger.info(f'Trivariate HMM  K={K}  {n_iter} iters  burn-in={burn_in}  thin={thin}')
    logger.info(f'Prior: nu0={nu0}  kappa0={kappa0}  '
                f'm0=[{m0_arr[0]:+.4f}, {m0_arr[1]:+.4f}, {m0_arr[2]:+.4f}]')
    for k in range(K):
        mu = mu_list[k]
        sr, sv, ss, rho_rv, rho_rs, rho_vs = decompose_Sigma_3d(Sigma_list[k])
        logger.info(
            f'  Regime {k} ({_regime_label(k, K)}) init:  '
            f'mu=[{mu[0]:+.4f}, {mu[1]:+.4f}, {mu[2]:+.4f}]  '
            f'sigma=({sr:.4f}, {sv:.4f}, {ss:.4f})  '
            f'rho=({rho_rv:+.3f}, {rho_rs:+.3f}, {rho_vs:+.3f})'
        )

    log_every = max(n_iter // 10, 1)

    for it in range(1, n_iter + 1):
        ll      = hmm_log_emission(Y, mu_list, Sigma_list, Sigma_inv_list, log_det_list)
        alpha   = forward_filter(ll, P)
        regimes = backward_sample(alpha, P)

        P, _ = sample_transition_matrix(regimes, prior_alpha=dirichlet_prior, K=K)

        for k in range(K):
            mask = regimes == k
            if int(mask.sum()) < d + 2:
                continue
            mu_list[k], Sigma_list[k] = sample_NIW_s(Y[mask], m0_arr, kappa0, Psi0, nu0)

        Sigma_inv_list, log_det_list = _precompute_sigma(Sigma_list)

        _record_sample(trace_full, mu_list, Sigma_list, P, K)

        if it % log_every == 0:
            phase  = 'BURN-IN' if it <= burn_in else 'SAMPLING'
            counts = '  '.join(f'n_{_regime_label(k, K)}={(regimes==k).sum()}' for k in range(K))
            mus    = ', '.join(
                f'[{mu_list[k][0]:+.3f},{mu_list[k][1]:+.3f},{mu_list[k][2]:+.3f}]'
                for k in range(K)
            )
            logger.info(f'[{phase}] iter {it:>5d}:  {counts}  mu=({mus})')
    # Apply post-hoc relabeling before extracting the final trace
    _post_hoc_relabel(trace_full, K)
    kept  = np.arange(burn_in, n_iter, thin)
    trace = {k: np.asarray(trace_full[k])[kept] for k in trace_full}

    _log_posterior(logger, trace, regimes, K, P_keys)

    return {
        'trace_full': trace_full,
        'trace':      trace,
        'mu_list':    mu_list,
        'Sigma_list': Sigma_list,
        'P':          P,
        'regimes':    regimes,
        'K':          K,
    }


def save_results(run_dir: Path, run_data: dict, config: dict) -> None:
    """Save trace arrays (.npz) and metadata (.json) to run_dir."""
    K      = run_data['K']
    P_keys = [f'P{i}{j}' for i in range(K) for j in range(K) if i != j]

    arrays = {pk: np.asarray(run_data['trace_full'][pk], dtype=np.float64) for pk in P_keys}
    for k in range(K):
        for pn in _PNAMES:
            key         = f'{pn}{k}'
            arrays[key] = np.asarray(run_data['trace_full'][key], dtype=np.float64)

    np.savez_compressed(str(run_dir / 'triv_hmm_mcmc.npz'), **arrays)

    meta = {
        'config':      config,
        'K':           K,
        'final_mu':    [mu.tolist() for mu in run_data['mu_list']],
        'final_Sigma': [S.tolist()  for S  in run_data['Sigma_list']],
        'final_P':     run_data['P'].tolist(),
    }
    with open(run_dir / 'triv_hmm_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)


def setup_logging(log_file: str, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger('silph_scope')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(100)
    sh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(sh)

    if verbose:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)

    return logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Fit Trivariate HMM via Gibbs+FFBS. '
                    'Input CSV must have columns: Return, LogRV, <signal_col>.'
    )
    parser.add_argument('--data',         type=str,   required=True)
    parser.add_argument('--signal-col',   type=str,   default='Signal')
    parser.add_argument('--K',            type=int,   default=2)
    parser.add_argument('--iters',        type=int,   default=5000)
    parser.add_argument('--burnin',       type=int,   default=1000)
    parser.add_argument('--thin',         type=int,   default=2)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--nu0',          type=int,   default=6)
    parser.add_argument('--kappa0',       type=float, default=1.0)
    parser.add_argument('--sticky-kappa', type=float, default=10.0)
    parser.add_argument('--max-lookback', type=int,   default=756)
    parser.add_argument('--outdir',       type=str,   default='runs')
    parser.add_argument('--verbose',      action='store_true')
    args = parser.parse_args()

    import pandas as pd
    df = pd.read_csv(args.data, index_col='Date', parse_dates=True)
    missing = {'Return', 'LogRV', args.signal_col} - set(df.columns)
    if missing:
        raise ValueError(f'Input CSV missing columns: {missing}')
    if df.shape[0] > args.max_lookback:
        df = df.iloc[-args.max_lookback:]

    Y = df[['Return', 'LogRV', args.signal_col]].values

    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(args.outdir) / f'triv_hmm_K{args.K}_{ts}'
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(str(run_dir / 'triv_hmm.log'), verbose=args.verbose)
    logger.info(f'Command: {" ".join(sys.argv)}')
    logger.info(f'Output: {run_dir}')
    logger.info(f'Data: {len(df)} rows  signal_col={args.signal_col}')

    run_data = fit_hmm(
        Y, K=args.K, n_iter=args.iters, burn_in=args.burnin, thin=args.thin,
        seed=args.seed, nu0=args.nu0, kappa0=args.kappa0,
        sticky_kappa=args.sticky_kappa,
    )
    save_results(run_dir, run_data, vars(args))
    logger.info(f'Saved to {run_dir}/')
    logger.info('COMPLETE')


if __name__ == '__main__':
    main()
