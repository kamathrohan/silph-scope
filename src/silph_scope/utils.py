"""
Core utilities for the Trivariate HMM vol-targeting strategy.

  - Forward-Filter Backward-Sample (FFBS)
  - Transition matrix sampling
  - MVN log-density
  - HMM emission matrix
  - Normal-Inverse-Wishart conjugate sampler
  - Regime initialisation and label-switching prevention
  - Shared prior / label helpers
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numba import njit, prange
from scipy.stats import invwishart

@njit
def _categorical(probs):
    u      = np.random.rand()
    running_sum = 0.0
    for i in range(len(probs)):
        running_sum += probs[i]
        if u < running_sum:
            return i
    return len(probs) - 1

@njit
def forward_filter(ll, P):
    T, K  = ll.shape[0], ll.shape[1]
    alpha = np.zeros((T, K))
    log_a = ll[0].copy()
    log_a -= log_a.max()
    alpha[0] = np.exp(log_a)
    alpha[0] /= alpha[0].sum()
    for t in range(1, T):
        alpha_pred  = alpha[t - 1] @ P
        log_joint   = np.log(np.maximum(alpha_pred, 1e-300)) + ll[t]
        log_joint  -= log_joint.max()
        alpha[t]    = np.exp(log_joint)
        alpha[t]   /= alpha[t].sum()
    return alpha

@njit
def backward_sample(alpha, P):
    np.random.seed(np.random.randint(0, 2**31))
    T       = alpha.shape[0]
    regimes = np.empty(T, dtype=np.int32)
    regimes[T - 1] = _categorical(alpha[T - 1])
    for t in range(T - 2, -1, -1):
        w = alpha[t] * P[:, regimes[t + 1]]
        w /= w.sum()
        regimes[t] = _categorical(w)
    return regimes

@njit
def count_transitions(regimes, K):
    n = np.zeros((K, K), dtype=np.int32)
    for t in range(len(regimes) - 1):
        n[regimes[t], regimes[t + 1]] += 1
    return n

def sample_transition_matrix(
    regimes: np.ndarray,
    prior_alpha: float | np.ndarray = 1.0,
    K: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample P | regimes from the Dirichlet posterior.

    Parameters
    ----------
    regimes     : (T,) integer regime sequence
    prior_alpha : scalar or (K, K) array of Dirichlet prior counts
    K           : number of regimes; inferred from regimes if None

    Returns
    -------
    P : (K, K) sampled transition matrix
    n : (K, K) transition count matrix
    """
    if K is None:
        K = int(regimes.max()) + 1
    n     = count_transitions(regimes.astype(np.int32), K)
    prior = np.asarray(prior_alpha)
    P_new = np.zeros((K, K))
    for k in range(K):
        alpha_k = prior[k] if prior.ndim == 2 else np.full(K, float(prior))
        P_new[k] = np.random.dirichlet(n[k] + alpha_k)
    return P_new, n

@njit
def ess(x: np.ndarray) -> int:
    """Effective sample size via AR(1) approximation: n*(1-rho)/(1+rho)."""
    x = np.asarray(x)
    if len(x) < 3:
        return len(x)
    rho = np.corrcoef(x[:-1], x[1:])[0, 1]
    if np.isnan(rho) or rho >= 1.0:
        return 1
    return max(int(len(x) * (1.0 - rho) / (1.0 + rho)), 1)

def _regime_label(k: int, K: int) -> str:
    """
    The names are descriptive of the vol ranking only (enforced by
    utils._post_hoc_relabel), not a narrative.
    Labels are weather-themed and ordinal:
    clear_skies < drizzle < hail < sandstorm < thunderstorm.
    """

    if K == 2: return ['clear_skies', 'thunderstorm'][k]
    if K == 3: return ['clear_skies', 'sandstorm', 'thunderstorm'][k]
    if K == 4: return ['clear_skies', 'drizzle', 'sandstorm', 'thunderstorm'][k]
    if K == 5: return ['clear_skies', 'drizzle', 'hail', 'sandstorm', 'thunderstorm'][k]
    return f'type{k}'

def _build_sticky_P(K: int, p_stay: float = 0.95) -> np.ndarray:
    """K×K initial transition matrix: p_stay on diagonal, rest split equally."""
    P = np.full((K, K), (1.0 - p_stay) / max(K - 1, 1))
    np.fill_diagonal(P, p_stay)
    return P

@njit(parallel=True)
def mvn_logpdf_batch(Y, mu, Sigma_inv, log_det):
    T, d = Y.shape
    log2pi = 1.8378770664093453  # log(2*pi)
    out = np.empty(T)
    for t in prange(T):
        mahal = 0.0
        for i in range(d):
            s = 0.0
            for j in range(d):
                s += Sigma_inv[i, j] * (Y[t, j] - mu[j])
            mahal += (Y[t, i] - mu[i]) * s
        out[t] = -0.5 * (d * log2pi + log_det + mahal)
    return out

def hmm_log_emission(
    Y: np.ndarray,
    mu_list: list[np.ndarray],
    Sigma_list: list[np.ndarray],
    Sigma_inv_list: list[np.ndarray] | None = None,
    log_det_list: list[float] | None = None,
) -> np.ndarray:
    if Sigma_inv_list is None:
        Sigma_inv_list = [np.linalg.inv(S) for S in Sigma_list]
    if log_det_list is None:
        log_det_list = [float(np.linalg.slogdet(S)[1]) for S in Sigma_list]
    K  = len(mu_list)
    ll = np.empty((Y.shape[0], K))
    for k in range(K):
        ll[:, k] = mvn_logpdf_batch(Y, mu_list[k], Sigma_inv_list[k], log_det_list[k])
    return ll

def sample_NIW_s(
    Y_s: np.ndarray,
    m0: np.ndarray,
    kappa0: float,
    Psi0: np.ndarray,
    nu0: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw (mu_s, Sigma_s) from the NIW posterior.

    Prior:  Sigma_s ~ IW(Psi0, nu0),  mu_s | Sigma_s ~ N(m0, Sigma_s/kappa0)

    Parameters
    ----------
    Y_s    : (n_s, d) regime observations
    m0     : (d,)     prior mean
    kappa0 : float    prior strength on mean
    Psi0   : (d, d)   IW prior scale
    nu0    : int      IW degrees of freedom (>= d+1)

    Returns
    -------
    mu_s    : (d,)
    Sigma_s : (d, d)
    """
    n_s     = Y_s.shape[0]
    ybar    = Y_s.mean(axis=0)
    kappa_n = kappa0 + n_s
    m_n     = (kappa0 * m0 + n_s * ybar) / kappa_n
    nu_n    = nu0 + n_s
    S_s     = (Y_s - ybar).T @ (Y_s - ybar)
    Psi_n   = Psi0 + S_s + (kappa0 * n_s / kappa_n) * np.outer(ybar - m0, ybar - m0)
    Sigma_s = invwishart.rvs(df=nu_n, scale=Psi_n)
    mu_s    = np.random.multivariate_normal(m_n, Sigma_s / kappa_n)
    return mu_s, Sigma_s

def init_regimes(
    Y: np.ndarray,
    K: int = 2,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Initialise regime assignments by K-quantile split on log-RV (column 1).

    Returns
    -------
    regimes    : (T,) int array in {0, ..., K-1}
    mu_list    : list of K (d,) initial means
    Sigma_list : list of K (d, d) initial covariances
    """
    log_rv    = Y[:, 1]
    quantiles = np.quantile(log_rv, np.linspace(0.0, 1.0, K + 1))
    regimes   = np.clip(np.digitize(log_rv, quantiles[1:-1]), 0, K - 1).astype(int)

    d, data_cov = Y.shape[1], np.cov(Y.T)
    mu_list, Sigma_list = [], []
    for k in range(K):
        mask = regimes == k
        Y_k  = Y[mask]
        if mask.sum() > d + 1:
            mu_list.append(Y_k.mean(axis=0))
            Sigma_list.append(np.cov(Y_k.T) + 1e-6 * np.eye(d))
        else:
            mu_list.append(Y.mean(axis=0))
            Sigma_list.append(data_cov + 1e-6 * np.eye(d))

    return regimes, mu_list, Sigma_list

def _post_hoc_relabel(trace_full: dict[str, list], K: int) -> None:
    """
    Post-hoc relabeling: Sorts regime labels for each MCMC iteration by mu_v.
    This ensures the trace is immune to mid-run label switching and correctly
    permutes the transition matrix P to match the sorted regimes.
    """
    n_samples = len(trace_full['mu_v0'])
    for i in range(n_samples):
        mu_vs = [trace_full[f'mu_v{k}'][i] for k in range(K)]
        perm = np.argsort(mu_vs)
        
        if np.array_equal(perm, np.arange(K)):
            continue
            
        # Swap emission parameters
        for pn in _PNAMES:
            vals = [trace_full[f'{pn}{k}'][i] for k in range(K)]
            for k in range(K):
                trace_full[f'{pn}{k}'][i] = vals[perm[k]]
                
        # Reconstruct full P matrix, permute it, and save off-diagonals back
        old_P = np.zeros((K, K))
        for r in range(K):
            for c in range(K):
                if r != c:
                    old_P[r, c] = trace_full[f'P{r}{c}'][i]
                else:
                    old_P[r, c] = 1.0 - sum(trace_full[f'P{r}{cc}'][i] for cc in range(K) if cc != r)
                    
        new_P = old_P[np.ix_(perm, perm)]
        
        for r in range(K):
            for c in range(K):
                if r != c:
                    trace_full[f'P{r}{c}'][i] = new_P[r, c]


def decompose_Sigma_3d(
    Sigma: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """Extract (sigma_r, sigma_v, sigma_s, rho_rv, rho_rs, rho_vs) from a (3,3) covariance."""
    sr = float(np.sqrt(Sigma[0, 0]))
    sv = float(np.sqrt(Sigma[1, 1]))
    ss = float(np.sqrt(Sigma[2, 2]))
    return sr, sv, ss, float(Sigma[0,1]/(sr*sv)), float(Sigma[0,2]/(sr*ss)), float(Sigma[1,2]/(sv*ss))
