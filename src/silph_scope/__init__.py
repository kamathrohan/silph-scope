"""
silph_scope — Trivariate HMM vol-targeting strategy.

Public API
----------
Model:
    fit_hmm  — fit the HMM via Gibbs + FFBS
    save_results       — persist trace and metadata to disk

Backtest:
    run_backtest   — expanding-window OOS backtest
    print_summary  — log performance and regime duration stats

Data:
    load_with_rv_lag  — lagged log-RV signal (any asset)
    load_with_vix     — VIX-derived signal (SPY + VIX)
"""

__all__ = [
    'fit_hmm',
    'save_results',
    'run_backtest',
    'print_summary',
    'load_with_rv_lag',
    'load_with_vix',
]

_LAZY = {
    'fit_hmm': '.runner',
    'save_results':       '.runner',
    'run_backtest':       '.backtest',
    'print_summary':      '.backtest',
    'load_with_rv_lag':   '.data',
    'load_with_vix':      '.data',
}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module 'silph_scope' has no attribute {name!r}")
    import importlib
    mod = importlib.import_module(_LAZY[name], package=__name__)
    obj = getattr(mod, name)
    globals()[name] = obj   # cache so subsequent access is direct
    return obj
