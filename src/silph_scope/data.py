"""
Data loaders — asset-class-specific preprocessing that produces a DataFrame
in the standard contract expected by silph_scope.backtest.run_backtest:

    Return   — log daily return
    LogRV    — log annualised realised variance
    Signal   — conditioning signal (third HMM observation)
    Close    — closing price
    Open     — opening price

Add new loaders here as new asset classes are onboarded.  The model code in
runner.py and backtest.py has no dependency on any loader.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

def _build_base(ohlcv_file: str) -> pd.DataFrame:
    """Read an OHLCV CSV, filter weekends, compute Return and LogRV."""
    df = pd.read_csv(ohlcv_file, index_col='Date', parse_dates=True)
    df = df[df.index.dayofweek < 5]
    df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['AnnRV']  = df['RealizedVar'] * 252
    df['LogRV']  = np.log(df['AnnRV'])
    return df

def _log_summary(df: pd.DataFrame, label: str, logger: logging.Logger) -> None:
    logger.info(f'Signal={label}  {len(df)} trading days  '
                f'{df.index[0].date()} to {df.index[-1].date()}')
    logger.info(f'Return:  mean={df["Return"].mean():.5f}  std={df["Return"].std():.5f}')
    logger.info(f'LogRV:   mean={df["LogRV"].mean():.3f}   std={df["LogRV"].std():.3f}')
    logger.info(f'Signal:  mean={df["Signal"].mean():.3f}   std={df["Signal"].std():.3f}')

def load_with_rv_lag(
    ohlcv_file: str,
    lag: int = 1,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Use lagged log-RV as the conditioning signal.

    Works for any asset with an OHLCV + RealizedVar CSV — no auxiliary file needed.

    Parameters
    ----------
    ohlcv_file : str
        CSV with columns: Date, Open, Close, RealizedVar.
    lag : int
        Number of days to lag the log-RV signal (default 1).

    Returns
    -------
    pd.DataFrame with columns: Return, LogRV, Signal, Close, Open
    """
    if lag < 1:
        raise ValueError(f'lag must be >= 1, got {lag}')
    logger = logger or logging.getLogger(__name__)

    base = _build_base(ohlcv_file)
    df   = base[['Return', 'LogRV', 'Close', 'Open']].copy()
    df['Signal'] = df['LogRV'].shift(lag)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    _log_summary(df, f'rv_lag_{lag}', logger)
    return df



def load_with_vix(
    spy_file: str,
    vix_file: str,
    vix_mode: Literal['open', 'close_lag', 'vrp_lag', 'open_vrp'] = 'close_lag',
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Use a VIX-derived quantity as the conditioning signal.

    Parameters
    ----------
    spy_file : str
        Path to SPY OHLCV CSV with columns: Date, Open, Close, RealizedVar.
    vix_file : str
        Path to VIX OHLCV CSV with columns: Date, Open, Close.
    vix_mode : str
        How to construct the Signal column:
            'open'       — today's VIX open (log annualised variance)
            'close_lag'  — yesterday's VIX close
            'vrp_lag'    — yesterday's log variance risk premium (log-VIX - log-RV)
            'open_vrp'   — today's VIX open minus yesterday's log-RV

    Returns
    -------
    pd.DataFrame with columns: Return, LogRV, Signal, Close, Open
    """
    logger = logger or logging.getLogger(__name__)

    base = _build_base(spy_file)
    spy  = base[['Return', 'LogRV', 'Close', 'Open']]

    vix = pd.read_csv(vix_file, index_col='Date', parse_dates=True)
    vix = vix[vix.index.dayofweek < 5]
    vix['LogVIX_Close'] = np.log((vix['Close'] / 100) ** 2)
    vix['LogVIX_Open']  = np.log((vix['Open']  / 100) ** 2)

    df = spy.join(vix[['LogVIX_Close', 'LogVIX_Open']], how='inner')

    if vix_mode == 'open':
        df['Signal'] = df['LogVIX_Open']
    elif vix_mode == 'close_lag':
        df['Signal'] = df['LogVIX_Close'].shift(1)
    elif vix_mode == 'vrp_lag':
        df['Signal'] = (df['LogVIX_Close'] - df['LogRV']).shift(1)
    elif vix_mode == 'open_vrp':
        df['Signal'] = df['LogVIX_Open'] - df['LogRV'].shift(1)
    else:
        raise ValueError(f'Unknown vix_mode: {vix_mode!r}')

    df = df.drop(columns=['LogVIX_Close', 'LogVIX_Open'])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    _log_summary(df, vix_mode, logger)
    return df
