"""AshFund-specific stop-loss adapter around the generic, project-agnostic
chandelier_stop.py. Owns the data access this pipeline's stop-loss needs
(per-symbol continuous daily OHLC) and hands run_pipeline.py a single
ready-to-display (stop, stop_pct) pair -- no ATR/chandelier math lives in
run_pipeline.py itself, it only imports and calls into this module.
"""
import pandas as pd

import premove_factor_analysis as nse_mod  # noqa: E402
import bse_premove_factor_analysis as bse_mod  # noqa: E402
from pattern_scan import apply_cached_split_adjustments

from chandelier_stop import chandelier_trail_stop, stop_distance_pct

ATR_PERIOD = 14


def load_continuous_hist(market):
    """Raw per-symbol daily OHLC, filtered only by listing series/group (EQ / A,B),
    NOT by the turnover/min-price liquidity filter run_pipeline.build_market()
    applies for scoring. That filter drops individual days a symbol wasn't
    liquid enough, which for thinly-traded names leaves gaps of months between
    consecutive rows -- feeding that into an ATR calc treats a multi-month
    price drift as a single day's true range and wildly inflates it."""
    df = nse_mod.load_bhav() if market == "NSE" else bse_mod.load_bhav()
    df = apply_cached_split_adjustments(df)  # else a raw split shows up as a fake ATR spike
    return df[["SYMBOL", "DATE1", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE"]]


def get_stop_loss(bhav_hist, symbol, score_date, entry_price):
    """Chandelier-exit stop (highest close since entry - 3x ATR14) for a fresh
    entry taken today, plus its % distance below entry_price.
    Returns (stop, stop_pct) rounded, or (None, None) if not enough history."""
    hist = bhav_hist[bhav_hist["SYMBOL"] == symbol].sort_values("DATE1")
    if hist.empty:
        return None, None
    hist = hist.set_index("DATE1")[["HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE"]].rename(
        columns={"HIGH_PRICE": "High", "LOW_PRICE": "Low", "CLOSE_PRICE": "Close"})
    stop = chandelier_trail_stop(hist, entry_date=pd.Timestamp(score_date), atr_period=ATR_PERIOD)
    if stop is None:
        return None, None
    return round(stop, 2), round(stop_distance_pct(entry_price, stop), 2)
