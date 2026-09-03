"""
Chandelier Exit trailing stop - a standalone, dependency-light module (pandas only)
so it can be dropped into any trading pipeline (paper trading, live trading, backtesting).
Vendored as-is from c:\\Mine\\TradingTool\\chandelier_stop.py (2026-08-24) -- no
changes made, so it can stay in sync if the source file is updated.

Chandelier Exit = highest close (or high) since entry, minus a multiple of ATR.
The stop only ever moves UP (for a long position), never down, and never resets
just because price dipped.

Two ways to use it:
1. `chandelier_trail_stop(df, entry_date, mult, atr_period)` - one-shot batch calc
   over a full OHLC history DataFrame (index must be datetime-like, or pass
   entry_date=None to use the whole df).
2. `ChandelierTrailingStop` - stateful class for bar-by-bar / streaming use, where
   you don't have (or don't want to recompute) the full history each time.
"""
import pandas as pd

CHANDELIER_ATR_MULT = 3.0  # standard chandelier-exit distance below the highest close since entry


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder-style ATR (EMA of true range) from a DataFrame with High/Low/Close columns."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def chandelier_trail_stop(df: pd.DataFrame, entry_date=None, mult: float = CHANDELIER_ATR_MULT, atr_period: int = 14):
    """One-shot chandelier stop from a full OHLC history DataFrame.
    Returns the stop level (float), or None if there isn't enough data yet.
    `entry_date` restricts the "highest close" lookback to bars on/after entry;
    pass None to use the whole DataFrame (e.g. for a pipeline with no fixed entry)."""
    if df is None or len(df) < atr_period + 1:
        return None
    atr_series = atr(df, atr_period)
    since_entry = df if entry_date is None else df[df.index >= entry_date]
    if since_entry.empty:
        since_entry = df.tail(1)
    highest_close = since_entry["Close"].max()
    latest_atr = atr_series.iloc[-1]
    if pd.isna(highest_close) or pd.isna(latest_atr):
        return None
    return float(highest_close - mult * latest_atr)


def stop_distance_pct(price: float, stop: float) -> float:
    """% distance from current price down to the stop level.
    Positive = room left before the stop is hit; negative = price is already below it."""
    return (price - stop) / price * 100


class ChandelierTrailingStop:
    """Stateful chandelier trailing stop for bar-by-bar / streaming pipelines
    (live trading loops, backtesters walking day-by-day) where recomputing the
    full history on every bar is wasteful or unavailable.

    Usage:
        trail = ChandelierTrailingStop(mult=3.0, atr_period=14, initial_stop=entry - 1.5*atr0)
        for bar in bars:
            stop = trail.update(bar.high, bar.low, bar.close)
            if bar.close <= stop:
                # exit
    """

    def __init__(self, mult: float = CHANDELIER_ATR_MULT, atr_period: int = 14, initial_stop: float = None):
        self.mult = mult
        self.atr_period = atr_period
        self.stop = initial_stop
        self._highest_close = None
        self._atr = None
        self._prev_close = None

    def update(self, high: float, low: float, close: float) -> float:
        """Feed one new bar. Returns the current stop level (never lower than before)."""
        prev_close = self._prev_close if self._prev_close is not None else close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        alpha = 1 / self.atr_period
        self._atr = tr if self._atr is None else alpha * tr + (1 - alpha) * self._atr
        self._prev_close = close
        self._highest_close = close if self._highest_close is None else max(self._highest_close, close)

        candidate = self._highest_close - self.mult * self._atr
        if self.stop is None or candidate > self.stop:
            self.stop = candidate
        return self.stop
