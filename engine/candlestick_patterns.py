"""Rule-based candlestick pattern detection + recent-run/volume/delivery context,
computed purely from OHLCV bhavcopy data already on disk (no external site/API needed).

Informational only -- nothing here is wired into combined_verdict(). Every function
takes plain scalars or a per-symbol history DataFrame (sorted by DATE1 ascending,
today's row last) so it can be reused identically by the live pipeline and by
analysis/backtest_runrisk_candle.py.
"""
import numpy as np
import pandas as pd

DOJI_BODY_RATIO = 0.1        # body <= 10% of the day's range
HAMMER_WICK_RATIO = 2.0      # opposite wick >= 2x body
SMALL_WICK_RATIO = 0.3       # same-side wick <= 30% of body for a clean hammer/star
REJECTION_WICK_RATIO = 0.4   # upper_wick / range >= 40% counts as a rejection wick
BASELINE_WINDOW = 20         # sessions used for volume/delivery/recent-run baselines

# Backtested thresholds (analysis/backtest_delivery_volume.py, NSE votes>=1 picks
# since 2025-01-01, n=4626): on days where volume is already >=1.4x its 20-session
# baseline (~Q4 of the volume-ratio distribution), delivery% >50% -> 70.5% hit-rate
# vs <30% -> 58.9% hit-rate. Delivery% alone (normal-volume days) showed no gradient.
ACCUM_VOL_RATIO_MIN = 1.4
ACCUM_DELIV_LOW = 30
ACCUM_DELIV_HIGH = 50


def classify_candle(open_, high, low, close, prev_open=None, prev_close=None):
    """Single-day pattern label from today's OHLC (+ yesterday's open/close for
    engulfing patterns). Returns one label -- priority order below, most
    specific/actionable pattern wins if more than one condition matches."""
    rng = high - low
    if rng <= 0 or any(pd.isna(v) for v in (open_, high, low, close)):
        return "-"
    body = abs(close - open_)
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low

    if prev_open is not None and prev_close is not None and not pd.isna(prev_open) and not pd.isna(prev_close):
        prev_bearish = prev_close < prev_open
        prev_bullish = prev_close > prev_open
        if prev_bearish and close > open_ and open_ <= prev_close and close >= prev_open:
            return "Bullish Engulfing"
        if prev_bullish and close < open_ and open_ >= prev_close and close <= prev_open:
            return "Bearish Engulfing"

    if body <= DOJI_BODY_RATIO * rng:
        return "Doji"

    if (lower_wick >= HAMMER_WICK_RATIO * body and upper_wick <= SMALL_WICK_RATIO * body):
        return "Hammer" if close >= open_ else "Hanging Man"

    if (upper_wick >= HAMMER_WICK_RATIO * body and lower_wick <= SMALL_WICK_RATIO * body):
        return "Shooting Star"

    if upper_wick / rng >= REJECTION_WICK_RATIO:
        return "Rejection Wick"

    return "-"


def candle_for_symbol(hist, open_col="OPEN_PRICE", high_col="HIGH_PRICE",
                       low_col="LOW_PRICE", close_col="CLOSE_PRICE"):
    """hist = a single symbol's OHLC history, sorted ascending by DATE1, today's
    row last. Returns today's pattern label using yesterday's open/close too."""
    if len(hist) < 1:
        return "-"
    today = hist.iloc[-1]
    prev_open = prev_close = None
    if len(hist) >= 2:
        prev = hist.iloc[-2]
        prev_open, prev_close = prev[open_col], prev[close_col]
    return classify_candle(today[open_col], today[high_col], today[low_col], today[close_col],
                            prev_open, prev_close)


def run_risk_for_symbol(hist, close_col="CLOSE_PRICE", vol_col="TTL_TRD_QNTY", deliv_col=None,
                         window=BASELINE_WINDOW):
    """hist = a single symbol's OHLCV history, sorted ascending by DATE1, today's
    row last. Returns a dict of informational (non-gating) "already extended"
    context: recent price run, volume vs baseline, delivery% vs baseline (NSE only,
    pass deliv_col=None for BSE which has no delivery% column)."""
    out = {"RecentRun3Pct": None, "VolFlag": "-", "DelivFlag": "-"}
    if len(hist) < 4:
        return out

    closes = hist[close_col].to_numpy(dtype=float)
    out["RecentRun3Pct"] = round((closes[-1] / closes[-4] - 1) * 100, 1) if closes[-4] else None

    if vol_col in hist.columns and len(hist) > window:
        baseline = hist[vol_col].iloc[-(window + 1):-1].mean()
        today_vol = hist[vol_col].iloc[-1]
        if baseline and not pd.isna(baseline) and baseline > 0:
            ratio = today_vol / baseline
            out["VolFlag"] = (f"EXTREME {ratio:.0f}x" if ratio >= 20 else
                               f"ELEVATED {ratio:.1f}x" if ratio >= 3 else
                               "NORMAL")

    if deliv_col and deliv_col in hist.columns and len(hist) > window:
        baseline = hist[deliv_col].iloc[-(window + 1):-1].mean()
        today_deliv = hist[deliv_col].iloc[-1]
        if baseline and not pd.isna(baseline) and not pd.isna(today_deliv):
            out["DelivFlag"] = (f"COLLAPSED {today_deliv:.0f}% (base {baseline:.0f}%)"
                                 if today_deliv <= 20 and baseline >= 35 else "NORMAL")

    return out


def volume_ratio_for_symbol(hist, vol_col="TTL_TRD_QNTY", window=BASELINE_WINDOW):
    """hist = a single symbol's history, sorted ascending by DATE1, today's row last.
    Returns today's volume / trailing-window average volume, or None if not resolvable.
    Backtested (analysis/backtest_delivery_volume.py, NSE votes>=1, n=7911): today's
    volume-ratio quartile predicts hit-rate on its own (Q1-Q3 ~67-68% vs Q4 ~61%)
    regardless of delivery% -- used to rank/deprioritize picks with an extreme spike."""
    if vol_col not in hist.columns or len(hist) <= window:
        return None
    baseline = hist[vol_col].iloc[-(window + 1):-1].mean()
    today_vol = hist[vol_col].iloc[-1]
    if pd.isna(baseline) or baseline <= 0 or pd.isna(today_vol):
        return None
    return today_vol / baseline


SUSTAINED_PUMP_WINDOW = 10    # trailing sessions, NOT including today
SUSTAINED_PUMP_DELIV_MAX = 20.0


def sustained_pump_flag(hist, deliv_col="DELIV_PER", window=SUSTAINED_PUMP_WINDOW, threshold=SUSTAINED_PUMP_DELIV_MAX):
    """hist = a single symbol's history, sorted ascending by DATE1, today's row last.
    True if the `window` sessions BEFORE today averaged delivery% below `threshold` --
    catches a stock that's been pumping long enough that VolRatioToday's own rolling
    baseline is already contaminated by the pump itself, so today can misleadingly read
    "calm" (VolRatioToday near/below 1x) even while the stock is still actively pumping.
    Real case: BAJAJHIND, 02-Sep-2026 run -- VolRatioToday=0.4x/"calm" while delivery%
    had been stuck at 13-19% for 10+ straight sessions (vs a 30-53% normal baseline).
    Backtested (analysis/backtest_sustained_pump_filter.py, NSE votes>=1, n=7911):
    within the "calm today" pool (VolRatio<=1.0x) this flag separates a ~5% subset
    with meaningfully worse outcomes -- 60.1% hit-rate/-15.3% avg drawdown vs 67.9%/
    -11.8% for the rest. NSE only -- pass deliv_col=None for BSE (no delivery% data)."""
    if deliv_col is None or deliv_col not in hist.columns or len(hist) < window + 1:
        return False
    trailing = hist[deliv_col].iloc[-(window + 1):-1]
    if len(trailing) < window or trailing.isna().any():
        return False
    return bool(trailing.mean() < threshold)


def accumulation_flag_for_symbol(hist, vol_col="TTL_TRD_QNTY", deliv_col="DELIV_PER", window=BASELINE_WINDOW):
    """hist = a single symbol's history, sorted ascending by DATE1, today's row last.
    Only informative when today's volume is already elevated (>=ACCUM_VOL_RATIO_MIN x
    its 20-session baseline) -- delivery% alone (normal-volume days) showed no gradient
    in the backtest. BSE has no delivery% column -- pass deliv_col=None there."""
    if deliv_col is None or deliv_col not in hist.columns or vol_col not in hist.columns:
        return "-"
    if len(hist) <= window:
        return "-"
    vol_baseline = hist[vol_col].iloc[-(window + 1):-1].mean()
    today_vol = hist[vol_col].iloc[-1]
    today_deliv = hist[deliv_col].iloc[-1]
    if pd.isna(vol_baseline) or vol_baseline <= 0 or pd.isna(today_deliv):
        return "-"
    if today_vol / vol_baseline < ACCUM_VOL_RATIO_MIN:
        return "-"
    if today_deliv >= ACCUM_DELIV_HIGH:
        return "STRONG (real buying)"
    if today_deliv < ACCUM_DELIV_LOW:
        return "CAUTION (speculative vol)"
    return "NEUTRAL"
