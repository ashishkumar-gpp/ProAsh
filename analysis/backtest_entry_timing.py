"""Does WHEN you enter (not which candle/volume/delivery flag is showing) actually
control your outcome on BUY-verdict picks? The candle/RunRisk backtest already showed
every single BUY pick dips ~13% on average before the (67% of the time) eventual move
to +12% -- this script quantifies whether waiting for part of that dip before buying
actually improves results, and what it costs you in missed trades.

Simulates, for every real BUY pick since 2025-01-01:
  - BASELINE: buy at signal-day close, target = entry*1.12, must hit within sessions [2,45].
  - WAIT-X%: wait up to WAIT_WINDOW sessions for a >=X% pullback from signal-day close;
    if it dips that much, buy there instead; if it never dips that much, buy at the
    WAIT_WINDOW-day close as a fallback (so no trade is silently dropped). Target is
    STILL the original entry*1.12 price level (i.e. does the stock still reach the level
    the pipeline said was achievable), re-checked within the remaining session budget.

Reuses build_ensemble() (votes) + build_market() (exact split-adjusted OHLCV the live
tool sees) -- same data sources as backtest_runrisk_candle.py. Informational analysis
only; does not change combined_verdict() or any live selection logic.

Usage: python analysis/backtest_entry_timing.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "engine"))
sys.path.insert(0, r"C:\Mine\Compare")

from proash_pipeline import build_market  # noqa: E402
from ensemble_model import build_ensemble, BACKTEST_START  # noqa: E402

DAY_MIN, DAY_MAX = 2, 45      # same objective window as the live pipeline
WAIT_WINDOW = 5                # sessions willing to wait for a pullback before giving up
PULLBACK_LEVELS = [0.0, 0.03, 0.05, 0.08]   # 0% = baseline (buy signal-day close)
TARGET_PCT = 0.12


def symbol_close_paths(market):
    """dict[(SYMBOL, Market)] -> (DATE1 array, CLOSE_PRICE array), both sorted ascending."""
    hist = build_market(market).sort_values(["SYMBOL", "DATE1"])
    paths = {}
    for sym, g in hist.groupby("SYMBOL", sort=False):
        paths[(sym, market)] = (g["DATE1"].to_numpy(), g["CLOSE_PRICE"].to_numpy(dtype=float))
    return paths


def simulate_pick(dates, closes, entry_idx, pullback_pct):
    """entry_idx = index of the signal day within this symbol's close array.
    Returns dict with entry_price, entry_offset (sessions waited), hit, hit_day,
    or None if there isn't enough forward history to evaluate at all."""
    n = len(closes)
    if entry_idx + DAY_MAX >= n:  # not enough forward history to resolve at day 45
        return None
    signal_close = closes[entry_idx]

    dip_achieved = pullback_pct <= 0
    if pullback_pct <= 0:
        entry_off, entry_price = 0, signal_close
    else:
        entry_off, entry_price = None, None
        for off in range(1, WAIT_WINDOW + 1):
            if closes[entry_idx + off] <= signal_close * (1 - pullback_pct):
                entry_off, entry_price, dip_achieved = off, closes[entry_idx + off], True
                break
        if entry_off is None:  # dip never came -- fallback: buy at WAIT_WINDOW close anyway
            entry_off, entry_price = WAIT_WINDOW, closes[entry_idx + WAIT_WINDOW]

    target_price = signal_close * (1 + TARGET_PCT)  # same absolute bar the pipeline promised
    lo = entry_idx + entry_off + 1          # first session strictly after actual entry
    hi = entry_idx + DAY_MAX                # same absolute deadline the pipeline promised (day 45 from signal)
    lo, hi = min(lo, n - 1), min(hi, n - 1)
    window = closes[lo:hi + 1]
    hit_idx = np.argmax(window >= target_price) if (window >= target_price).any() else None

    return {
        "entry_offset": entry_off,
        "entry_price": entry_price,
        "dip_achieved": dip_achieved,
        "discount_vs_signal_pct": (entry_price / signal_close - 1) * 100,
        "hit": hit_idx is not None,
        "effective_gain_if_hit_pct": ((target_price / entry_price - 1) * 100) if hit_idx is not None else None,
    }


def main():
    print("Building ensemble (votes) + NSE/BSE close-price paths...")
    merged = build_ensemble()
    merged = merged[merged["DATE1"] >= pd.Timestamp(BACKTEST_START)].copy()
    picks = merged[merged["votes"] >= 1][["key_symbol", "key_market", "DATE1"]].drop_duplicates()
    print(f"votes>=1 picks since {BACKTEST_START}: n={len(picks)}")

    paths = {}
    for m in ("NSE", "BSE"):
        paths.update(symbol_close_paths(m))

    print(f"\n{'Pullback req.':<15}{'n':>6}{'hit_rate%':>11}{'avg_days_waited':>17}"
          f"{'avg_discount%':>15}{'avg_eff_gain%_if_hit':>22}{'missed(no dip)%':>17}")
    for pb in PULLBACK_LEVELS:
        results = []
        for row in picks.itertuples(index=False):
            key = (row.key_symbol, row.key_market)
            if key not in paths:
                continue
            dates, closes = paths[key]
            idx = np.searchsorted(dates, np.datetime64(row.DATE1))
            if idx >= len(dates) or dates[idx] != np.datetime64(row.DATE1):
                continue
            r = simulate_pick(dates, closes, idx, pb)
            if r is None:
                continue
            results.append(r)

        if not results:
            continue
        df = pd.DataFrame(results)
        hits = df[df["hit"]]
        missed_pct = (~df["dip_achieved"]).mean() * 100
        label = "0% (baseline)" if pb == 0 else f">= {pb*100:.0f}% dip"
        print(f"{label:<15}{len(df):>6}{df['hit'].mean()*100:>10.1f}%{df['entry_offset'].mean():>16.1f}"
              f"{df['discount_vs_signal_pct'].mean():>14.1f}%{hits['effective_gain_if_hit_pct'].mean():>21.1f}%"
              f"{missed_pct:>16.1f}%")

        if pb > 0:
            achieved, missed_df = df[df["dip_achieved"]], df[~df["dip_achieved"]]
            print(f"    -> when dip WAS achieved (n={len(achieved)}): hit_rate={achieved['hit'].mean()*100:.1f}%, "
                  f"avg_discount={achieved['discount_vs_signal_pct'].mean():.1f}%, "
                  f"avg_eff_gain_if_hit={achieved[achieved['hit']]['effective_gain_if_hit_pct'].mean():.1f}%")
            print(f"    -> when dip NEVER came (n={len(missed_df)}, forced late fallback entry): "
                  f"hit_rate={missed_df['hit'].mean()*100:.1f}%, "
                  f"avg_discount={missed_df['discount_vs_signal_pct'].mean():.1f}%")

    print("\nColumns explained:")
    print("  hit_rate%            = % that still reached the ORIGINAL target price within the remaining session budget")
    print("  avg_days_waited      = avg sessions waited before entering (0 = bought signal day)")
    print("  avg_discount%        = avg entry price vs signal-day close (negative = bought cheaper)")
    print("  avg_eff_gain%_if_hit = actual %% gain from YOUR entry price when target was reached (>12% if you bought the dip)")
    print("  missed(no dip)%      = %% of trades where the requested pullback never came within the wait window"
          " (you'd have entered late at a worse price than intended, or should skip these)")


if __name__ == "__main__":
    main()
