"""Realistic hit-rate: does price reach +12% (TARGET_PCT) BEFORE it breaches the
same chandelier stop-loss get_stop_loss() would have printed at entry -- not just
"does it ever touch +12% within 45 sessions" (that ignores whether the stop-loss
would have knocked you out first). Uses the exact same chandelier_trail_stop()
call as engine/stop_loss.get_stop_loss() (entry_date=signal day -> stop is fixed
at entry_close - 3*ATR14, no lookahead), then walks forward day-by-day through the
same continuous OHLC history used live to see which level is breached first.
If both the stop and the target are touched on the same day, the stop is assumed
to hit first (conservative/worst case, since intraday order is unknown from EOD data).

NSE only (BSE bhavcopy has no delivery% column, but stop-loss/target math doesn't
need delivery -- can extend to BSE later if useful). Read-only analysis, does not
change any live selection logic.

Usage: python analysis/backtest_stop_vs_target.py
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "engine"))
sys.path.insert(0, r"C:\Mine\Compare")

from ensemble_model import build_ensemble, BACKTEST_START  # noqa: E402
from stop_loss import load_continuous_hist  # noqa: E402
from chandelier_stop import chandelier_trail_stop  # noqa: E402

TARGET_PCT = 0.12
FORWARD_WINDOW = 45
ATR_PERIOD = 14


def simulate(entry, stop, path):
    """path = DataFrame of forward daily High/Low rows (up to FORWARD_WINDOW), in date order.
    Returns 'TARGET' / 'STOPPED' / 'NEITHER'."""
    target_price = entry * (1 + TARGET_PCT)
    for _, day in path.iterrows():
        stopped = day["Low"] <= stop
        hit = day["High"] >= target_price
        if stopped and hit:
            return "STOPPED"   # same-day ambiguous -- assume worst case
        if stopped:
            return "STOPPED"
        if hit:
            return "TARGET"
    return "NEITHER"


def main():
    merged = build_ensemble()
    merged = merged[merged["DATE1"] >= pd.Timestamp(BACKTEST_START)].copy()
    picks = merged[(merged["votes"] >= 1) & merged["valid_forward_45"] & (merged["key_market"] == "NSE")].copy()

    hist = load_continuous_hist("NSE").rename(
        columns={"HIGH_PRICE": "High", "LOW_PRICE": "Low", "CLOSE_PRICE": "Close"})
    hist = hist.sort_values(["SYMBOL", "DATE1"]).reset_index(drop=True)
    by_symbol = {sym: g.set_index("DATE1") for sym, g in hist.groupby("SYMBOL", sort=False)}

    results = []
    for row in picks.itertuples():
        sym = row.key_symbol
        signal_date = pd.Timestamp(row.DATE1)
        g = by_symbol.get(sym)
        if g is None:
            continue
        hist_upto = g[g.index <= signal_date]
        if len(hist_upto) < ATR_PERIOD + 1:
            continue
        entry = hist_upto["Close"].iloc[-1]
        stop = chandelier_trail_stop(hist_upto, entry_date=signal_date, atr_period=ATR_PERIOD)
        if stop is None or stop >= entry:
            continue
        forward = g[g.index > signal_date].iloc[:FORWARD_WINDOW]
        if len(forward) < FORWARD_WINDOW:
            continue
        outcome = simulate(entry, stop, forward)
        results.append({"votes": row.votes, "outcome": outcome})

    df = pd.DataFrame(results)
    n = len(df)
    print(f"NSE picks with resolvable stop + full 45-day forward path: n={n}\n")

    print("OVERALL outcome (stop-loss-aware, same fixed chandelier stop as live pipeline):")
    print(df["outcome"].value_counts(normalize=True).mul(100).round(1).to_string(), "\n")

    print("BY VOTES:")
    tab = df.groupby("votes")["outcome"].value_counts(normalize=True).mul(100).round(1).unstack()
    counts = df.groupby("votes").size().rename("n")
    print(pd.concat([counts, tab], axis=1).to_string())


if __name__ == "__main__":
    main()
