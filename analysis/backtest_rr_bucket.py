"""Does ranking by R:R magnitude (tighter stop -> higher R:R since TARGET_PCT is fixed)
actually correlate with a better STOP-LOSS-AWARE hit rate, or does a tighter stop just
get clipped by ordinary noise more often, eating into the payoff advantage?
Reuses the exact same stop-vs-target-ordering simulation as backtest_stop_vs_target.py
(same fixed chandelier stop as get_stop_loss(), same day-by-day walk, same
same-day-ambiguous-assume-stopped rule) -- just buckets the outcome by R:R this time.

NSE only. Read-only analysis, does not change any live selection logic.

Usage: python analysis/backtest_rr_bucket.py
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "engine"))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
sys.path.insert(0, r"C:\Mine\Compare")

from ensemble_model import build_ensemble, BACKTEST_START  # noqa: E402
from stop_loss import load_continuous_hist  # noqa: E402
from chandelier_stop import chandelier_trail_stop, stop_distance_pct  # noqa: E402
from backtest_stop_vs_target import simulate, TARGET_PCT, FORWARD_WINDOW, ATR_PERIOD  # noqa: E402

RR_BINS = [0, 0.8, 1.0, 1.2, 1.5, float("inf")]
RR_LABELS = ["<0.8", "0.8-1.0", "1.0-1.2", "1.2-1.5", ">=1.5"]


def outcome_and_final_gain(entry, stop, path):
    """Same ordering rule as simulate(), plus the day-45 close/entry gain ratio
    for the NEITHER case (needed to compute a fair EV, not just hit-rate)."""
    target_price = entry * (1 + TARGET_PCT)
    for _, day in path.iterrows():
        if day["Low"] <= stop:
            return "STOPPED", None
        if day["High"] >= target_price:
            return "TARGET", None
    final_close = path["Close"].iloc[-1]
    return "NEITHER", (final_close / entry - 1)


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
        stop_pct = stop_distance_pct(entry, stop)
        if stop_pct <= 0:
            continue
        rr = TARGET_PCT * 100 / stop_pct
        forward = g[g.index > signal_date].iloc[:FORWARD_WINDOW]
        if len(forward) < FORWARD_WINDOW:
            continue
        outcome, neither_gain = outcome_and_final_gain(entry, stop, forward)
        # EV in units of RISK_PER_TRADE (the fixed rupee risk every trade is sized to):
        # TARGET pays +rr, STOPPED pays -1, NEITHER pays whatever day-45 gain%/stop_pct% worked out to.
        if outcome == "TARGET":
            ev = rr
        elif outcome == "STOPPED":
            ev = -1.0
        else:
            ev = (neither_gain * 100) / stop_pct
        results.append({"votes": row.votes, "rr": rr, "outcome": outcome, "ev": ev})

    df = pd.DataFrame(results)
    df["rr_bucket"] = pd.cut(df["rr"], bins=RR_BINS, labels=RR_LABELS)
    n = len(df)
    print(f"NSE picks with resolvable stop + full 45-day forward path: n={n}\n")

    print("BY R:R BUCKET (stop-loss-aware, same fixed chandelier stop as live pipeline):")
    tab = df.groupby("rr_bucket", observed=True)["outcome"].value_counts(normalize=True).mul(100).round(1).unstack()
    counts = df.groupby("rr_bucket", observed=True).size().rename("n")
    avg_rr = df.groupby("rr_bucket", observed=True)["rr"].mean().rename("avg_rr").round(2)
    avg_ev = df.groupby("rr_bucket", observed=True)["ev"].mean().rename("avg_EV(x risk/trade)").round(3)
    print(pd.concat([counts, avg_rr, tab, avg_ev], axis=1).reindex(RR_LABELS).to_string())


if __name__ == "__main__":
    main()
