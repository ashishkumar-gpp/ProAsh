"""Does a SUSTAINED low delivery% (trailing 10 sessions, NOT including entry day)
identify pumps that VolRatioToday (today's volume / 20-session baseline) misses --
specifically the case where a stock has been pumping for 2+ weeks straight, so its
own 20-session baseline is already inflated and today's ratio reads "calm" even
though the stock is still actively pumping (real case: BAJAJHIND, 02-Sep-2026 run,
VolRatioToday=0.4x yet delivery% stuck at 13-19% for 10+ straight sessions).

NSE only (BSE bhavcopy has no delivery% column). Reuses build_market_context() from
backtest_runrisk_candle.py. Informational analysis only -- does not change
combined_verdict() or any live selection logic until validated.

Usage: python analysis/backtest_sustained_pump_filter.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "engine"))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
sys.path.insert(0, r"C:\Mine\Compare")

from proash_pipeline import build_market  # noqa: E402
from ensemble_model import build_ensemble, BACKTEST_START  # noqa: E402

SUSTAINED_WINDOW = 10        # trailing sessions, excluding entry day
SUSTAINED_DELIV_MAX = 20.0   # avg delivery% below this over the window = sustained pump


def build_context(market):
    hist = build_market(market).sort_values(["SYMBOL", "DATE1"]).reset_index(drop=True)
    g = hist.groupby("SYMBOL", sort=False)
    hist["Vol20Avg"] = g["TTL_TRD_QNTY"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    if "DELIV_PER" in hist.columns:
        hist["Deliv20Avg"] = g["DELIV_PER"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
        hist["Deliv10AvgTrailing"] = g["DELIV_PER"].transform(
            lambda s: s.shift(1).rolling(SUSTAINED_WINDOW, min_periods=SUSTAINED_WINDOW).mean())
    else:
        hist["Deliv20Avg"] = np.nan
        hist["Deliv10AvgTrailing"] = np.nan
        hist["DELIV_PER"] = np.nan
    hist["Market"] = market
    keep = ["SYMBOL", "DATE1", "Market", "TTL_TRD_QNTY", "Vol20Avg", "DELIV_PER",
            "Deliv20Avg", "Deliv10AvgTrailing"]
    hist = hist[keep].rename(columns={"TTL_TRD_QNTY": "ctxTTL_TRD_QNTY", "DELIV_PER": "ctxDELIV_PER"})
    return hist


def summarize(df, group_col):
    g = df.groupby(group_col, observed=True)
    out = g.agg(
        n=("fwd_hit12_45", "size"),
        hit_rate_45=("fwd_hit12_45", lambda s: (s == 1).mean() * 100),
        avg_min_gain_45=("fwd_min_gain_45", lambda s: s.mean() * 100),
        avg_final_gain_45=("fwd_final_gain_45", lambda s: s.mean() * 100),
    ).round(1)
    return out


def main():
    print("Building ensemble (votes + 45-session forward labels)...")
    merged = build_ensemble()
    merged = merged[merged["DATE1"] >= pd.Timestamp(BACKTEST_START)].copy()
    picks = merged[(merged["votes"] >= 1) & merged["valid_forward_45"]].copy()

    print("Building NSE volume/delivery context (20-session + trailing-10-session)...")
    ctx = build_context("NSE")
    ctx["key_symbol"], ctx["key_market"] = ctx["SYMBOL"].str.upper(), ctx["Market"].str.upper()

    df = picks.merge(ctx, on=["key_symbol", "key_market", "DATE1"], how="inner")
    df = df[df["key_market"] == "NSE"].dropna(
        subset=["Vol20Avg", "Deliv20Avg", "Deliv10AvgTrailing", "ctxDELIV_PER"]).copy()
    df = df[df["Vol20Avg"] > 0].copy()
    df["VolRatio"] = df["ctxTTL_TRD_QNTY"] / df["Vol20Avg"]
    print(f"NSE picks with resolvable context: n={len(df)}\n")

    df["SustainedPump"] = df["Deliv10AvgTrailing"] < SUSTAINED_DELIV_MAX
    df["CalmToday"] = df["VolRatio"] <= 1.0   # today reads "calm" by the live ranking's own metric

    print("=" * 100)
    print(f"BY SustainedPump FLAG ALONE (trailing {SUSTAINED_WINDOW}-session avg delivery% < {SUSTAINED_DELIV_MAX}%)")
    print("=" * 100)
    print(summarize(df, "SustainedPump").to_string())

    print("\n" + "=" * 100)
    print("FOCUSED: among picks that read 'CALM TODAY' (VolRatio<=1.0x, what the live table would rank highest) --")
    print("does the SustainedPump flag still separate good from bad within that calm-looking pool?")
    print("=" * 100)
    calm = df[df["CalmToday"]]
    print(f"n calm-today picks: {len(calm)}")
    print(summarize(calm, "SustainedPump").to_string())

    print("\n" + "=" * 100)
    print("CROSS-TAB: CalmToday x SustainedPump")
    print("=" * 100)
    print(summarize(df, ["CalmToday", "SustainedPump"]).to_string())

    print("\nColumns explained:")
    print("  hit_rate_45       = % that reached +12% within 45 sessions (pipeline objective)")
    print("  avg_min_gain_45   = avg worst drawdown reached within 45 sessions")
    print("  avg_final_gain_45 = avg gain/loss sitting at when the 45-session window ends")


if __name__ == "__main__":
    main()
