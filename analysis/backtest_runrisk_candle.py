"""Backtest: does the RunRisk (already-extended-move) context and candlestick pattern
at entry actually predict outcomes -- both the pipeline's designed 45-session target
AND the short-horizon "does it dip first, then just get back to flat" chop pattern the
user is seeing in real trading. Informational-only features (see engine/candlestick_patterns.py)
-- this script does NOT change combined_verdict() or any live selection logic.

Reuses:
  - C:\\Mine\\Compare\\ensemble_model.py's build_ensemble() for votes + the pipeline's own
    45-session forward labels (fwd_hit12_45, fwd_min_gain_45, fwd_final_gain_45) -- same
    data source already used by analysis/backtest_clean_buy.py.
  - proash_pipeline.py's build_market() for the exact same split-adjusted OHLCV history
    the live tool sees, to compute candle pattern + volume/delivery/recent-run context
    and a genuine short-horizon (5-session) dip metric.

Usage: python analysis/backtest_runrisk_candle.py
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
from candlestick_patterns import classify_candle  # noqa: E402
from ensemble_model import build_ensemble, BACKTEST_START  # noqa: E402

FWD_HORIZON = 5
VOL_EXTREME, VOL_ELEVATED = 20.0, 3.0
DELIV_COLLAPSE_MAX, DELIV_BASELINE_MIN = 20.0, 35.0
BAD_CANDLES = {"Bearish Engulfing", "Shooting Star", "Hanging Man", "Rejection Wick"}


def build_market_context(market):
    """Split-adjusted OHLCV history (same as the live tool) + PrevOpen/PrevClose for
    candle patterns, 20-session volume/delivery baselines, and forward 1-5 session
    closes for a genuine short-horizon dip/return metric."""
    hist = build_market(market).sort_values(["SYMBOL", "DATE1"]).reset_index(drop=True)
    g = hist.groupby("SYMBOL", sort=False)
    hist["PrevOpen"] = g["OPEN_PRICE"].shift(1)
    hist["PrevClose"] = g["CLOSE_PRICE"].shift(1)
    hist["Vol20Avg"] = g["TTL_TRD_QNTY"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    if "DELIV_PER" in hist.columns:
        hist["Deliv20Avg"] = g["DELIV_PER"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    else:
        hist["Deliv20Avg"] = np.nan
        hist["DELIV_PER"] = np.nan
    for h in range(1, FWD_HORIZON + 1):
        hist[f"fwd_close_{h}"] = g["CLOSE_PRICE"].shift(-h)
    fwd_cols = [f"fwd_close_{h}" for h in range(1, FWD_HORIZON + 1)]
    hist["fwd_min_close_5"] = hist[fwd_cols].min(axis=1, skipna=True)
    hist["fwd_ret_5"] = hist[f"fwd_close_{FWD_HORIZON}"] / hist["CLOSE_PRICE"] - 1
    hist["fwd_min_ret_5"] = hist["fwd_min_close_5"] / hist["CLOSE_PRICE"] - 1
    hist["Market"] = market
    keep = ["SYMBOL", "DATE1", "Market", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE",
            "PrevOpen", "PrevClose", "TTL_TRD_QNTY", "Vol20Avg", "DELIV_PER", "Deliv20Avg",
            "fwd_ret_5", "fwd_min_ret_5"]
    hist = hist[keep].rename(columns={
        "OPEN_PRICE": "ctxOPEN_PRICE", "HIGH_PRICE": "ctxHIGH_PRICE", "LOW_PRICE": "ctxLOW_PRICE",
        "CLOSE_PRICE": "ctxCLOSE_PRICE", "TTL_TRD_QNTY": "ctxTTL_TRD_QNTY", "DELIV_PER": "ctxDELIV_PER",
    })
    return hist


def classify_row(r):
    return classify_candle(r["ctxOPEN_PRICE"], r["ctxHIGH_PRICE"], r["ctxLOW_PRICE"], r["ctxCLOSE_PRICE"],
                            r["PrevOpen"], r["PrevClose"])


def vol_flag(r):
    if pd.isna(r["Vol20Avg"]) or r["Vol20Avg"] <= 0:
        return "-"
    ratio = r["ctxTTL_TRD_QNTY"] / r["Vol20Avg"]
    return "EXTREME" if ratio >= VOL_EXTREME else "ELEVATED" if ratio >= VOL_ELEVATED else "NORMAL"


def deliv_flag(r):
    if pd.isna(r["Deliv20Avg"]) or pd.isna(r["ctxDELIV_PER"]):
        return "-"
    return ("COLLAPSED" if r["ctxDELIV_PER"] <= DELIV_COLLAPSE_MAX and r["Deliv20Avg"] >= DELIV_BASELINE_MIN
            else "NORMAL")


def summarize(df, group_col):
    g = df.groupby(group_col)
    out = g.agg(
        n=("fwd_hit12_45", "size"),
        hit_rate_45=("fwd_hit12_45", lambda s: (s == 1).mean() * 100),
        avg_min_gain_45=("fwd_min_gain_45", lambda s: s.mean() * 100),
        avg_final_gain_45=("fwd_final_gain_45", lambda s: s.mean() * 100),
        avg_5sess_dip=("fwd_min_ret_5", lambda s: s.mean() * 100),
        avg_5sess_ret=("fwd_ret_5", lambda s: s.mean() * 100),
    ).round(1)
    return out.sort_values("n", ascending=False)


def main():
    print("Building ensemble (votes + 45-session forward labels)...")
    merged = build_ensemble()
    merged = merged[merged["DATE1"] >= pd.Timestamp(BACKTEST_START)].copy()
    picks = merged[(merged["votes"] >= 1) & merged["valid_forward_45"]].copy()
    print(f"votes>=1 & resolved-45-session picks since {BACKTEST_START}: n={len(picks)}")

    print("Building NSE+BSE split-adjusted OHLCV context (candle + volume/delivery + 5-session dip)...")
    ctx = pd.concat([build_market_context("NSE"), build_market_context("BSE")], ignore_index=True)
    ctx["key_symbol"], ctx["key_market"] = ctx["SYMBOL"].str.upper(), ctx["Market"].str.upper()

    df = picks.merge(ctx, on=["key_symbol", "key_market", "DATE1"], how="inner")
    print(f"After joining OHLCV context: n={len(df)} (some rows drop if OHLC context unavailable)\n")

    df["Candle"] = df.apply(classify_row, axis=1)
    df["VolFlag"] = df.apply(vol_flag, axis=1)
    df["DelivFlag"] = df.apply(deliv_flag, axis=1)

    print("=" * 100)
    print("BY CANDLE PATTERN (at entry day)")
    print("=" * 100)
    print(summarize(df, "Candle").to_string())

    print("\n" + "=" * 100)
    print("BY VOLUME FLAG (today's volume vs 20-session baseline)")
    print("=" * 100)
    print(summarize(df, "VolFlag").to_string())

    nse_df = df[df["key_market"] == "NSE"]
    print("\n" + "=" * 100)
    print("BY DELIVERY FLAG (NSE only -- today's delivery% vs 20-session baseline)")
    print("=" * 100)
    print(summarize(nse_df, "DelivFlag").to_string())

    print("\n" + "=" * 100)
    print("COMBO: 'RED FLAG' (bad candle OR extreme volume OR collapsed delivery) vs 'CLEAN'")
    print("=" * 100)
    red_flag = (df["Candle"].isin(BAD_CANDLES) | (df["VolFlag"] == "EXTREME") | (df["DelivFlag"] == "COLLAPSED"))
    df["RedFlagBucket"] = np.where(red_flag, "RED FLAG", "CLEAN")
    print(summarize(df, "RedFlagBucket").to_string())

    print("\nColumns explained:")
    print("  hit_rate_45      = % that reached +12% at some point within 45 sessions (existing pipeline objective)")
    print("  avg_min_gain_45  = avg worst drawdown reached within 45 sessions (negative = how deep the dip went)")
    print("  avg_final_gain_45= avg gain/loss actually sitting at when the 45-session window ends")
    print("  avg_5sess_dip    = avg worst dip in just the next 5 sessions (the 'falls first' user complaint)")
    print("  avg_5sess_ret    = avg plain return after exactly 5 sessions (the 'net no gain yet' user complaint)")


if __name__ == "__main__":
    main()
