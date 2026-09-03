"""Does high delivery% (real investors taking delivery, not day-trading churn) combined
with elevated volume on the entry day change how a BUY pick performs -- vs a volume
spike with LOW delivery% (more likely speculative/intraday churn, not real accumulation)?

NSE only (BSE bhavcopy has no delivery% column). Reuses build_market_context() from
backtest_runrisk_candle.py so the OHLCV/volume/delivery baselines are computed exactly
once, the same way, everywhere. Informational analysis only -- does not change
combined_verdict() or any live selection logic.

Usage: python analysis/backtest_delivery_volume.py
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "engine"))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
sys.path.insert(0, r"C:\Mine\Compare")

from backtest_runrisk_candle import build_market_context  # noqa: E402
from ensemble_model import build_ensemble, BACKTEST_START  # noqa: E402


def bucket_label(qseries):
    return pd.qcut(qseries, 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"], duplicates="drop")


def summarize(df, group_cols):
    g = df.groupby(group_cols, observed=True)
    out = g.agg(
        n=("fwd_hit12_45", "size"),
        hit_rate_45=("fwd_hit12_45", lambda s: (s == 1).mean() * 100),
        avg_min_gain_45=("fwd_min_gain_45", lambda s: s.mean() * 100),
        avg_final_gain_45=("fwd_final_gain_45", lambda s: s.mean() * 100),
        avg_5sess_dip=("fwd_min_ret_5", lambda s: s.mean() * 100),
        avg_5sess_ret=("fwd_ret_5", lambda s: s.mean() * 100),
    ).round(1)
    return out


def main():
    print("Building ensemble (votes + 45-session forward labels)...")
    merged = build_ensemble()
    merged = merged[merged["DATE1"] >= pd.Timestamp(BACKTEST_START)].copy()
    picks = merged[(merged["votes"] >= 1) & merged["valid_forward_45"]].copy()

    print("Building NSE OHLCV/volume/delivery context...")
    ctx = build_market_context("NSE")
    ctx["key_symbol"], ctx["key_market"] = ctx["SYMBOL"].str.upper(), ctx["Market"].str.upper()

    df = picks.merge(ctx, on=["key_symbol", "key_market", "DATE1"], how="inner")
    df = df[df["key_market"] == "NSE"].dropna(subset=["Vol20Avg", "Deliv20Avg", "ctxDELIV_PER"]).copy()
    df = df[df["Vol20Avg"] > 0].copy()
    df["VolRatio"] = df["ctxTTL_TRD_QNTY"] / df["Vol20Avg"]
    print(f"NSE picks with resolvable volume+delivery baseline: n={len(df)}\n")

    df["DelivBucket"] = bucket_label(df["ctxDELIV_PER"])
    df["VolBucket"] = bucket_label(df["VolRatio"])

    print("=" * 100)
    print("BY DELIVERY% QUARTILE ALONE (today's raw delivery%, NSE)")
    print("=" * 100)
    print(summarize(df, "DelivBucket").to_string())

    print("\n" + "=" * 100)
    print("BY VOLUME-RATIO QUARTILE ALONE (today's volume vs 20-session baseline, NSE)")
    print("=" * 100)
    print(summarize(df, "VolBucket").to_string())

    print("\n" + "=" * 100)
    print("CROSS-TAB: DELIVERY% QUARTILE x VOLUME-RATIO QUARTILE")
    print("=" * 100)
    print(summarize(df, ["DelivBucket", "VolBucket"]).to_string())

    print("\n" + "=" * 100)
    print("FOCUSED COMPARISON: high-volume days only (Q4 volume) -- high vs low delivery%")
    print("=" * 100)
    hv = df[df["VolBucket"] == "Q4 (highest)"]
    hv_bucket = pd.cut(hv["ctxDELIV_PER"], bins=[-1, 30, 50, 200],
                        labels=["LOW deliv <30% (speculative?)", "MED deliv 30-50%", "HIGH deliv >50% (real buying?)"])
    print(summarize(hv.assign(DelivTier=hv_bucket), "DelivTier").to_string())

    print("\nColumns explained:")
    print("  hit_rate_45      = % that reached +12% within 45 sessions (pipeline objective)")
    print("  avg_min_gain_45  = avg worst drawdown reached within 45 sessions")
    print("  avg_final_gain_45= avg gain/loss sitting at when the 45-session window ends")
    print("  avg_5sess_dip    = avg worst dip in just the next 5 sessions")
    print("  avg_5sess_ret    = avg plain return after exactly 5 sessions")


if __name__ == "__main__":
    main()
