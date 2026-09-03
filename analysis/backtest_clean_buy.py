"""Backtest the CLEAN BUY tier (see /memories/repo/daily-pipeline-review.md) over real
historical data, reusing the exact same votes/score computation as C:\\Mine\\Compare\\ensemble_model.py
so results are consistent with the already-quoted HIST_WINRATE_BY_VOTES numbers.

CLEAN BUY proxy filter (delivery% not available in the ML labeled dataset, so the
volume/candle/trend legs are approximated from ML features instead of raw bhavcopy):
  - votes >= 1 (the only tier with both winrate>=65% AND n>=1000, see repo memory)
  - vol_ratio, turnover_expansion, rel_volume_today all below their own 90th percentile
    (not an outlier-volume day)
  - close_strength_avg3 >= 0.6 (close sits in the upper 40% of the recent 3-day range,
    i.e. no big rejection wick)
  - ret_20 <= 0.15 (not already chasing a move that ran +15%+ in the last ~4 weeks)

Usage: python analysis/backtest_clean_buy.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Mine\Compare")
from ensemble_model import build_ensemble, BACKTEST_START  # noqa: E402

WEEK_SESSIONS = 5
FALLBACK_STOP_PCT = -0.08  # matches ProAsh's paper_trade_prompt.py fallback stop


def add_week_forward_return(df):
    df = df.sort_values(["key_symbol", "DATE1"]).reset_index(drop=True)
    parts = []
    for _, g in df.groupby("key_symbol", sort=False):
        g = g.copy()
        g["fwd_close_5"] = g["CLOSE_PRICE"].shift(-WEEK_SESSIONS)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    out["fwd_ret_5"] = out["fwd_close_5"] / out["CLOSE_PRICE"] - 1
    return out


def main():
    merged = build_ensemble()
    merged = merged[merged["DATE1"] >= pd.Timestamp(BACKTEST_START)].copy()
    merged = add_week_forward_return(merged)

    base = merged[(merged["votes"] >= 1) & merged["fwd_ret_5"].notna()].copy()
    print(f"All votes>=1 rows since {BACKTEST_START}: n={len(base)}")
    print(f"  win rate (fwd 5-session return > 0): {(base['fwd_ret_5'] > 0).mean()*100:.1f}%")
    print(f"  avg fwd 5-session return: {base['fwd_ret_5'].mean()*100:.2f}%")
    print(f"  median fwd 5-session return: {base['fwd_ret_5'].median()*100:.2f}%\n")

    q90 = base[["vol_ratio_af", "turnover_expansion_af", "rel_volume_today_af"]].quantile(0.90)
    clean = base[
        (base["vol_ratio_af"] <= q90["vol_ratio_af"])
        & (base["turnover_expansion_af"] <= q90["turnover_expansion_af"])
        & (base["rel_volume_today_af"] <= q90["rel_volume_today_af"])
        & (base["close_strength_avg3_af"] >= 0.6)
        & (base["ret_20_af"] <= 0.15)
    ].copy()

    print(f"CLEAN BUY proxy filter since {BACKTEST_START}: n={len(clean)}")
    if clean.empty:
        print("  (no rows passed the filter)")
        return
    win = clean["fwd_ret_5"] > 0
    hit_stop = clean["fwd_ret_5"] <= FALLBACK_STOP_PCT
    print(f"  win rate (fwd 5-session return > 0): {win.mean()*100:.1f}%")
    print(f"  avg fwd 5-session return: {clean['fwd_ret_5'].mean()*100:.2f}%")
    print(f"  median fwd 5-session return: {clean['fwd_ret_5'].median()*100:.2f}%")
    print(f"  hit fallback stop ({FALLBACK_STOP_PCT*100:.0f}%) within the week: {hit_stop.mean()*100:.1f}%")
    print(f"  best week: {clean['fwd_ret_5'].max()*100:.1f}%  worst week: {clean['fwd_ret_5'].min()*100:.1f}%")

    print("\nBy market:")
    print(clean.groupby("key_market")["fwd_ret_5"].agg(n="count", win_rate=lambda s: (s > 0).mean()*100,
                                                          avg_ret=lambda s: s.mean()*100).round(2))

    resolved = clean[clean["valid_forward_45"]]
    print(f"\nSame CLEAN BUY rows, but the model's actual designed horizon (45 sessions, ~2 months), n={len(resolved)}:")
    print(f"  hit +12% target within 45 sessions: {(resolved['fwd_hit12_45'] == 1).mean()*100:.1f}%")
    print(f"  avg max gain reached within 45 sessions: {resolved['fwd_max_gain_45'].mean()*100:.2f}%")


if __name__ == "__main__":
    main()
