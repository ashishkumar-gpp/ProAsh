"""
BSE pre-move factor analysis (mirrors analysis/premove_factor_analysis.py, adapted
for BSE bhavcopy schema).

Question: same as the NSE version -- what does a stock look like 3-10 sessions
BEFORE a +7%+ pop, using BhavCopy_BSE_CM_*.csv files in bse_bhavcopy/?

Two differences vs the NSE script, both due to data availability:
  - BSE bhavcopy has no delivery-quantity/delivery-% field, so delivery-based
    factors (Delivery Trend, Delivery Expansion) are NOT computed here.
  - BSE bhavcopy has no companion index-level file (no Sensex closing series
    downloaded). Sensex-tracking ETFs (SENSEXBEES etc.) exist in the data but
    carry their own NAV tracking noise, so instead the "market" benchmark used
    for Relative Strength is the cross-sectional median same-day return of the
    liquid BSE universe itself (a breadth-based benchmark) -- documented here
    so results aren't mistaken for true index-relative RS.

Read-only analysis script; does not change AshishTrade_v4.txt scoring.
"""
import glob
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BSE_DIR = os.path.join(ROOT, "bse_bhavcopy")
HIST_BSE_DIR = os.path.join(ROOT, "analysis", "history-data", "bse_bhavcopy")  # ~1yr backfill, already on disk

TARGET_GAIN = 0.07
FWD_WINDOW = 10
MIN_TURNOVER = 200_000_000.0  # Rs 20 Cr, same threshold as the NSE analysis (2000 lacs)
MIN_PRICE = 20.0
LIQUID_GROUPS = {"A", "B"}  # regular rolling-settlement equity, excludes T2T/SME/Z/F/G


# --------------------------------------------------------------- loading
def load_bhav():
    files = sorted(glob.glob(os.path.join(BSE_DIR, "BhavCopy_BSE_CM_*.CSV")))
    files += sorted(glob.glob(os.path.join(HIST_BSE_DIR, "BhavCopy_BSE_CM_*.CSV")))
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for c in df.select_dtypes(include=["object", "str"]).columns:
        df[c] = df[c].astype(str).str.strip()
    df = df[df["SctySrs"].isin(LIQUID_GROUPS)].copy()
    df["DATE1"] = pd.to_datetime(df["TradDt"])
    num_cols = ["OpnPric", "HghPric", "LwPric", "ClsPric", "PrvsClsgPric",
                "TtlTradgVol", "TtlTrfVal", "TtlNbOfTxsExctd"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.rename(columns={
        "TckrSymb": "SYMBOL", "OpnPric": "OPEN_PRICE", "HghPric": "HIGH_PRICE",
        "LwPric": "LOW_PRICE", "ClsPric": "CLOSE_PRICE", "PrvsClsgPric": "PREV_CLOSE",
        "TtlTradgVol": "TTL_TRD_QNTY", "TtlTrfVal": "TURNOVER", "TtlNbOfTxsExctd": "NO_OF_TRADES",
    })
    df = df.drop_duplicates(subset=["SYMBOL", "DATE1"])
    return df.sort_values(["SYMBOL", "DATE1"]).reset_index(drop=True)


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ------------------------------------------------------- feature pipeline
def build_features(df):
    g = df.groupby("SYMBOL", group_keys=False)

    for n in (1, 3, 5, 10):
        df[f"ret_{n}"] = g["CLOSE_PRICE"].transform(lambda s, n=n: s.pct_change(n))

    # breadth benchmark: cross-sectional median daily return of the whole liquid
    # universe that day, compounded over N sessions (proxy for a true index level)
    day_med_ret1 = df.groupby("DATE1")["ret_1"].median()
    day_bench = (1 + day_med_ret1).cumprod()
    bench_n = {n: day_bench.pct_change(n) for n in (3, 5, 10)}
    for n in (3, 5, 10):
        df[f"rs_{n}"] = df[f"ret_{n}"] - df["DATE1"].map(bench_n[n])

    df["vol_avg5"] = g["TTL_TRD_QNTY"].transform(lambda s: s.rolling(5).mean())
    df["vol_avg20"] = g["TTL_TRD_QNTY"].transform(lambda s: s.rolling(20).mean())
    df["vol_ratio"] = df["vol_avg5"] / df["vol_avg20"]
    df["rel_volume_today"] = df["TTL_TRD_QNTY"] / df["vol_avg20"]

    df["trades_avg5"] = g["NO_OF_TRADES"].transform(lambda s: s.rolling(5).mean())
    df["trades_avg20"] = g["NO_OF_TRADES"].transform(lambda s: s.rolling(20).mean())
    df["trades_ratio"] = df["trades_avg5"] / df["trades_avg20"]

    df["turn_avg20"] = g["TURNOVER"].transform(lambda s: s.rolling(20).mean())
    df["turnover_expansion"] = df["TURNOVER"] / df["turn_avg20"]

    df["day_range_pct"] = (df["HIGH_PRICE"] - df["LOW_PRICE"]) / df["CLOSE_PRICE"]
    df["range_avg5"] = g["day_range_pct"].transform(lambda s: s.rolling(5).mean())
    df["range_avg20"] = g["day_range_pct"].transform(lambda s: s.rolling(20).mean())
    df["range_compression"] = df["range_avg5"] / df["range_avg20"]

    df["close_strength"] = (df["CLOSE_PRICE"] - df["LOW_PRICE"]) / (
        df["HIGH_PRICE"] - df["LOW_PRICE"]).replace(0, np.nan)
    df["close_strength_avg3"] = g["close_strength"].transform(lambda s: s.rolling(3).mean())

    df["period_high"] = g["CLOSE_PRICE"].transform(lambda s: s.shift(1).cummax())
    df["resistance_distance"] = (df["period_high"] - df["CLOSE_PRICE"]) / df["period_high"]

    df["ema20"] = g["CLOSE_PRICE"].transform(lambda s: ema(s, 20))
    df["ema50"] = g["CLOSE_PRICE"].transform(lambda s: ema(s, 50))
    df["ema20_gt_ema50"] = (df["ema20"] > df["ema50"]).astype(float)
    df["rsi14"] = g["CLOSE_PRICE"].transform(lambda s: rsi(s, 14))

    df["gap_pct"] = (df["OPEN_PRICE"] - df["PREV_CLOSE"]) / df["PREV_CLOSE"]
    df["gap_avg5"] = g["gap_pct"].transform(lambda s: s.rolling(5).mean())

    df["ret1_std5"] = g["ret_1"].transform(lambda s: s.rolling(5).std())
    df["ret1_std20"] = g["ret_1"].transform(lambda s: s.rolling(20).std())
    df["volatility_contraction"] = df["ret1_std5"] / df["ret1_std20"]

    up_day = (df["ret_1"] > 0).astype(int)
    grp = df["SYMBOL"]
    df["consecutive_up_days"] = (
        up_day.groupby([grp, (up_day != up_day.shift()).groupby(grp).cumsum()]).cumsum() * up_day
    )

    def fwd_stats(s):
        arr = s.values
        max_ret = np.full(len(arr), np.nan)
        peak_day = np.full(len(arr), np.nan)
        for i in range(len(arr) - 1):
            window = arr[i + 1: i + 1 + FWD_WINDOW]
            if len(window) > 0:
                rel = window / arr[i] - 1
                j = int(np.nanargmax(rel))
                max_ret[i] = rel[j]
                peak_day[i] = j + 1
        return pd.DataFrame({"fwd_max_ret": max_ret, "peak_day": peak_day}, index=s.index)

    fwd = g["CLOSE_PRICE"].apply(fwd_stats)
    fwd.index = fwd.index.droplevel(0) if isinstance(fwd.index, pd.MultiIndex) else fwd.index
    df["fwd_max_ret"] = fwd["fwd_max_ret"]
    df["peak_day"] = fwd["peak_day"]
    df["hit"] = (df["fwd_max_ret"] >= TARGET_GAIN).astype(float)
    return df


FACTORS = [
    ("rs_10", "Relative Strength vs BSE breadth (10d)"),
    ("rs_5", "Relative Strength vs BSE breadth (5d)"),
    ("rs_3", "Relative Strength vs BSE breadth (3d)"),
    ("ret_3", "3-Day Momentum"),
    ("ret_5", "5-Day Momentum"),
    ("ret_1", "1-Day Momentum"),
    ("vol_ratio", "Volume Ratio (5d/20d)"),
    ("rel_volume_today", "Relative Volume Today"),
    ("trades_ratio", "No. of Trades Ratio (5d/20d)"),
    ("turnover_expansion", "Turnover Expansion (today/20d avg)"),
    ("range_compression", "Range Compression (VCP-style)"),
    ("volatility_contraction", "Volatility Contraction (ret stdev 5d/20d)"),
    ("close_strength_avg3", "Close Strength (3d avg)"),
    ("resistance_distance", "Distance from Period High (lower=closer)"),
    ("rsi14", "RSI(14)"),
    ("ema20_gt_ema50", "EMA20 > EMA50 (trend flag)"),
    ("gap_avg5", "Avg Gap-Up %% (5d)"),
    ("consecutive_up_days", "Consecutive Up-Days"),
]


def cohens_d(a, b):
    a, b = a.dropna(), b.dropna()
    pooled_std = np.sqrt(((a.std() ** 2) + (b.std() ** 2)) / 2)
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else np.nan


def rank_factors(liquid):
    rows = []
    for col, label in FACTORS:
        d = liquid.dropna(subset=[col, "hit"])
        if len(d) < 100:
            continue
        base = d["hit"].mean()
        try:
            d = d.copy()
            d["q"] = pd.qcut(d[col], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        top_q = d["q"].max()
        top_rate = d.loc[d["q"] == top_q, "hit"].mean()
        bot_rate = d.loc[d["q"] == 0, "hit"].mean()
        best_rate, best_side = (top_rate, "top") if top_rate >= bot_rate else (bot_rate, "bottom")
        lift = best_rate / base if base > 0 else np.nan
        eff = cohens_d(d.loc[d["hit"] == 1, col], d.loc[d["hit"] == 0, col])
        corr = d[col].corr(d["hit"])
        rows.append({
            "Factor": label, "BaseRate": base, "BestQuintile": best_side,
            "BestQuintileRate": best_rate, "Lift": lift, "CohensD": eff,
            "PointBiserialCorr": corr, "N": len(d),
        })
    out = pd.DataFrame(rows).sort_values("Lift", ascending=False).reset_index(drop=True)
    return out


def case_studies(liquid_all, n_cases=15):
    """One real single-day +7% pop per recent trading day, with pre-move snapshots."""
    pops = liquid_all[liquid_all["ret_1"] >= TARGET_GAIN].copy()
    pops = pops.sort_values(["DATE1", "ret_1"], ascending=[False, False])
    pops = pops.groupby("DATE1", as_index=False).first().sort_values("DATE1", ascending=False).head(n_cases)

    lookbacks = [3, 5, 10]
    snap_cols = ["rs_10", "vol_ratio", "range_compression", "volatility_contraction",
                 "close_strength_avg3", "resistance_distance", "consecutive_up_days"]

    records = []
    for _, row in pops.iterrows():
        sym = row["SYMBOL"]
        sub = liquid_all[liquid_all["SYMBOL"] == sym].sort_values("DATE1").reset_index(drop=True)
        pop_idx = sub.index[sub["DATE1"] == row["DATE1"]]
        if len(pop_idx) == 0:
            continue
        pop_idx = pop_idx[0]
        rec = {"Symbol": sym, "PopDate": row["DATE1"].date(), "PopRet1d": row["ret_1"]}
        for lb in lookbacks:
            i = pop_idx - lb
            if i < 0:
                continue
            for c in snap_cols:
                rec[f"{c}_T-{lb}"] = sub.loc[i, c] if i in sub.index else np.nan
        records.append(rec)
    return pd.DataFrame(records)


def main():
    bhav = load_bhav()
    bhav = build_features(bhav)

    liquid_all = bhav[(bhav["TURNOVER"] >= MIN_TURNOVER) & (bhav["CLOSE_PRICE"] >= MIN_PRICE)].copy()
    liquid = liquid_all.dropna(subset=["hit"])

    print("=" * 90)
    print("BSE PRE-MOVE FACTOR ANALYSIS")
    print(f"Universe: liquid subset (A/B group, turnover>=Rs20Cr, price>=Rs20), "
          f"{liquid['SYMBOL'].nunique()} symbols, {liquid['DATE1'].min().date()} - "
          f"{liquid['DATE1'].max().date()}")
    print(f"Benchmark: cross-sectional median daily return of liquid BSE universe "
          f"(no Sensex index series in the downloaded files)")
    print(f"Target: forward max close return >= +{TARGET_GAIN:.0%} within {FWD_WINDOW} sessions")
    print(f"Base hit-rate: {liquid['hit'].mean():.2%}  (n={len(liquid)})")
    print("=" * 90)

    ranked = rank_factors(liquid)
    print("\n--- RANKED PRECURSOR FACTORS (by lift over base hit-rate) ---\n")
    print(ranked.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    hit_rows = liquid[liquid["hit"] == 1].dropna(subset=["peak_day"])
    print(f"\n--- TIMING: when does the +{TARGET_GAIN:.0%} peak occur, among hits? ---")
    print(hit_rows["peak_day"].value_counts().sort_index().rename("count").to_string())
    print(f"Median session-to-peak: {hit_rows['peak_day'].median():.0f}, "
          f"Mean: {hit_rows['peak_day'].mean():.1f}")

    print("\n" + "=" * 90)
    print("CASE STUDIES: one recent real single-day +7% pop per trading day, with pre-move snapshots")
    print("=" * 90)
    n_days_total = liquid_all["DATE1"].nunique()
    pop_days = liquid_all[liquid_all["ret_1"] >= TARGET_GAIN].groupby("DATE1").size()
    print(f"Trading days in dataset: {n_days_total} | Days with >=1 single-day +{TARGET_GAIN:.0%} "
          f"pop in liquid universe: {pop_days.shape[0]} ({pop_days.shape[0]/n_days_total:.0%} of days)")
    print(f"Avg pops/day (on days it happens): {pop_days.mean():.1f}, Max in a day: {pop_days.max()}")
    cases = case_studies(liquid_all)
    if cases.empty:
        print("No single-day +7% pops found in the liquid universe in this dataset.")
    else:
        pd.set_option("display.width", 220)
        pd.set_option("display.max_columns", 50)
        print(cases.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
