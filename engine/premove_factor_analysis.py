"""
Pre-move factor analysis.

Question: almost every day some NSE stock pops ~7-10%+. What did a broader
set of factors look like 3-10 sessions BEFORE that move, across the whole
dataset -- not just the 6 factors already tested in backtest_momentum.py?

Two outputs:
  1. RANKED FACTOR TABLE - every candidate precursor factor, measured at
     day T, correlated against "hit" = forward max close over the next
     1-10 sessions >= +7% from day T. Ranked by top-quintile lift and
     point-biserial correlation so new candidate signals surface even if
     they weren't in the original Step 4A list.
  2. CASE STUDIES - the most recent actual single-day +7% pops in the
     dataset, with each candidate factor's value 3/5/10 sessions before
     the pop, so the statistics can be sanity-checked against real,
     nameable stocks.

Read-only analysis script; does not change AshishTrade_v4.txt scoring.
"""
import glob
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BHAV_DIR = os.path.join(ROOT, "Sec_bhavdata")
HIST_BHAV_DIR = os.path.join(ROOT, "analysis", "history-data", "Sec_bhavdata")  # ~1yr backfill, already on disk
PR_FILES = (glob.glob(os.path.join(ROOT, "pr_*.csv"))
            + glob.glob(os.path.join(ROOT, "analysis", "history-data", "pr_auto_*.csv")))
SECTOR_FILE = os.path.join(ROOT, "nse_sector_mapping.csv")

TARGET_GAIN = 0.07
FWD_WINDOW = 10
MIN_TURNOVER_LACS = 2000.0
MIN_PRICE = 20.0


# --------------------------------------------------------------- loading
def load_bhav():
    files = glob.glob(os.path.join(BHAV_DIR, "sec_bhavdata_full_*.csv"))
    files += glob.glob(os.path.join(ROOT, "sec_bhavdata_*.csv"))
    files += glob.glob(os.path.join(HIST_BHAV_DIR, "sec_bhavdata_full_*.csv"))
    frames = []
    for f in files:
        df = pd.read_csv(f, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for c in df.select_dtypes(include=["object", "str"]).columns:
        df[c] = df[c].astype(str).str.strip()
    df = df[df["SERIES"] == "EQ"].copy()
    df["DATE1"] = pd.to_datetime(df["DATE1"], format="%d-%b-%Y")
    num_cols = ["PREV_CLOSE", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE",
                "LAST_PRICE", "CLOSE_PRICE", "AVG_PRICE", "TTL_TRD_QNTY",
                "TURNOVER_LACS", "NO_OF_TRADES", "DELIV_QTY", "DELIV_PER"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates(subset=["SYMBOL", "DATE1"])
    return df.sort_values(["SYMBOL", "DATE1"]).reset_index(drop=True)


def load_nifty():
    frames = []
    for f in PR_FILES:
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        df = df[df["Symbol"] == "Nifty 50"][["Date", "Close"]]
        frames.append(df)
    n = pd.concat(frames, ignore_index=True).drop_duplicates("Date")
    n["Date"] = pd.to_datetime(n["Date"])
    return n.sort_values("Date").reset_index(drop=True)


def load_sector_map():
    if not os.path.exists(SECTOR_FILE):
        return pd.DataFrame(columns=["Symbol", "Sector"])
    return pd.read_csv(SECTOR_FILE)


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ------------------------------------------------------- feature pipeline
def build_features(df, nifty):
    g = df.groupby("SYMBOL", group_keys=False)

    for n in (1, 3, 5, 10):
        df[f"ret_{n}"] = g["CLOSE_PRICE"].transform(lambda s, n=n: s.pct_change(n))

    nret = {n: nifty.set_index("Date")["Close"].pct_change(n) for n in (3, 5, 10)}
    for n in (3, 5, 10):
        df[f"rs_{n}"] = df[f"ret_{n}"] - df["DATE1"].map(nret[n])

    df["vol_avg5"] = g["TTL_TRD_QNTY"].transform(lambda s: s.rolling(5).mean())
    df["vol_avg20"] = g["TTL_TRD_QNTY"].transform(lambda s: s.rolling(20).mean())
    df["vol_ratio"] = df["vol_avg5"] / df["vol_avg20"]
    df["rel_volume_today"] = df["TTL_TRD_QNTY"] / df["vol_avg20"]

    df["trades_avg5"] = g["NO_OF_TRADES"].transform(lambda s: s.rolling(5).mean())
    df["trades_avg20"] = g["NO_OF_TRADES"].transform(lambda s: s.rolling(20).mean())
    df["trades_ratio"] = df["trades_avg5"] / df["trades_avg20"]

    df["turn_avg20"] = g["TURNOVER_LACS"].transform(lambda s: s.rolling(20).mean())
    df["turnover_expansion"] = df["TURNOVER_LACS"] / df["turn_avg20"]

    df["deliv_avg5"] = g["DELIV_PER"].transform(lambda s: s.rolling(5).mean())
    df["deliv_avg20"] = g["DELIV_PER"].transform(lambda s: s.rolling(20).mean())
    df["deliv_trend"] = df["deliv_avg5"] - df["deliv_avg20"]
    df["deliv_expansion"] = df["DELIV_PER"] / df["deliv_avg20"]

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

    up_day = (df["ret_1"] > 0).astype(int)
    grp = df["SYMBOL"]
    df["consecutive_up_days"] = (
        up_day.groupby([grp, (up_day != up_day.shift()).groupby(grp).cumsum()]).cumsum() * up_day
    )

    # forward max return over next FWD_WINDOW sessions + which session it peaks on
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
    ("rs_10", "Relative Strength vs Nifty (10d)"),
    ("rs_5", "Relative Strength vs Nifty (5d)"),
    ("rs_3", "Relative Strength vs Nifty (3d)"),
    ("ret_3", "3-Day Momentum"),
    ("ret_5", "5-Day Momentum"),
    ("ret_1", "1-Day Momentum"),
    ("vol_ratio", "Volume Ratio (5d/20d)"),
    ("rel_volume_today", "Relative Volume Today"),
    ("trades_ratio", "No. of Trades Ratio (5d/20d)"),
    ("turnover_expansion", "Turnover Expansion (today/20d avg)"),
    ("deliv_trend", "Delivery %% Trend (5d-20d)"),
    ("deliv_expansion", "Delivery %% Expansion (today/20d avg)"),
    ("range_compression", "Range Compression (VCP-style)"),
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
        # direction-aware: also check bottom quintile in case factor is inverse (e.g. resistance_distance)
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


def case_studies(liquid_all, sector_map, n_cases=15):
    """One real single-day +7% pop per recent trading day, with pre-move factor snapshots
    (spreads across distinct days instead of clustering on one broad-rally session)."""
    pops = liquid_all[liquid_all["ret_1"] >= TARGET_GAIN].copy()
    pops = pops.merge(sector_map[["Symbol", "Sector"]], left_on="SYMBOL", right_on="Symbol", how="left")
    pops = pops.sort_values(["DATE1", "ret_1"], ascending=[False, False])
    pops = pops.groupby("DATE1", as_index=False).first().sort_values("DATE1", ascending=False).head(n_cases)

    df_idx = liquid_all.set_index(["SYMBOL", "DATE1"])
    lookbacks = [3, 5, 10]
    snap_cols = ["rs_10", "vol_ratio", "deliv_trend", "range_compression",
                 "close_strength_avg3", "resistance_distance", "consecutive_up_days"]

    records = []
    for _, row in pops.iterrows():
        sym = row["SYMBOL"]
        sub = liquid_all[liquid_all["SYMBOL"] == sym].sort_values("DATE1").reset_index(drop=True)
        pop_idx = sub.index[sub["DATE1"] == row["DATE1"]]
        if len(pop_idx) == 0:
            continue
        pop_idx = pop_idx[0]
        rec = {"Symbol": sym, "Sector": row.get("Sector", "Unmapped"),
               "PopDate": row["DATE1"].date(), "PopRet1d": row["ret_1"]}
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
    nifty = load_nifty()
    sector_map = load_sector_map()
    bhav = build_features(bhav, nifty)

    liquid_all = bhav[(bhav["TURNOVER_LACS"] >= MIN_TURNOVER_LACS) &
                       (bhav["CLOSE_PRICE"] >= MIN_PRICE)].copy()
    liquid = liquid_all.dropna(subset=["hit"])

    print("=" * 90)
    print("PRE-MOVE FACTOR ANALYSIS")
    print(f"Universe: liquid subset (turnover>=Rs20Cr, price>=Rs20), "
          f"{liquid['SYMBOL'].nunique()} symbols, {liquid['DATE1'].min().date()} - "
          f"{liquid['DATE1'].max().date()}")
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
    cases = case_studies(liquid_all, sector_map)
    if cases.empty:
        print("No single-day +7% pops found in the liquid universe in this dataset.")
    else:
        pd.set_option("display.width", 220)
        pd.set_option("display.max_columns", 50)
        print(cases.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
