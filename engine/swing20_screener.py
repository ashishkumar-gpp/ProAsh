"""
Swing model: ranks NSE/BSE stocks for a ~20% gain within a 1-2 month hold
(21-45 trading sessions), pure buy-and-hold (no stop-loss enforced).

Weights = Cohen's d effect sizes from a full-history backtest ranking (all
246 available NSE+BSE trading days, hit = closing price reaching +20% at any
point within sessions 21-45 after the pick) -- see analysis/pattern_scan.py
for the grid scan that discovered this (target%, horizon) combination has a
real, statistically significant edge in an out-of-sample train/test split
(analysis/pattern_scan_results.csv), unlike the smaller 10-15%/1-1.5mo
objective originally considered (analysis/target_1to1_5mo_10to15pct.py).

Honest caveat: only ~1 year of data (Aug 2025-Aug 2026, a broadly rising
market) backs these weights. The ~1.5-1.7x lift over the ~17-23% base rate
held out-of-sample in the single train/test split, but has NOT yet been
walk-forward tested across multiple independent monthly folds the way
live_screener.py's short-horizon model was -- run analysis/
swing20_walkforward.py to do that before trusting this with real capital.
No stop-loss is enforced -- average drawdown while waiting for the target
historically ran -6% to -9%; size positions accordingly.

Run analysis/download_data.py first (or run_daily_update.bat). Read-only;
does not change AshishTrade_v4.txt, live_screener.py, or any other script.
"""
import os
import sys
import textwrap

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import premove_factor_analysis as nse_mod
import bse_premove_factor_analysis as bse_mod
from pattern_scan import add_extra_momentum, apply_cached_split_adjustments

# Cohen's d from the full-history (246-day) backtest ranking, target=+20% within
# sessions 21-45 (see docstring above / analysis/pattern_scan_results.csv)
WEIGHTS = {
    "rs_40": 0.2782, "rs_20": 0.2265, "ret_40": 0.1600, "rs_10": 0.1593,
    "rs_5": 0.1117, "rs_3": 0.0895, "ret_20": 0.0814, "ret_10": 0.0657,
    "range_compression": 0.0602, "ret_5": 0.0592, "trades_ratio": 0.0581,
    "ret_3": 0.0496, "ret_1": 0.0391, "gap_avg5": 0.0352, "vol_ratio": 0.0306,
    "turnover_expansion": 0.0221, "rel_volume_today": 0.0198,
}
TOP_N_PER_MARKET = 15
TOP_N_FINAL = 5
DAY_MIN, DAY_MAX = 21, 45          # ~1-2 month hold window
TARGET_PCT = 0.20                  # this study's discovered objective
SOFT_STOP_REF_PCT = -0.08          # informational only -- NOT enforced (see caveat above)

# Neither NSE nor BSE bhavcopy tags instrument type (ETFs share SERIES=="EQ" with real
# stocks), so ETFs are excluded by symbol keyword -- commodity/index ETFs track NAV
# momentum, not company fundamentals, and can dominate rankings during a commodity
# mania/crash the model has no way to distinguish from a genuine equity breakout.
# Caveat: this is a heuristic and can false-positive on real tickers that happen to
# contain these words (e.g. GOLDIAM, a jewellery exporter, not a gold ETF).
ETF_KEYWORDS = ("GOLD", "SILV", "SLVR", "NIFTY", "SENSEX", "BEES", "ETF", "LIQUID", "GILT")


def _is_etf(symbol):
    s = symbol.upper()
    return any(kw in s for kw in ETF_KEYWORDS)


# Longest lookback among WEIGHTS (rs_40/ret_40) -- a split/bonus inside this window
# distorts that factor's raw return, so today's Score for that symbol may be unreliable.
SPLIT_CHECK_LOOKBACK = 45
SPLIT_CHECK_THRESH = 0.30


def _recent_split_flag(bhav, symbol, lookback=SPLIT_CHECK_LOOKBACK, thresh=SPLIT_CHECK_THRESH):
    """True if SYMBOL had a single-day |% change| > thresh within the last
    `lookback` sessions -- near-certain sign of an unadjusted split/bonus in
    the bhavcopy data, not a real one-day move, that would distort the
    rs_40/ret_40-style lookback factors feeding today's EdgeScore."""
    hist = bhav[bhav["SYMBOL"] == symbol].sort_values("DATE1").tail(lookback + 1)
    chg = hist["CLOSE_PRICE"].pct_change().abs()
    return bool((chg > thresh).any())


def _edge_score(liquid):
    liquid = liquid.copy()
    for f in WEIGHTS:
        mu, sd = liquid[f].mean(), liquid[f].std()
        liquid[f"_z_{f}"] = (liquid[f] - mu) / sd if sd else 0.0
    liquid["EdgeScore"] = sum(liquid[f"_z_{f}"] * w for f, w in WEIGHTS.items())
    return liquid


def _market_regime_label(nifty, latest_date):
    """Informational only -- NEVER filters/blocks picks. Nifty 20-session return's
    historical percentile as of latest_date, the same signal analysis/
    regime_and_stoploss_test.py already tested as a HARD entry gate for this exact
    20%/21-45-session model: it did NOT hold up (TEST p=0.59, walk-forward p=0.12), so
    it is surfaced here for trader awareness only, not used to suppress any pick."""
    ret20 = nifty.set_index("Date")["Close"].pct_change(20).dropna()
    if latest_date not in ret20.index or len(ret20) < 20:
        return "unknown (insufficient history)"
    pct_rank = (ret20 <= ret20.loc[latest_date]).mean()
    if pct_rank >= 0.8:
        return f"Bullish -- Nifty 20d-return in top quintile ({pct_rank:.0%}ile)"
    if pct_rank <= 0.2:
        return f"Weak/Caution -- Nifty 20d-return in bottom quintile ({pct_rank:.0%}ile)"
    return f"Neutral -- Nifty 20d-return {pct_rank:.0%}ile"


def _earnings_alert(corp_action_text):
    """Derived from the CorpAction text already fetched for the pick (no extra network
    call) -- flags an upcoming/recent board meeting, which in India usually means a
    results announcement that can gap the stock outside the model's pure momentum read."""
    return "Board meeting/results nearby" if "board meeting" in corp_action_text.lower() else "-"


CORR_LOOKBACK = 20   # trading sessions of daily returns used for the diversification check
CORR_THRESH = 0.85   # near-duplicate movers above this correlation aren't both kept


def _select_diversified(all_rows, bhav_n, bhav_b, n_final):
    """Fills n_final slots by RawScore, skipping a candidate whose trailing
    CORR_LOOKBACK-session daily-return correlation with an already-selected pick is
    >= CORR_THRESH (near-duplicate movers -- e.g. group companies/cross-listings/the
    same stock on both exchanges -- add concentration, not diversification). No sector
    mapping data exists in this repo (see repo memory), so this price-correlation proxy
    is used instead of a true GICS-sector check.
    Never under-fills: any candidate skipped for correlation is backfilled at the end if
    fewer than n_final were otherwise selected, so a strict filter can never shrink the
    picks below what the model would show without it (as long as candidates exist)."""
    def ret_series(market, symbol):
        bhav = bhav_n if market == "NSE" else bhav_b
        hist = bhav[bhav["SYMBOL"] == symbol].sort_values("DATE1").tail(CORR_LOOKBACK)
        return hist.set_index("DATE1")["ret_1"]

    selected, skipped = [], []
    for r in all_rows:
        if len(selected) >= n_final:
            break
        s = ret_series(r["Market"], r["Symbol"])
        too_correlated = False
        for sel in selected:
            joined = pd.concat([s, ret_series(sel["Market"], sel["Symbol"])], axis=1, join="inner")
            if len(joined) >= 10 and joined.iloc[:, 0].corr(joined.iloc[:, 1]) >= CORR_THRESH:
                too_correlated = True
                break
        (skipped if too_correlated else selected).append(r)

    for r in skipped:
        if len(selected) >= n_final:
            break
        selected.append(r)
    return selected[:n_final]


def screen(bhav, turnover_col, min_turnover, min_price, market):
    liquid = bhav[(bhav[turnover_col] >= min_turnover) & (bhav["CLOSE_PRICE"] >= min_price)].copy()
    liquid = liquid[~liquid["SYMBOL"].apply(_is_etf)]
    liquid = _edge_score(liquid)

    latest_date = liquid["DATE1"].max()
    today = liquid[liquid["DATE1"] == latest_date].dropna(subset=list(WEIGHTS) + ["EdgeScore"]).copy()
    # A/B tested (Jan-Mar 2026): stricter EdgeScore floor + volume-confirmation floor
    # showed no benefit (37.8% vs 39.7% hit rate) -- reverted to plain EdgeScore > 0.
    today = today[today["EdgeScore"] > 0].sort_values("EdgeScore", ascending=False).head(TOP_N_PER_MARKET)

    rows = []
    for _, r in today.iterrows():
        entry = r["CLOSE_PRICE"]
        rows.append({
            "Market": market, "Symbol": r["SYMBOL"], "CMP": entry, "Entry": entry,
            "Target(+20%)": round(entry * (1 + TARGET_PCT), 2),
            "SoftStopRef(-8%)": round(entry * (1 + SOFT_STOP_REF_PCT), 2),
            "RawScore": round(r["EdgeScore"], 3),
        })
    return latest_date, len(liquid[liquid["DATE1"] == latest_date]), rows


def render_table(headers, rows, right_align_from=3, wrap_widths=None):
    """Thick '=' box borders around the header AND after every single row, so
    each stock reads as its own boxed entry (numeric columns right-aligned).
    wrap_widths: optional {header: max_width} to word-wrap long text columns
    (e.g. CorpAction/RecentNews) onto extra lines within the same boxed row,
    instead of relying on the terminal to wrap, which breaks box alignment."""
    wrap_widths = wrap_widths or {}
    widths = []
    for i, h in enumerate(headers):
        natural = max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
        cap = wrap_widths.get(h)
        widths.append(min(natural, cap) if cap else natural)
    sep = "+=" + "=+=".join("=" * w for w in widths) + "=+"

    def fmt(vals):
        col_lines = []
        for i, v in enumerate(vals):
            cap = wrap_widths.get(headers[i])
            col_lines.append(textwrap.wrap(str(v), width=widths[i]) if cap else [str(v)])
        height = max(len(c) for c in col_lines)
        rows_out = []
        for line_no in range(height):
            cells = []
            for i, col in enumerate(col_lines):
                piece = col[line_no] if line_no < len(col) else ""
                right = i >= right_align_from and headers[i] not in wrap_widths
                cells.append(piece.rjust(widths[i]) if right else piece.ljust(widths[i]))
            rows_out.append("| " + " | ".join(cells) + " |")
        return rows_out

    lines = [sep, *fmt(headers), sep]
    for r in rows:
        lines.extend(fmt(r))
        lines.append(sep)
    return "\n".join(lines)


def get_top_picks(lookup_extras=True):
    """Computes today's Stage-1 picks and returns (top, date_n, n_liq_n, date_b, n_liq_b,
    regime_label). top is a list of dicts (Market/Symbol/CMP/Entry/Target/SoftStopRef/
    RawScore/Score, plus SplitAlert/CorpAction/RecentNews/EarningsAlert if
    lookup_extras=True) -- same data main() prints, but importable so other scripts (e.g.
    the fundamental overlay model / combined pipeline) can consume it directly instead of
    parsing console output. regime_label is informational only -- see _market_regime_label."""
    nifty = nse_mod.load_nifty()
    bhav_n = nse_mod.build_features(nse_mod.load_bhav(), nifty)
    bhav_n = apply_cached_split_adjustments(bhav_n)
    bhav_n = add_extra_momentum(bhav_n, nifty.set_index("Date")["Close"])
    date_n, n_liq_n, rows_n = screen(bhav_n, "TURNOVER_LACS", nse_mod.MIN_TURNOVER_LACS, nse_mod.MIN_PRICE, "NSE")

    bhav_b = bse_mod.build_features(bse_mod.load_bhav())
    bhav_b = apply_cached_split_adjustments(bhav_b)
    bse_bench = (1 + bhav_b.groupby("DATE1")["ret_1"].median()).cumprod()
    bhav_b = add_extra_momentum(bhav_b, bse_bench)
    date_b, n_liq_b, rows_b = screen(bhav_b, "TURNOVER", bse_mod.MIN_TURNOVER, bse_mod.MIN_PRICE, "BSE")

    regime_label = _market_regime_label(nifty, max(date_n, date_b))

    all_rows = sorted(rows_n + rows_b, key=lambda r: r["RawScore"], reverse=True)
    top = _select_diversified(all_rows, bhav_n, bhav_b, TOP_N_FINAL)
    if not top:
        return top, date_n, n_liq_n, date_b, n_liq_b, regime_label

    # Score (0-100): 100 = today's single strongest combined-factor reading, 0 = the
    # historical average stock. Relative ranking scale, NOT a win probability.
    best = top[0]["RawScore"]
    for r in top:
        r["Score"] = round(100 * r["RawScore"] / best, 1)

    if lookup_extras:
        # Only looked up for the final shortlist (not the whole universe) to keep the NSE
        # API calls few; a lookup failure never blocks the picks from printing (see module doc).
        from corporate_actions import get_corporate_actions
        from news_check import get_recent_news
        for r in top:
            bhav_for_market = bhav_n if r["Market"] == "NSE" else bhav_b
            r["SplitAlert"] = "SPLIT? verify chart" if _recent_split_flag(bhav_for_market, r["Symbol"]) else "-"
            r["CorpAction"] = get_corporate_actions(r["Symbol"])
            r["RecentNews"] = get_recent_news(r["Symbol"])
            r["EarningsAlert"] = _earnings_alert(r["CorpAction"])

    return top, date_n, n_liq_n, date_b, n_liq_b, regime_label


def main():
    top, date_n, n_liq_n, date_b, n_liq_b, regime_label = get_top_picks()

    print(f"\nNSE latest trading day: {date_n.date()} ({n_liq_n} liquid symbols)")
    print(f"BSE latest trading day: {date_b.date()} ({n_liq_b} liquid symbols)")
    print(f"Market regime (informational only, does not filter picks): {regime_label}")

    if not top:
        print("\nNo candidates with a positive EdgeScore today.")
        return

    headers = ["Market", "Symbol", "CMP", "Entry", "Target(+20%)", "SoftStopRef(-8%)", "Score",
               "SplitAlert", "EarningsAlert", "CorpAction", "RecentNews"]
    rows_fmt = [[r[h] for h in headers] for r in top]
    print(f"\nTop {len(top)} picks (diversified -- see DIVERSIFICATION note below):\n")
    print("Score (0-100): 100 = today's single strongest combined-factor reading, 0 = the")
    print("historical average stock. It is a relative ranking scale, NOT a win probability.")
    print(f"SplitAlert: local check (no network) for a single-day price move > {SPLIT_CHECK_THRESH:.0%} in the")
    print(f"last {SPLIT_CHECK_LOOKBACK} sessions -- likely an unadjusted split/bonus that may have distorted")
    print("this pick's momentum factors and Score. '-' = none detected.")
    print("EarningsAlert: flags a nearby NSE board meeting (derived from CorpAction, no extra")
    print("lookup) -- board meetings in India usually mean a results announcement, which can")
    print("gap the stock outside pure momentum. '-' = none detected.")
    print("CorpAction: any NSE corporate action within +/-45 days of today ('-' = none found,")
    print("'unknown' = lookup failed) -- check it manually before trusting the pick.")
    print("RecentNews: latest Google News headlines, no sentiment scoring -- read them yourself,")
    print("'-' = none found, 'unknown' = lookup failed.")
    print(f"DIVERSIFICATION: candidates with >= {CORR_THRESH:.0%} trailing {CORR_LOOKBACK}-session")
    print("return correlation to an already-picked stock (e.g. group companies/cross-listings) are")
    print("skipped in favor of the next-best candidate -- never shrinks the pick count below what's")
    print("available.\n")
    wrap_widths = {"CorpAction": 40, "RecentNews": 50}
    print(render_table(headers, rows_fmt, wrap_widths=wrap_widths))

    from paper_trade_prompt import offer_paper_trade
    offer_paper_trade(top)


if __name__ == "__main__":
    main()
