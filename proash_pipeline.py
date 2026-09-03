"""
ProAsh -- standalone live ensemble pipeline.

Combines the SAME 3 independent models as C:\\Mine\\Compare\\ensemble_model.py
(all three already target the shared objective: +12% within 45 trading
sessions of the signal day), but computes everything FRESH from raw NSE+BSE
bhavcopy on every run instead of reading pre-built parquet label files --
so this project works standalone even if AshFund/AshishTrade are never
touched again:
  1. AshFund's raise_factor_model.pkl (P(+12% in 45 sessions)) x
     risk_factor_model.pkl (P(failure)) -> CombinedScore, top-5/market flag
  2. AshishTrade's mined pattern rules (data/patterns.json) + frozen
     calibration table (data/calibration.json) -> CalibratedWinRate flag
  3. pro_screener-style relative-strength percentile leadership (top 1% of
     today's combined NSE+BSE universe by rs_20) flag

"Qualified" = at least 1 of the 3 flags fires (votes>=1) -- guarantees this
always shows something on a normal trading day. Of the qualified names, the
top 3 by blended confidence (ensemble_score) are the week's picks, after a
correlation-based diversification pass (skip near-duplicate movers).

Stage 2 (best-effort, never blocks the picks): CorpAction/M&A/RecentNews
lookups on the final shortlist only, chandelier trailing stop + suggested
position size, then an interactive "add to paper trading?" prompt.

Usage: run ProAsh.bat, or `python proash_pipeline.py` directly.
"""
import json
import os
import pickle
import re
import socket
import sys
import threading

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
ENGINE_DIR = os.path.join(ROOT, "engine")
sys.path.insert(0, ENGINE_DIR)

import premove_factor_analysis as nse_mod  # noqa: E402
import bse_premove_factor_analysis as bse_mod  # noqa: E402
import swing20_screener as s20  # noqa: E402
import pattern_scan as ps  # noqa: E402
from corporate_actions import get_corporate_actions, get_ma_open_offer_alert, CHART_RISK_KEYWORDS, _session as _nse_session  # noqa: E402
from news_check import get_recent_news  # noqa: E402
from stop_loss import load_continuous_hist, get_stop_loss  # noqa: E402
from candlestick_patterns import accumulation_flag_for_symbol, volume_ratio_for_symbol, sustained_pump_flag  # noqa: E402

sys.path.insert(0, ROOT)
from paper_trade_prompt import offer_paper_trade  # noqa: E402

LAG_SOURCE_COLS = ["ret_5", "ret_10", "rs_5", "rs_10", "vol_ratio",
                    "turnover_expansion", "range_compression", "rsi14"]
LAGS = (5, 10, 15)

AF_TOPN_PER_DAY = 5        # AshFund flag = today's top-5 CombinedScore per market
AT_MIN_WINRATE = 0.65      # AshishTrade flag threshold on CalibratedWinRate
RS_TOP_PCTILE = 0.99       # pro_screener's own validated top-1% RS tier
QUALIFY_MIN_VOTES = 1      # "qualified" = at least 1 of 3 models flags it (guarantees output)
SHORTLIST_N = 30           # how many candidates to show in the Stage 1 shortlist
SHORTLIST_DISPLAY_N = 7    # how many of the Stage 1 shortlist to print on console (display-only cap)
FINAL_TOP_N = 30           # how many candidates get full Stage 2 analysis (corp actions/news/accum/R:R)
DISPLAY_TOP_N = 5          # how many QUALIFIED (BUY) picks to actually show/offer on console
TARGET_PCT = 0.12
RISK_PER_TRADE = 2000      # Rs risked per trade for suggested position sizing (informational only)

# Backtested hit-rate (+12% within 45 sessions) conditioned on vote count, from
# C:\Mine\Compare\ensemble_model.py --backtest (2025-01-01 to 2026-06-18, all flagged
# days -- not just weekly top-3, so sample sizes are as large as possible per tier).
# Replaces the old blended ensemble_score "Confidence %", which could read high (e.g. 72%)
# even on a single-vote pick purely because the RS percentile component sits near its own
# 0.99+ ceiling whenever it's the one flag that fired -- not because 3 signals agreed.
HIST_WINRATE_BY_VOTES = {
    0: (42.8, 69203),
    1: (68.1, 4376),
    2: (77.5, 80),
    3: (None, 1),   # only ever observed once in the backtest window -- not statistically usable
}


def _hist_winrate_label(votes):
    wr, n = HIST_WINRATE_BY_VOTES.get(votes, (None, 0))
    if wr is None:
        return f"n={n} (unreliable)"
    return f"{wr}% (n={n:,})"


_ACCUM_RANK = {"STRONG": 0, "NEUTRAL": 1, "-": 2, "CAUTION": 3}


def _accum_rank(flag):
    for key, rank in _ACCUM_RANK.items():
        if flag.startswith(key):
            return rank
    return 2


def _rr_tier(rr):
    """Higher R:R is NOT better -- it only rises because the stop gets tighter (target is
    fixed at +12%), and a tighter stop gets clipped by ordinary noise far more often than
    the bigger payoff compensates for. Stop-loss-aware backtest (analysis/backtest_rr_bucket.py,
    n=7913): EV(x risk/trade) by R:R bucket -- <0.8: +0.168, 0.8-1.2: +0.214/+0.109 (best),
    1.2-1.5: -0.059 (proven NEGATIVE EV), >=1.5: +0.072 (thin, n=171). Tiers below rank the
    empirically best bucket first."""
    if rr is None:
        return 4
    if 0.8 <= rr < 1.2:
        return 0
    if rr < 0.8:
        return 1
    if rr >= 1.5:
        return 2
    return 3   # 1.2 <= rr < 1.5 -- worst, proven negative EV


_COND_RE = re.compile(r"^(\S+)\s*(>=|<=|==)\s*([\-0-9.eE]+)$")


def _cond_mask(d, desc):
    m = _COND_RE.match(desc.strip())
    col, op, thresh = m.group(1), m.group(2), float(m.group(3))
    s = d[col]
    if op == ">=":
        return (s >= thresh).to_numpy()
    if op == "<=":
        return (s <= thresh).to_numpy()
    return (s == thresh).to_numpy()


def _rule_mask(d, rule):
    mask = np.ones(len(d), dtype=bool)
    for cond in rule["conditions"]:
        mask &= _cond_mask(d, cond)
    return mask


def _lookup_calibrated(count, table):
    by_count = {r["count"]: r["win_rate"] for r in table}
    if count in by_count:
        return by_count[count]
    lower = [c for c in by_count if c <= count]
    return by_count[max(lower)] if lower else by_count[min(by_count)]


def load_models():
    with open(os.path.join(DATA_DIR, "raise_factor_model.pkl"), "rb") as f:
        raise_bundle = pickle.load(f)
    with open(os.path.join(DATA_DIR, "risk_factor_model.pkl"), "rb") as f:
        risk_bundle = pickle.load(f)
    with open(os.path.join(DATA_DIR, "patterns.json")) as f:
        patterns = json.load(f)
    with open(os.path.join(DATA_DIR, "calibration.json")) as f:
        calib = json.load(f)
    return raise_bundle, risk_bundle, patterns, calib


def build_market(market):
    """Same feature build AshFund's run_pipeline.py uses, so the frozen raise/
    risk models see exactly the feature distribution they were trained on."""
    if market == "NSE":
        nifty = nse_mod.load_nifty()
        bhav = nse_mod.build_features(nse_mod.load_bhav(), nifty)
        bhav = ps.apply_cached_split_adjustments(bhav)
        bhav = ps.add_extra_momentum(bhav, nifty.set_index("Date")["Close"])
        turnover_col, min_turnover, min_price = "TURNOVER_LACS", nse_mod.MIN_TURNOVER_LACS, nse_mod.MIN_PRICE
    else:
        bhav = bse_mod.build_features(bse_mod.load_bhav())
        bhav = ps.apply_cached_split_adjustments(bhav)
        bench = (1 + bhav.groupby("DATE1")["ret_1"].median()).cumprod()
        bhav = ps.add_extra_momentum(bhav, bench)
        turnover_col, min_turnover, min_price = "TURNOVER", bse_mod.MIN_TURNOVER, bse_mod.MIN_PRICE

    bhav = bhav[(bhav[turnover_col] >= min_turnover) & (bhav["CLOSE_PRICE"] >= min_price)].copy()
    bhav = bhav[~bhav["SYMBOL"].apply(s20._is_etf)]
    bhav = bhav.sort_values(["SYMBOL", "DATE1"]).reset_index(drop=True)

    g = bhav.groupby("SYMBOL", sort=False)
    for col in LAG_SOURCE_COLS:
        for lag in LAGS:
            bhav[f"{col}_lag{lag}"] = g[col].shift(lag)

    bhav["Market"] = market
    return bhav


def score_latest(bhav, raise_bundle, risk_bundle, patterns, min_valid_rows=20):
    """Scores the most recent trading day with enough valid feature rows,
    falling back to an earlier day if today's rows are mostly NaN (e.g. a
    stale/not-yet-refreshed benchmark file) instead of silently returning zero
    candidates."""
    need_cols = list(set(raise_bundle["features"]) | set(risk_bundle["features"]))
    true_latest = bhav["DATE1"].max()
    for candidate_date in sorted(bhav["DATE1"].unique(), reverse=True):
        today = bhav[bhav["DATE1"] == candidate_date].copy()
        valid = today.dropna(subset=need_cols)
        if len(valid) >= min_valid_rows:
            if candidate_date != true_latest:
                print(f"  [warning] Latest bhavcopy date {pd.Timestamp(true_latest).date()} has insufficient "
                      f"valid feature rows -- using {pd.Timestamp(candidate_date).date()} instead.")
            valid["RaiseScore"] = raise_bundle["model"].predict_proba(valid[raise_bundle["features"]])[:, 1]
            valid["RiskScore"] = risk_bundle["model"].predict_proba(valid[risk_bundle["features"]])[:, 1]
            valid["CombinedScore"] = valid["RaiseScore"] * (1 - valid["RiskScore"])

            succ = np.zeros(len(valid), dtype=int)
            for rule in patterns["success_rules"]:
                succ += _rule_mask(valid, rule).astype(int)
            fail = np.zeros(len(valid), dtype=int)
            for rule in patterns["failure_rules"]:
                fail += _rule_mask(valid, rule).astype(int)
            valid["success_rule_count"] = succ
            valid["failure_rule_count"] = fail
            valid["momentum_alive"] = (valid["ret_3"] >= 0.0) & (valid["ret_5"] >= -0.02)
            valid["Dual"] = (valid["success_rule_count"] >= 1) & (valid["failure_rule_count"] == 0)
            return valid, pd.Timestamp(candidate_date)

    na_counts = bhav[bhav["DATE1"] == true_latest][need_cols].isna().sum().sort_values(ascending=False)
    print(f"  [diagnostic] No date found with >= {min_valid_rows} valid rows; "
          f"top NaN columns on {pd.Timestamp(true_latest).date()}:\n{na_counts.head(8)}")
    return bhav.iloc[0:0], pd.Timestamp(true_latest)


def _weekday_lag(d):
    from datetime import date, timedelta
    d = d.date() if hasattr(d, "date") else d
    ref = date.today()
    if d >= ref:
        return 0
    lag, cur = 0, d + timedelta(days=1)
    while cur <= ref:
        if cur.weekday() < 5:
            lag += 1
        cur += timedelta(days=1)
    return lag


def combined_verdict(votes, ma_alert, split_alert, earnings_alert):
    if ma_alert != "-":
        return "WAIT (M&A/open-offer -- not organic momentum)"
    if split_alert != "-":
        return "WAIT (verify chart -- recent split/bonus/dividend nearby)"
    if earnings_alert != "-":
        return "WAIT (results nearby)"
    return f"BUY ({votes}/3 models agree)"


def _earnings_alert(corp_action_text):
    return "Board meeting/results nearby" if "board meeting" in corp_action_text.lower() else "-"


ROUTINE_MA_OVERRIDE_PHRASES = ("no encumbrance", "no new encumbrance", "no new share encumbrance",
                                "nil encumbrance", "no pledge")


def _is_routine_ma_disclosure(ma_alert, news):
    """NSE's corporate-announcements 'desc' is often just a generic regulation-name label
    (e.g. "Disclosure under SEBI Takeover Regulations") that alone can't tell a real open-offer/
    acquisition apart from a routine annual promoter encumbrance disclosure -- but the news feed
    (a different, already-fetched source) usually spells out which it is. Verified false positive:
    OMAXE's 24-Jul-2026 filing was "promoter declares no new share encumbrances", not a takeover."""
    if ma_alert in ("-", "unknown", "unknown (no network)"):
        return False
    return any(p in (news or "").lower() for p in ROUTINE_MA_OVERRIDE_PHRASES)


def _is_chart_risk_action(corp_action_text):
    return any(k in corp_action_text.lower() for k in CHART_RISK_KEYWORDS)


def _events_display(corp_action, ma_alert):
    parts = [p for p in (str(corp_action or "-"), str(ma_alert or "-")) if p != "-"]
    return "; ".join(parts) if parts else "-"


def _network_reachable(host="www.nseindia.com", port=443, timeout=5):
    """One fast probe instead of letting every Stage-2 lookup (2-3 requests each,
    15s timeout apiece) time out independently -- on a blocked/corporate network
    that's the difference between a few seconds and several minutes for 3 stocks.
    Runs in a daemon thread because DNS lookup (getaddrinfo) isn't bounded by socket
    timeouts and can hang indefinitely on a broken/proxied network -- a daemon thread
    (unlike ThreadPoolExecutor) never blocks process exit even if it never returns."""
    result = {"ok": False}

    def _probe():
        try:
            socket.create_connection((host, port), timeout=timeout).close()
            result["ok"] = True
        except Exception:
            pass

    t = threading.Thread(target=_probe, daemon=True)
    t.start()
    t.join(timeout=timeout + 2)
    return result["ok"]


def main():
    calib_table = None
    raise_bundle, risk_bundle, patterns, calib = load_models()
    calib_table = calib["success_rule_count_table"]

    bhav_n = build_market("NSE")
    bhav_b = build_market("BSE")
    print(f"Latest local NSE bhav date: {pd.Timestamp(bhav_n['DATE1'].max()).date()}  "
          f"(lag {_weekday_lag(bhav_n['DATE1'].max())} weekday(s))")
    print(f"Latest local BSE bhav date: {pd.Timestamp(bhav_b['DATE1'].max()).date()}  "
          f"(lag {_weekday_lag(bhav_b['DATE1'].max())} weekday(s))")

    today_n, date_n = score_latest(bhav_n, raise_bundle, risk_bundle, patterns)
    today_b, date_b = score_latest(bhav_b, raise_bundle, risk_bundle, patterns)

    frames = []
    for today, market in ((today_n, "NSE"), (today_b, "BSE")):
        if today.empty:
            print(f"No scorable {market} candidates today -- skipping.")
            continue
        today = today.copy()
        today["Market"] = market
        today["af_rank"] = today["CombinedScore"].rank(ascending=False, method="first")
        today["CalibratedWinRate"] = today["success_rule_count"].apply(lambda c: _lookup_calibrated(c, calib_table))
        frames.append(today)

    if not frames:
        print("\nNo scorable candidates on either exchange today -- no picks.")
        return
    universe = pd.concat(frames, ignore_index=True)
    universe["rs_20_pctile"] = universe["rs_20"].rank(pct=True)

    universe["flagA_ashfund"] = universe["af_rank"] <= AF_TOPN_PER_DAY
    universe["flagB_ashishtrade"] = universe["Dual"] & universe["momentum_alive"] & (universe["CalibratedWinRate"] >= AT_MIN_WINRATE)
    universe["flagC_proscreener"] = universe["rs_20_pctile"] >= RS_TOP_PCTILE
    universe["votes"] = (universe["flagA_ashfund"].astype(int) + universe["flagB_ashishtrade"].astype(int)
                          + universe["flagC_proscreener"].astype(int))
    universe["ensemble_flag"] = universe["votes"] >= QUALIFY_MIN_VOTES
    universe["ensemble_score"] = (universe["CombinedScore"].fillna(0) + universe["CalibratedWinRate"].fillna(0)
                                   + universe["rs_20_pctile"].fillna(0)) / 3

    qualified = universe[universe["ensemble_flag"]].sort_values("ensemble_score", ascending=False)
    print(f"\n{len(qualified)} of {len(universe)} scored candidates qualify (votes>=1) today.")
    if qualified.empty:
        print("(no stock currently has any model support -- no pick today)")
        return

    shortlist = qualified.head(SHORTLIST_N)
    shortlist_rows = [{"Market": r["Market"], "Symbol": r["SYMBOL"], "CMP": r["CLOSE_PRICE"],
                        "votes": int(r["votes"]), "ensemble_score": r["ensemble_score"],
                        "RaiseScore": r["RaiseScore"], "CombinedScore": r["CombinedScore"],
                        "CalibratedWinRate": r["CalibratedWinRate"], "rs_20_pctile": r["rs_20_pctile"]}
                       for _, r in shortlist.iterrows()]

    print(f"\n=== Today's Top {SHORTLIST_DISPLAY_N} Shortlist (of {len(shortlist_rows)} analyzed, ranked by internal score) ===")
    headers1 = ["Rank", "Mkt", "Symbol", "CMP", "Votes", "HistWinRate(votes)"]
    rows1 = [[i + 1, r["Market"], r["Symbol"], round(r["CMP"], 2), f"{r['votes']}/3 models agree",
              _hist_winrate_label(r["votes"])]
             for i, r in enumerate(shortlist_rows[:SHORTLIST_DISPLAY_N])]
    print(s20.render_table(headers1, rows1))
    print("(Votes = how many of the 3 independent signals flagged this stock. HistWinRate = backtested "
          "% of stocks with that exact vote count that hit +12% within 45 sessions -- not a per-stock "
          "probability, and the 2/3 and 3/3 tiers have too few historical samples to trust closely.)")

    # Top-3 by ensemble_score, exactly as validated in the backtest (ensemble_model.py's
    # live_picks()/run_summary_tables() -- no correlation/diversification filter there).
    final = shortlist_rows[:FINAL_TOP_N]

    stop_hist = {"NSE": load_continuous_hist("NSE"), "BSE": load_continuous_hist("BSE")}
    score_date_by_market = {"NSE": date_n, "BSE": date_b}
    network_ok = _network_reachable()
    # one shared session for the whole Stage-2 loop -- avoids a fresh NSE homepage
    # cookie visit per candidate (get_corporate_actions/get_ma_open_offer_alert used to
    # each open their own session, doubling homepage round-trips for every symbol).
    # Falls back to None (each call opens its own session, as before) if this fails.
    try:
        nse_session = _nse_session() if network_ok else None
    except Exception:
        nse_session = None
    if not network_ok:
        print("\n[warning] nseindia.com unreachable -- Stage 2 (corp actions/M&A/news) skipped for this run, "
              "picks below are Stage-1-only (still the same votes/ensemble_score as the backtest).")

    bhav_by_market = {"NSE": bhav_n, "BSE": bhav_b}
    combined_rows = []
    for r in final:
        entry = r["CMP"]
        stop, stop_pct = get_stop_loss(stop_hist[r["Market"]], r["Symbol"], score_date_by_market[r["Market"]], entry)
        risk_per_share = round(entry - stop, 2) if stop is not None else None
        qty = int(RISK_PER_TRADE // risk_per_share) if risk_per_share and risk_per_share > 0 else None
        rr = round(TARGET_PCT * 100 / stop_pct, 2) if stop_pct else None

        sym_hist = bhav_by_market[r["Market"]]
        sym_hist = sym_hist[sym_hist["SYMBOL"] == r["Symbol"]].sort_values("DATE1")
        deliv_col = "DELIV_PER" if r["Market"] == "NSE" else None
        accum_flag = accumulation_flag_for_symbol(sym_hist, deliv_col=deliv_col)
        vol_ratio_today = volume_ratio_for_symbol(sym_hist)
        sustained_pump = sustained_pump_flag(sym_hist, deliv_col=deliv_col)

        if network_ok:
            corp_action = get_corporate_actions(r["Symbol"], session=nse_session)
            ma_alert = get_ma_open_offer_alert(r["Symbol"], session=nse_session)
            news = get_recent_news(r["Symbol"])
        else:
            corp_action = ma_alert = news = "unknown (no network)"
        # split_alert only fires for genuine bonus/split/dividend/rights/buyback hits --
        # a board-meeting-only corp_action must fall through to earnings_alert instead,
        # not get mislabeled as "verify chart" (still passes through on a network failure).
        split_alert = corp_action if (corp_action == "unknown (no network)" or _is_chart_risk_action(corp_action)) else "-"
        earnings_alert = _earnings_alert(corp_action if corp_action not in ("-", "unknown (no network)") else "")
        # ma_alert text is still shown in Events for transparency -- but a routine encumbrance
        # disclosure (confirmed via the separately-fetched news feed) shouldn't block the verdict.
        ma_alert_for_verdict = "-" if _is_routine_ma_disclosure(ma_alert, news) else ma_alert
        verdict = combined_verdict(r["votes"], ma_alert_for_verdict, split_alert, earnings_alert)

        row = {
            "Market": r["Market"], "Symbol": r["Symbol"], "CMP": entry,
            "Target": round(entry * (1 + TARGET_PCT), 2), "StopLoss": stop, "StopLossPct": stop_pct,
            "RiskReward": rr, "Qty": qty, "Votes": r["votes"], "EnsembleScore": round(r["ensemble_score"], 3),
            "Events": _events_display(corp_action, ma_alert), "RecentNews": news, "Verdict": verdict,
            "AccumFlag": accum_flag, "VolRatioToday": vol_ratio_today, "SustainedPump": sustained_pump,
        }
        combined_rows.append(row)

    # scan a wide pool (FINAL_TOP_N) so a bad day for the raw top-3 doesn't leave zero
    # actionable picks. All FINAL_TOP_N candidates that qualify as BUY are equally eligible --
    # this sort only decides which of them is displayed/offered first, it does NOT change
    # who qualifies. Priority: SustainedPump veto, then Accumulation quality, then R:R TIER
    # (NOT raw magnitude -- higher R:R only means a tighter stop, and analysis/backtest_rr_bucket.py
    # found tighter stops get clipped by noise faster than the bigger payoff compensates for --
    # the 1.2-1.5 R:R bucket has NEGATIVE expected value, while 0.8-1.2 is the empirical best),
    # then Votes, with VolRatio only as the last tiebreaker. Votes ranks below R:R tier/Accumulation --
    # analysis/backtest_stop_vs_target.py (stop-loss-ordering-aware) found votes=2 does NOT
    # empirically beat votes=1 once the stop-loss is accounted for (54.3% vs 56.9% target-before-
    # stop, n=116 too thin to trust as worse either way), so raw vote count alone isn't a strong
    # enough signal to outrank a materially better R:R tier/Accumulation reading.
    # (backtested: analysis/backtest_delivery_volume.py, analysis/backtest_sustained_pump_filter.py,
    # real case BAJAJHIND 02-Sep-2026 ranking as falsely "calmest" mid multi-week pump).
    buy_rows_all = [r for r in combined_rows if r["Verdict"].startswith("BUY")]
    buy_rows_all.sort(key=lambda r: (r["SustainedPump"],
                                      _accum_rank(r["AccumFlag"]),
                                      _rr_tier(r["RiskReward"]),
                                      -r["Votes"],
                                      r["VolRatioToday"] if r["VolRatioToday"] is not None else float("inf")))
    buy_rows = buy_rows_all[:DISPLAY_TOP_N]
    paper_trade_candidates = [{"Market": r["Market"], "Symbol": r["Symbol"], "CMP": r["CMP"], "StopLoss": r["StopLoss"]}
                               for r in buy_rows]

    print(f"\n=== FINAL {len(buy_rows)} QUALIFIED BUY PICKS (top {DISPLAY_TOP_N} of {len(buy_rows_all)} that "
          f"cleared BUY out of the top {FINAL_TOP_N} analyzed by internal score, ranked by accumulation/R:R-tier/votes) ===")
    if not buy_rows:
        print(f"(none of the top {FINAL_TOP_N} analyzed candidates cleared the BUY verdict today -- no picks)")
    headers2 = ["Rank", "Mkt", "Symbol", "CMP", f"Target(+{int(TARGET_PCT*100)}%)", "StopLoss", "R:R",
                f"Qty(Rs{RISK_PER_TRADE} risk)", "Votes", "HistWinRate(votes)", "Accumulation", "Action"]
    rows2 = [[i + 1, r["Market"], r["Symbol"], r["CMP"], r["Target"],
              f"{r['StopLoss']} (-{r['StopLossPct']}%)" if r["StopLoss"] else "-",
              f"{r['RiskReward']}:1" if r["RiskReward"] else "-", r["Qty"] or "-",
              f"{r['Votes']}/3", _hist_winrate_label(r["Votes"]),
              r["AccumFlag"], "BUY"]
             for i, r in enumerate(buy_rows)]
    if buy_rows:
        print(s20.render_table(headers2, rows2))
    print("(Accumulation = informational only, does NOT affect BUY/WAIT. Only shown when today's volume is "
          ">=1.4x its 20-session baseline: STRONG = delivery% >50% (real buying), CAUTION = delivery% <30% "
          "(likely speculative/intraday volume, not genuine accumulation), NEUTRAL = 30-50%. '-' = volume not "
          "elevated today, or BSE (no delivery% data) -- backtested on NSE only. Also used as a sort tier: "
          "STRONG < NEUTRAL < '-' < CAUTION.)")
    print("(A pick mid a sustained multi-week pump -- 10 sessions before today averaging delivery% <20% -- is "
          "auto-sorted to the bottom of this list and excluded whenever a cleaner candidate is available, NSE only.)")
    print("(R:R here is NOT sorted highest-first -- a higher R:R only means a tighter stop (target is fixed at "
          "+12%), and a tighter stop gets clipped by ordinary price noise far more often than the bigger payoff "
          "makes up for. Backtested (analysis/backtest_rr_bucket.py): R:R 0.8-1.2 is the empirical sweet spot, "
          "R:R 1.2-1.5 has shown NEGATIVE expected value historically -- don't manually favor a high R:R number.)")

    print("\n--- Notes (events, recent news, for the qualified BUY picks above) ---")
    for i, r in enumerate(buy_rows):
        print(f"\n#{i + 1} {r['Symbol']}: {r['Verdict']}")
        print(f"   Events     : {r['Events']}")
        print(f"   Recent News: {r['RecentNews']}")

    print("\nExit rule (backtested): reaching the +12% target is NOT a sell "
          "signal -- only a hit trailing stop or the ~90-calendar-day time exit are real exits.")

    offer_paper_trade(paper_trade_candidates)


if __name__ == "__main__":
    main()
