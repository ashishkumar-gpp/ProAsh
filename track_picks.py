"""Paper-trade tracker for ProAsh's ensemble picks (paper_trade_prompt.py adds to it
right after proash_pipeline.py prints its FINAL PICKS table). Checks each tracked
pick's latest available close price against its target/stop -- ProAsh has no live
intraday quote source, so "current price" is the latest close from the same
bhavcopy files the pipeline reads. Appends a dated row to the single ongoing
paper_trades_log.csv and offers to stop tracking (remove) any of the checked picks.

Exit rule (same evidence-backed rule as AshFund, see AshFund/pipeline/backtest_exit_rules.py):
hitting the +12% target is NOT a sell signal -- 'stop_loss' is recomputed fresh on every
check as a chandelier trailing stop (highest close since entry - 3xATR14), it only
ratchets up, never down. The only real exit triggers are the trailing stop being hit,
or the max-hold window elapsing; reaching +12% just means "still holding, stop has now
tightened".

Usage:
    python track_picks.py                                  # check, print, log, offer removal
    python track_picks.py --list                           # list currently tracked picks
    python track_picks.py --history                        # print the full logged history
    python track_picks.py --add NSE DIXON 14197 15900 --stop-loss 13065
    python track_picks.py --remove DIXON [--entry-date 2026-08-07] [--buy-price 14197]

Or just run PaperTrade.bat.
"""
import argparse
import csv
import os
import json
import sys
from datetime import date, datetime

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.join(ROOT, "engine")
sys.path.insert(0, ENGINE_DIR)
import premove_factor_analysis as nse_mod  # noqa: E402
import bse_premove_factor_analysis as bse_mod  # noqa: E402
from swing20_screener import render_table  # noqa: E402
from stop_loss import get_stop_loss  # noqa: E402

PICKS_PATH = os.path.join(ROOT, "data", "paper_trades_picks.json")
LOG_PATH = os.path.join(ROOT, "data", "paper_trades_log.csv")
DAY_MAX = 90  # paper-trade time-exit, extended from the pipeline's 45-session label window
             # (AshFund backtested 45->90 sessions and found it raised avg_return/sortino --
             # this only delays the no-stop/no-target forced sale, the live chandelier stop
             # still protects downside the whole time)
HOLD_CALENDAR_DAYS = round(DAY_MAX * 7 / 5)  # ~126 calendar days

CHECKPOINT_SESSIONS = 20
CHECKPOINT_CALENDAR_DAYS = round(CHECKPOINT_SESSIONS * 7 / 5)  # ~28 calendar days
CHECKPOINT_MIN_GAIN_PCT = 5.0

_BHAV_CACHE = {}


def _bhav(market):
    if market not in _BHAV_CACHE:
        _BHAV_CACHE[market] = nse_mod.load_bhav() if market == "NSE" else bse_mod.load_bhav()
    return _BHAV_CACHE[market]


def _load_picks():
    if not os.path.exists(PICKS_PATH):
        return []
    with open(PICKS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_picks(picks):
    with open(PICKS_PATH, "w", encoding="utf-8") as f:
        json.dump(picks, f, indent=2)


def add_pick(market, symbol, buy_price, target, stop_loss=None, entry_date=None):
    pick = {
        "market": market.upper(), "symbol": symbol.upper(),
        "entry_date": entry_date or date.today().isoformat(),
        "buy_price": buy_price, "target": target, "stop_loss": stop_loss,
    }
    picks = _load_picks()
    picks.append(pick)
    _save_picks(picks)
    return pick


def remove_picks(symbol, entry_date=None, buy_price=None):
    picks = _load_picks()
    symbol = symbol.upper()
    kept, removed = [], 0
    for p in picks:
        if (p["symbol"] == symbol
                and (entry_date is None or p["entry_date"] == entry_date)
                and (buy_price is None or abs(p["buy_price"] - buy_price) < 0.01)):
            removed += 1
        else:
            kept.append(p)
    _save_picks(kept)
    return removed


def _latest_close(market, symbol):
    df = _bhav(market)
    sub = df[df["SYMBOL"] == symbol]
    if sub.empty:
        return None
    return sub.loc[sub["DATE1"].idxmax(), "CLOSE_PRICE"]


def _status(current_price, target, stop_loss, days_held):
    if stop_loss is not None and current_price <= stop_loss:
        return "STOP HIT"
    if days_held >= HOLD_CALENDAR_DAYS:
        return "EXPIRED (time exit)"
    if current_price >= target:
        return "ABOVE TARGET (hold -- trail stop)"
    return "OPEN"


def _exit_signal(status, days_held, pct_move):
    if status == "STOP HIT":
        return "EXIT NOW (stop hit)"
    if status.startswith("EXPIRED"):
        return "EXIT (45-session window over)"
    if status.startswith("ABOVE TARGET"):
        return "HOLD (trailing stop protects gain)"
    if days_held >= CHECKPOINT_CALENDAR_DAYS:
        if pct_move < CHECKPOINT_MIN_GAIN_PCT:
            return "CONSIDER EXIT (stalled -- not gaining by day~20 checkpoint)"
        return "HOLD (on pace)"
    return "HOLD (too early to judge)"


def _validation(status, pct_move, days_held):
    if status == "STOP HIT":
        return "Invalidated"
    if status.startswith("EXPIRED"):
        return "Validated" if pct_move > 0 else "Invalidated"
    if status.startswith("ABOVE TARGET"):
        return "Validated (keep holding)"
    if days_held < 15:
        return "Monitoring"
    if days_held < 40:
        return "Under Review" if pct_move < -5 else "Monitoring"
    return "Under Review" if pct_move < 0 else "Monitoring"


def _current_trailing_stop(pick):
    hist = _bhav(pick["market"])
    fresh, _ = get_stop_loss(hist, pick["symbol"], pd.Timestamp(pick["entry_date"]), pick["buy_price"])
    prior = pick["stop_loss"]
    if fresh is None:
        return prior
    if prior is None:
        return fresh
    return max(prior, fresh)


def check_picks():
    results = []
    picks = _load_picks()
    dirty = False
    for pick in picks:
        current_price = _latest_close(pick["market"], pick["symbol"])
        if current_price is None:
            print(f"[track_picks] no bhavcopy data for {pick['market']} {pick['symbol']}, skipping")
            continue
        entry = datetime.strptime(pick["entry_date"], "%Y-%m-%d").date()
        days_held = (date.today() - entry).days
        pct_move = round((current_price / pick["buy_price"] - 1) * 100, 2)
        trailing_stop = _current_trailing_stop(pick)
        if trailing_stop != pick["stop_loss"]:
            pick["stop_loss"] = trailing_stop
            dirty = True
        status = _status(current_price, pick["target"], pick["stop_loss"], days_held)
        exit_signal = _exit_signal(status, days_held, pct_move)
        validation = _validation(status, pct_move, days_held)
        results.append({
            "date": date.today().isoformat(), "market": pick["market"], "symbol": pick["symbol"],
            "entry_date": pick["entry_date"], "buy_price": pick["buy_price"], "current_price": current_price,
            "pct_move": pct_move, "target": pick["target"], "stop_loss": pick["stop_loss"],
            "days_held": days_held, "status": status, "exit_signal": exit_signal, "validation": validation,
        })
    if dirty:
        _save_picks(picks)
    return results


def _append_log(results):
    if not results:
        return
    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(results)


def _offer_removal(results):
    stop_or_expired = [r for r in results if r["status"] in ("STOP HIT",) or r["status"].startswith("EXPIRED")]
    if not stop_or_expired or not sys.stdin.isatty():
        return
    print("\nThese picks hit their stop or expired -- stop tracking any of them?")
    for i, r in enumerate(stop_or_expired, 1):
        print(f"  {i}. {r['market']} {r['symbol']}  {r['status']}  ({r['pct_move']:+.2f}%)")
    try:
        choice = input("Enter number(s) comma-separated, 'all', or Enter to keep tracking all: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if not choice:
        return
    selected = stop_or_expired if choice.lower() == "all" else [
        stop_or_expired[int(t) - 1] for t in choice.split(",") if t.strip().isdigit() and 1 <= int(t) <= len(stop_or_expired)
    ]
    for r in selected:
        n = remove_picks(r["symbol"], entry_date=r["entry_date"], buy_price=r["buy_price"])
        print(f"  Removed {r['symbol']} ({n} entry removed).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--add", nargs=4, metavar=("MARKET", "SYMBOL", "BUY_PRICE", "TARGET"))
    ap.add_argument("--stop-loss", type=float, default=None)
    ap.add_argument("--remove", metavar="SYMBOL")
    ap.add_argument("--entry-date", default=None)
    ap.add_argument("--buy-price", type=float, default=None)
    args = ap.parse_args()

    if args.list:
        picks = _load_picks()
        print(f"{len(picks)} tracked pick(s):")
        for p in picks:
            print(f"  {p['market']} {p['symbol']}  entry {p['entry_date']} @ {p['buy_price']}  "
                  f"target {p['target']}  stop {p['stop_loss']}")
        return

    if args.history:
        if not os.path.exists(LOG_PATH):
            print("No history logged yet.")
            return
        print(pd.read_csv(LOG_PATH).to_string(index=False))
        return

    if args.add:
        market, symbol, buy_price, target = args.add
        pick = add_pick(market, symbol, float(buy_price), float(target), args.stop_loss)
        print(f"Added {pick['market']} {pick['symbol']} (entry {pick['entry_date']}, buy {pick['buy_price']}, "
              f"target {pick['target']}, stop {pick['stop_loss']}).")
        return

    if args.remove:
        n = remove_picks(args.remove, args.entry_date, args.buy_price)
        print(f"Removed {n} entry(ies) for {args.remove.upper()}.")
        return

    results = check_picks()
    if not results:
        print("No tracked picks yet. Run ProAsh.bat and add some from the FINAL PICKS prompt, "
              "or use: python track_picks.py --add NSE SYMBOL 100 112 --stop-loss 92")
        return

    headers = ["Symbol", "Mkt", "Entry", "Buy", "CMP", "%Move", "Target", "Stop", "DaysHeld", "Status", "ExitSignal", "Validation"]
    rows = [[r["symbol"], r["market"], r["entry_date"], r["buy_price"], r["current_price"], f"{r['pct_move']:+.2f}%",
             r["target"], r["stop_loss"], r["days_held"], r["status"], r["exit_signal"], r["validation"]]
            for r in results]
    print(render_table(headers, rows, wrap_widths={"ExitSignal": 40, "Validation": 22}))

    _append_log(results)
    print(f"\nLogged {len(results)} row(s) to {LOG_PATH}")
    _offer_removal(results)


if __name__ == "__main__":
    main()
