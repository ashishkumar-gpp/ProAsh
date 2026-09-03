"""Interactive "add to paper trading?" prompt shown after run_pipeline.py prints its
Combined Result. Lets you selectively add picks to the tracker (track_picks.py /
PaperTrade.bat) and optionally override the entry price -- target/stop are then set from
the pipeline's own +12% target and its chandelier-trailing StopLoss (same value shown in
the Combined Result table), NOT a flat percentage -- track_picks.py recomputes this stop
fresh (trailing) on every future check.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_picks import add_pick

TARGET_PCT = 0.12    # matches run_pipeline.py's Target(+12%) column
FALLBACK_STOP_PCT = -0.08  # used only if the pipeline couldn't compute a chandelier stop (not enough history)


def offer_paper_trade(picks):
    """picks: list of dicts each with Market/Symbol/CMP (entry) and StopLoss (chandelier,
    may be None). Press Enter to skip all."""
    if not picks or not sys.stdin.isatty():
        return

    print("\nAdd any of these to the paper-trade tracker (PaperTrade.bat)?")
    for i, p in enumerate(picks, 1):
        target = round(p["CMP"] * (1 + TARGET_PCT), 2)
        stop_ref = p.get("StopLoss") or round(p["CMP"] * (1 + FALLBACK_STOP_PCT), 2)
        print(f"  {i}. {p['Market']} {p['Symbol']}  CMP {p['CMP']}  target {target}  stop-ref {stop_ref}")

    try:
        choice = input("Enter number(s) comma-separated, 'all', or Enter to skip: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if not choice:
        return

    if choice.lower() == "all":
        selected = picks
    else:
        selected = []
        for tok in choice.split(","):
            tok = tok.strip()
            if tok.isdigit() and 1 <= int(tok) <= len(picks):
                selected.append(picks[int(tok) - 1])
            elif tok:
                print(f"  Skipping invalid selection: {tok!r}")

    for p in selected:
        default_entry = p["CMP"]
        try:
            raw = input(f"  Entry price for {p['Symbol']} [Enter for CMP {default_entry}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        try:
            entry = float(raw) if raw else default_entry
        except ValueError:
            print(f"  Invalid price {raw!r}, using CMP {default_entry}.")
            entry = default_entry

        target = round(entry * (1 + TARGET_PCT), 2)
        stop_loss = p.get("StopLoss")
        if stop_loss is None or stop_loss >= entry:
            stop_loss = round(entry * (1 + FALLBACK_STOP_PCT), 2)
        pick = add_pick(p["Market"], p["Symbol"], entry, target, stop_loss)
        print(f"  Added {pick['market']} {pick['symbol']} (entry {pick['entry_date']}, buy {pick['buy_price']}, "
              f"target {pick['target']}, stop-ref {pick['stop_loss']}).")
