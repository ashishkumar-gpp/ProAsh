"""
Daily data downloader for the AshishStock swing20 model -- runs entirely
inside this project (self-contained, path computed from this file's own
location). Downloads, per trading day missing from the local files:
  1. NSE full bhavcopy       -> Sec_bhavdata/sec_bhavdata_full_DDMMYYYY.csv
  2. NSE all-index closes    -> pr_auto_YYYYMM.csv (Nifty 50 / India VIX /
                                 Nifty Midcap 100 / Nifty SMLCAP 250 rows only)
  3. BSE full bhavcopy       -> bse_bhavcopy/BhavCopy_BSE_CM_0_0_0_YYYYMMDD_F_0000.CSV

Weekends/holidays simply return no data and are skipped (not treated as
errors). Already-downloaded files are never re-fetched unless --force is
passed. Run it, then run analysis/swing20_screener.py (or use AshStock.bat,
which chains both).
"""
import argparse
import io
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import pandas as pd
import requests

# Project root (parent of analysis/), derived from this file's own path -- no hardcoded location.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BHAV_DIR = os.path.join(ROOT, "Sec_bhavdata")
BSE_DIR = os.path.join(ROOT, "bse_bhavcopy")

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
TIMEOUT = 20
_SESSION = requests.Session()  # connection pooling/keep-alive across worker threads
_INDEX_LOCK = threading.Lock()  # pr_auto_YYYYMM.csv is shared across days in the same month
_PRINT_LOCK = threading.Lock()

# NSE index name (as it appears in ind_close_all_*.csv) -> Symbol label used in pr_*.csv
INDEX_MAP = {
    "nifty 50": "Nifty 50",
    "india vix": "India VIX",
    "nifty midcap 100": "Nifty Midcap 100",
    "nifty smallcap 250": "Nifty SMLCAP 250",
}
PR_COLUMNS = ["Date", "Symbol", "Open", "High", "Low", "Close", "Volume",
              "Turnover", "52W High", "52W Low", "Index/Sector Data"]


def trading_days(start, end):
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri only; NSE/BSE holidays are handled as empty-download skips
            yield d
        d += timedelta(days=1)


def _get(url):
    r = _SESSION.get(url, headers=HEADERS, timeout=TIMEOUT)
    return r


def _classify_no_data(status_code, body_text, expected_prefix=None, expected_hint=None):
    head = (body_text or "").lstrip()
    if status_code == 404:
        return "no-data: not-published-or-holiday (404)"
    if status_code != 200:
        return f"no-data: http-{status_code}"

    if expected_prefix and head.upper().startswith(expected_prefix.upper()):
        return None
    if expected_hint and expected_hint in (body_text or "")[:500]:
        return None

    # 200 with unexpected payload is usually an HTML page, bot-check page, or schema drift.
    return "no-data: unexpected-content (200)"


# --------------------------------------------------------------- 1. NSE bhavcopy
def download_nse_bhav(d, force=False):
    fname = f"sec_bhavdata_full_{d.strftime('%d%m%Y')}.csv"
    path = os.path.join(BHAV_DIR, fname)
    if os.path.exists(path) and not force:
        return "skip-exists"
    url = f"https://nsearchives.nseindia.com/products/content/{fname}"
    try:
        r = _get(url)
    except requests.RequestException as e:
        return f"error: {e}"
    no_data = _classify_no_data(r.status_code, r.text, expected_prefix="SYMBOL")
    if no_data:
        return no_data
    os.makedirs(BHAV_DIR, exist_ok=True)
    with open(path, "wb") as f:
        f.write(r.content)
    return "downloaded"


# --------------------------------------------------------------- 2. NSE index closes
def download_index_close(d, force=False):
    url = f"https://nsearchives.nseindia.com/content/indices/ind_close_all_{d.strftime('%d%m%Y')}.csv"
    try:
        r = _get(url)
    except requests.RequestException as e:
        return f"error: {e}"
    no_data = _classify_no_data(r.status_code, r.text, expected_hint="Index Name")
    if no_data:
        return no_data

    raw = pd.read_csv(io.StringIO(r.text))
    raw.columns = [c.strip() for c in raw.columns]
    raw["_key"] = raw["Index Name"].astype(str).str.strip().str.lower()
    rows = []
    for key, symbol in INDEX_MAP.items():
        match = raw[raw["_key"] == key]
        if match.empty:
            continue
        m = match.iloc[0]

        def num(v):
            v = str(v).strip()
            return 0.0 if v in ("-", "", "nan") else float(v)

        rows.append({
            "Date": d.strftime("%Y-%m-%d"),
            "Symbol": symbol,
            "Open": num(m["Open Index Value"]),
            "High": num(m["High Index Value"]),
            "Low": num(m["Low Index Value"]),
            "Close": num(m["Closing Index Value"]),
            "Volume": num(m["Volume"]),
            "Turnover": num(m["Turnover (Rs. Cr.)"]),
            "52W High": 0.0,
            "52W Low": 0.0,
            "Index/Sector Data": "Y",
        })
    if not rows:
        return "no-matching-indices"

    out_path = os.path.join(ROOT, f"pr_auto_{d.strftime('%Y%m')}.csv")
    new_df = pd.DataFrame(rows, columns=PR_COLUMNS)
    with _INDEX_LOCK:  # multiple worker threads can hit the same monthly file at once
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path)
            existing = existing[~((existing["Date"] == new_df["Date"].iloc[0]) &
                                   (existing["Symbol"].isin(new_df["Symbol"])))]
            new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df = new_df.sort_values(["Date", "Symbol"])
        new_df.to_csv(out_path, index=False)
    return "downloaded"


# --------------------------------------------------------------- 3. BSE bhavcopy
def download_bse_bhav(d, force=False):
    fname = f"BhavCopy_BSE_CM_0_0_0_{d.strftime('%Y%m%d')}_F_0000.CSV"
    path = os.path.join(BSE_DIR, fname)
    if os.path.exists(path) and not force:
        return "skip-exists"
    url = f"https://www.bseindia.com/download/BhavCopy/Equity/{fname}"
    try:
        r = _get(url)
    except requests.RequestException as e:
        return f"error: {e}"
    no_data = _classify_no_data(r.status_code, r.text, expected_prefix="TradDt")
    if no_data:
        return no_data
    os.makedirs(BSE_DIR, exist_ok=True)
    with open(path, "wb") as f:
        f.write(r.content)
    return "downloaded"


def latest_local_date(pattern_dir, prefix, date_fmt, suffix=".csv"):
    if not os.path.isdir(pattern_dir):
        return None
    dates = []
    for fn in os.listdir(pattern_dir):
        if fn.startswith(prefix) and fn.endswith(suffix):
            token = fn[len(prefix):-len(suffix)]
            try:
                dates.append(datetime.strptime(token, date_fmt).date())
            except ValueError:
                continue
    return max(dates) if dates else None


def latest_local_pr_index_date(root_dir):
    latest = None
    for fn in os.listdir(root_dir):
        if not (fn.startswith("pr_auto_") and fn.endswith(".csv")):
            continue
        path = os.path.join(root_dir, fn)
        try:
            df = pd.read_csv(path, usecols=["Date"])
        except Exception:
            continue
        if df.empty:
            continue
        dt = pd.to_datetime(df["Date"], errors="coerce").max()
        if pd.isna(dt):
            continue
        d = dt.date()
        if latest is None or d > latest:
            latest = d
    return latest


def _fetch_day(d, args):
    results = {}
    if not args.skip_nse:
        results["NSE bhav"] = download_nse_bhav(d, force=args.force)
    if not args.skip_index:
        results["Index close"] = download_index_close(d, force=args.force)
    if not args.skip_bse:
        results["BSE bhav"] = download_bse_bhav(d, force=args.force)
    return d, results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", help="DD-MM-YYYY, default = day after latest local NSE bhavcopy")
    ap.add_argument("--end", help="DD-MM-YYYY, default = today")
    ap.add_argument("--skip-nse", action="store_true")
    ap.add_argument("--skip-index", action="store_true")
    ap.add_argument("--skip-bse", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-download even if local file already exists")
    ap.add_argument("--workers", type=int, default=8, help="parallel download threads (default 8)")
    args = ap.parse_args()

    end = datetime.strptime(args.end, "%d-%m-%Y").date() if args.end else date.today()
    if args.start:
        start = datetime.strptime(args.start, "%d-%m-%Y").date()
    else:
        latest = latest_local_date(BHAV_DIR, "sec_bhavdata_full_", "%d%m%Y")
        start = (latest + timedelta(days=1)) if latest else (end - timedelta(days=7))

    days = list(trading_days(start, end))
    print(f"Fetching trading days {start} -> {end} ({len(days)} days, {args.workers} parallel workers)")
    counts = {"downloaded": 0, "skip-exists": 0, "no-data": 0, "error": 0, "no-matching-indices": 0}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fetch_day, d, args): d for d in days}
        done = 0
        for fut in as_completed(futures):
            d, results = fut.result()
            done += 1
            line = f"[{done}/{len(days)}] {d}  " + "  |  ".join(f"{k}: {v}" for k, v in results.items())
            with _PRINT_LOCK:
                print(line)
            for v in results.values():
                for key in counts:
                    if v.startswith(key):
                        counts[key] += 1
                        break

    print(f"\nDone. {counts}")

    latest_nse = latest_local_date(BHAV_DIR, "sec_bhavdata_full_", "%d%m%Y")
    latest_bse = latest_local_date(BSE_DIR, "BhavCopy_BSE_CM_0_0_0_", "%Y%m%d", suffix="_F_0000.CSV")
    latest_idx = latest_local_pr_index_date(ROOT)
    print("Local data snapshot:")
    print(f"  NSE bhav latest file date: {latest_nse}")
    print(f"  BSE bhav latest file date: {latest_bse}")
    print(f"  NSE index latest row date: {latest_idx}")


if __name__ == "__main__":
    main()
