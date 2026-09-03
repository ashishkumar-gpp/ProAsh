"""
Looks up recent/upcoming NSE corporate actions (bonus, split, rights, dividend,
buyback, board meeting, AGM) for a symbol, so the screener can flag picks whose
price history may be distorted by an unadjusted action, or whose hold window
overlaps a pending one.

Read-only, best-effort: any network/parsing failure returns "unknown" rather
than raising, so a lookup problem never blocks swing20_screener.py from
printing its picks. NSE's live API (unlike the static bhavcopy archive URLs
download_data.py uses) needs browser-like session cookies or it 401s on a
bare request, so a homepage visit is done first to pick those up.
"""
from datetime import datetime

import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}
TIMEOUT = 15
WINDOW_DAYS = 45  # only actions whose ex-date falls within +/- this many days of today matter

# Ex-dates for these subjects still matter up to window_days in the PAST too (unadjusted
# chart data risk lingers after the ex-date) -- everything else (board meeting/AGM/other)
# is only a real pending risk if it's still upcoming, so those are matched forward-only.
CHART_RISK_KEYWORDS = ("bonus", "split", "dividend", "rights", "buyback")


def _session():
    s = requests.Session()
    s.headers.update(HEADERS)
    s.get("https://www.nseindia.com", timeout=TIMEOUT)  # sets cookies the API below requires
    return s


def get_corporate_actions(symbol, window_days=WINDOW_DAYS, session=None):
    """Short human-readable summary of any corporate action with an ex-date within
    +/- window_days of today, "-" if none, or "unknown" if the lookup failed.
    Pass a shared `session` (from `_session()`) when calling this for many symbols
    in one run, to avoid a fresh homepage cookie visit per symbol."""
    try:
        s = session or _session()
        r = s.get("https://www.nseindia.com/api/corporates-corporateActions",
                   params={"index": "equities", "symbol": symbol}, timeout=TIMEOUT)
        if r.status_code != 200:
            return "unknown"
        rows = r.json()
    except Exception:
        return "unknown"

    today = datetime.now().date()
    hits = []
    for row in rows or []:
        ex = row.get("exDate")
        if not ex or ex == "-":
            continue
        try:
            ex_date = datetime.strptime(ex, "%d-%b-%Y").date()
        except ValueError:
            continue
        subject = (row.get("subject") or "").strip()
        days = (ex_date - today).days
        is_chart_risk = any(k in subject.lower() for k in CHART_RISK_KEYWORDS)
        in_window = abs(days) <= window_days if is_chart_risk else 0 <= days <= window_days
        if in_window:
            hits.append(f"{subject} (ex {ex_date.strftime('%d-%b')})")

    if not hits:
        return "-"
    return "; ".join(hits[:2])  # keep the table cell short


MA_KEYWORDS = ("open offer", "acquisition", "change in control", "substantial acquisition")
# "takeover"/"takeovers"/"sast" alone are too generic -- they're also just the regulation's NAME,
# cited in routine annual promoter encumbrance disclosures (Regulation 31 of SEBI SAST) that report
# NO acquisition activity (e.g. "promoter declares no new share encumbrances"). Only count these as
# a real M&A hit when the text isn't also one of those routine/nil disclosures (see OMAXE 24-Jul-2026
# false positive: "Disclosure under SEBI Takeover Regulations" was just a NIL encumbrance filing).
WEAK_MA_KEYWORDS = ("takeover", "takeovers", "sast")
ROUTINE_DISCLOSURE_PHRASES = ("no encumbrance", "nil encumbrance", "no new encumbrance", "regulation 31")


def get_ma_open_offer_alert(symbol, window_days=WINDOW_DAYS, session=None):
    """Short human-readable summary of any M&A / open-offer / change-of-control
    disclosure (NSE's corporate-announcements feed -- Regulation 30/SAST filings,
    a DIFFERENT NSE endpoint than get_corporate_actions' bonus/split/dividend one,
    which never carries these) within the last window_days, "-" if none, "unknown"
    if the lookup failed. Verified against KRONOX's 20-Aug-2026 Indo Borax/Zenrock
    open-offer filings, which get_corporate_actions cannot see at all.
    Pass a shared `session` (from `_session()`) when calling this for many symbols
    in one run, to avoid a fresh homepage cookie visit per symbol."""
    try:
        s = session or _session()
        r = s.get("https://www.nseindia.com/api/corporate-announcements",
                   params={"index": "equities", "symbol": symbol}, timeout=TIMEOUT)
        if r.status_code != 200:
            return "unknown"
        rows = r.json()
    except Exception:
        return "unknown"

    today = datetime.now().date()
    hits = []
    for row in rows or []:
        an_dt = row.get("an_dt")
        if not an_dt:
            continue
        try:
            ann_date = datetime.strptime(an_dt.split()[0], "%d-%b-%Y").date()
        except ValueError:
            continue
        if (today - ann_date).days > window_days:
            continue
        haystack = f"{row.get('desc') or ''} {row.get('attchmntText') or ''}".lower()
        strong_hit = any(k in haystack for k in MA_KEYWORDS)
        weak_hit = any(k in haystack for k in WEAK_MA_KEYWORDS)
        routine = any(p in haystack for p in ROUTINE_DISCLOSURE_PHRASES)
        if strong_hit or (weak_hit and not routine):
            hits.append(f"{(row.get('desc') or '').strip()} ({ann_date.strftime('%d-%b')})")

    if not hits:
        return "-"
    return "; ".join(list(dict.fromkeys(hits))[:2])  # de-dup (same event often filed twice), keep short


def _parse_split_ratio(subject_text):
    """Parse split/bonus ratio from NSE corporate action subject text.
    Returns adjustment factor: e.g. 'Bonus 1:1' -> 0.5 (halve pre-bonus closes),
    'Stock Split 1:2' -> 2.0 (double pre-split closes), None if unparseable."""
    import re
    if not subject_text:
        return None
    s = subject_text.upper().strip()
    # Common patterns: "Bonus 1:1", "Bonus 1:2", "Stock Split 1:2", "Stock Split 1:5", etc.
    patterns = [
        r"BONUS\s+(\d+)\s*:\s*(\d+)",
        r"STOCK\s+SPLIT\s+(\d+)\s*:\s*(\d+)",
        r"SPLIT\s+(\d+)\s*:\s*(\d+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, s)
        if m:
            numerator, denominator = int(m.group(1)), int(m.group(2))
            if denominator == 0:
                return None
            if "BONUS" in s:
                # Bonus: pre-bonus close needs to be scaled DOWN by the bonus factor
                return denominator / (numerator + denominator)
            else:
                # Split: pre-split close needs to be scaled (inverse of split ratio)
                return denominator / numerator
    return None


def get_split_adjustments_for_symbol(symbol):
    """Fetch all historical corporate actions for SYMBOL and return list of
    (ex_date, adjustment_factor) tuples for any splits/bonuses found.
    Returns empty list if none found or lookup fails."""
    try:
        s = _session()
        r = s.get("https://www.nseindia.com/api/corporates-corporateActions",
                   params={"index": "equities", "symbol": symbol}, timeout=TIMEOUT)
        if r.status_code != 200:
            return []
        rows = r.json()
    except Exception:
        return []

    adjustments = []
    for row in rows or []:
        ex = row.get("exDate")
        subject = (row.get("subject") or "").strip()
        if not ex or ex == "-" or not subject:
            continue
        try:
            ex_date = datetime.strptime(ex, "%d-%b-%Y").date()
        except ValueError:
            continue
        ratio = _parse_split_ratio(subject)
        if ratio is not None:
            adjustments.append((ex_date, ratio))
    return adjustments
