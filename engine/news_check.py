"""
Looks up recent Google News headlines for a symbol (free, no API key -- see
evaluation in chat: yfinance/Alpha Vantage had poor/no coverage for the
Indian small/mid-cap names this screener actually picks, plain Google News
RSS search by symbol did not).

This is a HEADLINE FEED ONLY, no sentiment/impact scoring -- it tells you
something was published, not whether it's good or bad news. Read the
headlines yourself before trusting a pick. Source quality varies (mainline
press vs. aggregator/blog sites); Google News RSS is an unofficial, undocumented
feed with no SLA, so treat coverage as best-effort, not guaranteed-complete.

Read-only, best-effort: any network/parsing failure returns "unknown" rather
than raising, so a lookup problem never blocks swing20_screener.py from
printing its picks.
"""
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
TIMEOUT = 15
MAX_ITEMS = 2  # keep the table cell short
POOL_ITEMS = 10  # fetch more than we need so we can sort by actual pubDate


def get_recent_news(symbol, max_items=MAX_ITEMS):
    """Short human-readable summary of the most recent (by pubDate, not feed
    order) Google News headlines for "<symbol> stock India", each tagged with
    its publish date. "-" if none found, "unknown" if the lookup failed."""
    try:
        r = requests.get(
            "https://news.google.com/rss/search",
            params={"q": f"{symbol} stock India", "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
            headers=HEADERS, timeout=TIMEOUT,
        )
        if r.status_code != 200:
            return "unknown"
        root = ET.fromstring(r.content)
    except Exception:
        return "unknown"

    items = root.findall("./channel/item")[:POOL_ITEMS]
    if not items:
        return "-"

    parsed = []
    for item in items:
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        pub_raw = item.findtext("pubDate")
        try:
            pub_dt = parsedate_to_datetime(pub_raw) if pub_raw else None
        except (TypeError, ValueError):
            pub_dt = None
        parsed.append((pub_dt, title))

    if not parsed:
        return "-"

    # feed order isn't guaranteed chronological -- sort so the newest article
    # leads; undated items (rare) sort last rather than being dropped.
    parsed.sort(key=lambda p: p[0] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    hits = []
    for pub_dt, title in parsed[:max_items]:
        date_str = pub_dt.strftime("%d-%b-%Y") if pub_dt else "date unknown"
        hits.append(f"{title} ({date_str})")
    return "; ".join(hits)

