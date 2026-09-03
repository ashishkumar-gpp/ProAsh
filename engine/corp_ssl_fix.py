"""
Fixes yfinance/curl_cffi SSL failures on this machine (corporate TLS-intercepting proxy).

Root cause: `requests` works out of the box here because `pip_system_certs` patches Python's
own `ssl` module to trust the Windows certificate store (which has the corporate proxy's root
CA installed). yfinance's `.info`/quoteSummary calls go through `curl_cffi` (a separate compiled
libcurl/BoringSSL binding used for browser-TLS-impersonation to get past Yahoo's bot detection)
which does NOT go through Python's `ssl` module at all, so that patch has no effect on it -- it
keeps using only its own bundled public CA list and fails with "unable to get local issuer
certificate" against the proxy's re-signed certs.

Fix: build a PEM bundle from the Windows ROOT+CA certificate stores (pure stdlib `ssl.
enum_certificates`, no PowerShell/subprocess needed) and point curl_cffi at it via the
CURL_CA_BUNDLE env var, which it does honor when set before use.

Call ensure_ca_bundle() once before importing/using yfinance. No-op (fast) on non-Windows or if
already applied this process.
"""
import os
import ssl
import sys
import time

_CACHE_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "AshishStock")
_CACHE_FILE = os.path.join(_CACHE_DIR, "corp_ca_bundle.pem")
_MAX_AGE_SECS = 7 * 24 * 3600  # rebuild weekly in case the corporate CA rotates
_applied = False


def _build_bundle():
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(_CACHE_FILE, "w") as f:
        for store in ("ROOT", "CA"):
            for der, _encoding, _trust in ssl.enum_certificates(store):
                f.write(ssl.DER_cert_to_PEM_cert(der))


def ensure_ca_bundle():
    global _applied
    if _applied or sys.platform != "win32":
        return
    stale = not os.path.exists(_CACHE_FILE) or (time.time() - os.path.getmtime(_CACHE_FILE) > _MAX_AGE_SECS)
    if stale:
        _build_bundle()
    os.environ["CURL_CA_BUNDLE"] = _CACHE_FILE
    os.environ["SSL_CERT_FILE"] = _CACHE_FILE
    _applied = True
