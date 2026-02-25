#!/usr/bin/env python3
"""
phases/phase1/1_fetch.py
========================

Step 1: Fetch historical OHLCV data from Angel One API.

Reads tickers from Nifty 500 CSV file and downloads historical data.
All output paths are derived dynamically from __file__ — never hardcoded.

Usage:
    python3 phases/phase1/1_fetch.py
    python3 phases/phase1/1_fetch.py --as-of-date 2024-01-01   # pin for reproducibility
    python3 phases/phase1/1_fetch.py --limit 50 --skip-existing
    python3 phases/phase1/1_fetch.py --start-from 100 --verbose

Output (all under <script_dir>/phase01_stage01_data_fetcher_output/):
    temp_stocks/<symbol>.csv        — OHLCV data, UTC timestamps
    temp_stocks/manifest.json       — run parameters + schema declaration
    report/download_report.csv      — success + failure audit log
    logs/fetch_<timestamp>.log      — full execution log
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import time
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

# =============================================================================
# DYNAMIC OUTPUT PATHS  (idiomatic: Path(__file__).stem only)
# =============================================================================

_SCRIPT_DIR  = Path(__file__).resolve().parent
_SCRIPT_STEM = Path(__file__).stem                        # e.g. "phase01_stage01_data_fetcher"

OUTPUT_DIR  = _SCRIPT_DIR / f"{_SCRIPT_STEM}_output"
STOCKS_DIR  = OUTPUT_DIR / "temp_stocks"
LOGS_DIR    = OUTPUT_DIR / "logs"
REPORT_DIR  = OUTPUT_DIR / "report"

for _d in (OUTPUT_DIR, STOCKS_DIR, LOGS_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DEFAULT_CSV_PATH = _SCRIPT_DIR / "nifty_500" / "ind_nifty500list.csv"

# =============================================================================
# LOAD ENVIRONMENT VARIABLES
# =============================================================================

for _env_path in [_SCRIPT_DIR / "angel.env", _SCRIPT_DIR / ".env"]:
    if _env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(_env_path)
        except ImportError:
            with open(_env_path) as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line and not _line.startswith('#') and '=' in _line:
                        _k, _v = _line.split('=', 1)
                        os.environ[_k.strip()] = _v.strip()
        print(f"[INFO] Loaded environment from: {_env_path}")
        break
else:
    print("[WARNING] No .env file found. Using system environment variables.")

# =============================================================================
# ANGEL ONE SMART API
# =============================================================================

try:
    from SmartApi import SmartConnect
    import pyotp
except ImportError as _e:
    print(f"ERROR: Missing dependency — {_e}")
    print("Run: pip install smartapi-python pyotp")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_HISTORICAL_DAYS     = 2000
DEFAULT_EXCHANGE            = "NSE"
DEFAULT_REQUESTS_PER_MINUTE = 120       # 2 calls/second — under Angel One's 10/s limit

# NSE timezone (IST = UTC+5:30).
# For Python 3.9+ prefer: from zoneinfo import ZoneInfo; NSE_TZ = ZoneInfo("Asia/Kolkata")
# We use a fixed-offset here for 3.8 compatibility; NSE does not observe DST so this is correct.
NSE_TZ = timezone(timedelta(hours=5, minutes=30))

# Gap detection: NSE worst-case holiday cluster ≈ Diwali + surrounding weekends (~12 days).
# Gaps up to this threshold are normal; beyond it a warning is emitted per gap instance.
MAX_EXPECTED_CALENDAR_GAP_DAYS = 15

# Symbols that are known non-stock sentinels in the Nifty 500 CSV export.
# DUMMYHDLVR is a placeholder row sometimes injected by NSE data providers to
# signal a delivery-only dummy instrument; it has no real OHLCV data.
_SKIP_SYMBOLS = frozenset({'nan', 'DUMMYHDLVR'})

# Rate-limit error phrases — explicit multi-word phrases to avoid false matches
# on innocent words like "exchange rate" or "interest rate".
_RATE_LIMIT_PHRASES = ('rate limit', 'too many requests', 'access denied', '502', '429')

# =============================================================================
# LOGGING  (setup deferred to main() — avoids file/dir side-effects on import)
# =============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure root logger. Must be called from main(), NOT at module level."""
    level    = logging.DEBUG if verbose else logging.INFO
    log_file = LOGS_DIR / f"fetch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


# Placeholder; populated in main() after argument parsing.
logger: logging.Logger = logging.getLogger(__name__)

# =============================================================================
# SYMBOL SANITISATION  (single canonical function)
# =============================================================================

def sanitise_symbol(symbol: str) -> str:
    """
    Convert a raw ticker to a safe, lowercase filesystem stem.

    Steps (applied in order):
      1. Replace '&' with 'and'
      2. Replace spaces with '_'
      3. Strip characters illegal on all major OSes: < > : " / \\ | ? *
      4. Lower-case

    This is the ONE canonical implementation used both when writing files
    and when checking for existing files, preventing skip-existing mismatches.
    """
    s = symbol.replace('&', 'and').replace(' ', '_')
    s = re.sub(r'[<>:"/\\|?*]', '', s)
    return s.lower()

# =============================================================================
# RATE-LIMIT HELPER
# =============================================================================

def _is_rate_limit_error(msg: str) -> bool:
    """True if the message signals a rate-limit or gateway error."""
    m = msg.lower()
    return any(phrase in m for phrase in _RATE_LIMIT_PHRASES)

# =============================================================================
# ANGEL ONE API CLIENT
# =============================================================================

class AngelOneClient:
    """
    Angel One SmartAPI client with rate limiting, retry/backoff, and
    strict exact-match token resolution.

    Design decisions
    ----------------
    * Rate limiting uses time.monotonic() to avoid wall-clock drift.
    * Exponential backoff on rate-limit responses (5/10/15 s tiers).
    * Token resolution is EXACT-MATCH ONLY — no fuzzy fallback that could
      silently return a different ticker's data.
    * fetch_historical_data requires timezone-aware datetimes so the
      fetch window is unambiguous across runs.
    """

    def __init__(self, requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE):
        self.api_key      = os.getenv('ANGEL_API_KEY')
        self.secret_key   = os.getenv('ANGEL_SECRET_KEY')
        self.username     = os.getenv('ANGEL_USERNAME')
        self.mpin         = os.getenv('ANGEL_MPIN')
        self.totp_secret  = os.getenv('ANGEL_TOTP_SECRET')
        self._check_credentials()

        self.obj             = None
        self.is_logged_in    = False
        self.min_interval    = 60.0 / requests_per_minute
        self._last_req_mono  = 0.0

    # ------------------------------------------------------------------
    # Credential validation
    # ------------------------------------------------------------------

    def _check_credentials(self) -> None:
        missing = [k for k, v in {
            'ANGEL_API_KEY':     self.api_key,
            'ANGEL_SECRET_KEY':  self.secret_key,
            'ANGEL_USERNAME':    self.username,
            'ANGEL_MPIN':        self.mpin,
            'ANGEL_TOTP_SECRET': self.totp_secret,
        }.items() if not v]

        if missing:
            lines = '\n'.join(f'  - {k}' for k in missing)
            print(f"\n{'='*60}\nERROR: Missing environment variables:\n{lines}\n"
                  "Create angel.env next to this script with the above keys.\n"
                  + '='*60)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_req_mono
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def _update_request_time(self) -> None:
        self._last_req_mono = time.monotonic()

    # ------------------------------------------------------------------
    # TOTP  (RFC 4648 §6 — Base32 padding)
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_totp_secret(secret: str) -> str:
        """
        Normalise a Base32 TOTP secret per RFC 4648 §6.

        Valid Base32 strings must be a multiple of 8 characters.
        Generic formula:  padding = (8 - len % 8) % 8
        Covers all lengths including Angel One's common 26-char case (pad = 6).
        """
        if not secret:
            raise ValueError("TOTP secret is empty")
        secret  = re.sub(r'[^A-Z2-7]', '', secret.upper())
        padding = (8 - len(secret) % 8) % 8
        return secret + '=' * padding

    def _generate_totp(self) -> str:
        return pyotp.TOTP(self._fix_totp_secret(self.totp_secret)).now()

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    def login(self, max_retries: int = 3, retry_delay: int = 5) -> bool:
        if self.is_logged_in and self.obj:
            return True

        self.obj = SmartConnect(api_key=self.api_key)

        for attempt in range(max_retries):
            try:
                logger.info(f"Login attempt {attempt + 1}/{max_retries}")
                otp = self._generate_totp()
                logger.debug(f"TOTP: {otp}")

                try:
                    data = self.obj.generateSession(self.username, self.mpin, otp)
                except TypeError as e:
                    if "'clientCode'" in str(e):
                        data = self.obj.generateSession(
                            self.username, self.mpin, otp, self.username
                        )
                    else:
                        raise

                if data.get('status') and data.get('message') == 'SUCCESS':
                    self.is_logged_in = True
                    logger.info("Login successful")
                    return True

                logger.warning(f"Login failed: {data.get('message', 'Unknown')}")

            except Exception as e:
                logger.error(f"Login attempt {attempt + 1} exception: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        logger.error("Max login retries exceeded")
        return False

    def logout(self) -> None:
        try:
            if self.obj and self.is_logged_in:
                result = self.obj.terminateSession(self.username)
                if result.get('status'):
                    logger.info("Session terminated")
                else:
                    logger.warning(f"Termination warning: {result.get('message')}")
        except Exception as e:
            logger.error(f"Logout error: {e}")
        finally:
            self.is_logged_in = False

    # ------------------------------------------------------------------
    # Token resolution — EXACT MATCH ONLY (no fuzzy fallback)
    # ------------------------------------------------------------------

    def get_token_from_symbol(
        self,
        trading_symbol: str,
        exchange: str = DEFAULT_EXCHANGE,
        max_retries: int = 3,
    ) -> Optional[str]:
        """
        Return the API token for exactly ``trading_symbol`` on ``exchange``.

        If the exact tradingsymbol is absent from search results the method
        returns None and logs the candidates returned, for diagnosis.
        No fuzzy fallback — returning the wrong ticker's token is a
        silent data-corruption bug worse than a clean failure.
        """
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                result = self.obj.searchScrip(exchange, trading_symbol)
                self._update_request_time()

                if result.get('status') and result.get('data'):
                    for item in result['data']:
                        if item.get('tradingsymbol', '') == trading_symbol:
                            token = item['symboltoken']
                            logger.info(f"Token resolved: {trading_symbol} → {token}")
                            return token

                    candidates = [i.get('tradingsymbol') for i in result['data'][:5]]
                    logger.error(
                        f"No exact match for '{trading_symbol}'. "
                        f"API candidates: {candidates}. "
                        f"Skipping — will NOT use an approximate match."
                    )
                    return None

                error_msg = result.get('message', '')
                if _is_rate_limit_error(error_msg):
                    backoff = (attempt + 1) * 5
                    logger.warning(f"Rate limit (token search). Waiting {backoff}s…")
                    time.sleep(backoff)
                    continue

                logger.error(f"searchScrip error for '{trading_symbol}': {error_msg}")
                return None

            except Exception as e:
                if _is_rate_limit_error(str(e)):
                    backoff = (attempt + 1) * 10
                    logger.warning(f"Rate limit exception. Waiting {backoff}s…")
                    time.sleep(backoff)
                elif attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    logger.error(f"Token resolution failed for '{trading_symbol}': {e}")
                    return None

        return None

    # ------------------------------------------------------------------
    # Historical candle data
    # ------------------------------------------------------------------

    def fetch_historical_data(
        self,
        symbol_token: str,
        exchange: str,
        from_date: datetime,
        to_date: datetime,
        interval: str = "ONE_DAY",
        max_retries: int = 3,
    ) -> Optional[List]:
        """
        Fetch OHLCV candles.

        Both ``from_date`` and ``to_date`` MUST be timezone-aware so the
        fetch window is unambiguous and reproducible across runs.
        """
        if from_date.tzinfo is None or to_date.tzinfo is None:
            raise ValueError(
                "from_date and to_date must be timezone-aware. "
                "Use datetime.now(tz=NSE_TZ) rather than datetime.now()."
            )

        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()

                params = {
                    "exchange":    exchange,
                    "symboltoken": symbol_token,
                    "interval":    interval,
                    "fromdate":    from_date.strftime('%Y-%m-%d %H:%M'),
                    "todate":      to_date.strftime('%Y-%m-%d %H:%M'),
                }

                logger.info(
                    f"Fetching {symbol_token}: {from_date.date()} → {to_date.date()} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                data = self.obj.getCandleData(params)
                self._update_request_time()

                if data.get('status') and data.get('message') == 'SUCCESS':
                    candles = data.get('data', [])
                    logger.info(f"Retrieved {len(candles)} candles")
                    return candles

                error_msg = data.get('message', '')
                if _is_rate_limit_error(error_msg):
                    backoff = (attempt + 1) * 10
                    logger.warning(f"Rate limit (candle fetch). Waiting {backoff}s…")
                    time.sleep(backoff)
                    continue

                logger.error(f"getCandleData error: {error_msg}")
                return None

            except Exception as e:
                if _is_rate_limit_error(str(e)):
                    backoff = (attempt + 1) * 15
                    logger.warning(f"Rate limit exception (candle). Waiting {backoff}s…")
                    time.sleep(backoff)
                elif attempt < max_retries - 1:
                    time.sleep(3)
                else:
                    logger.error(f"fetch_historical_data failed: {e}")
                    return None

        return None


# =============================================================================
# CSV READING
# =============================================================================

def read_symbols_from_csv(
    csv_path: str,
    limit: Optional[int] = None,
    start_from: int = 0,
) -> List[Tuple[str, str]]:
    """
    Parse an NSE Nifty 500 CSV and return (symbol, trading_symbol) pairs.

    Uses iloc-based slicing instead of iterrows() for start_from/limit,
    then a single-pass loop for symbol extraction.
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"CSV columns: {list(df.columns)}")

        col_lower  = {c.lower(): c for c in df.columns}
        symbol_col = col_lower.get('symbol')
        series_col = col_lower.get('series')

        if not symbol_col or not series_col:
            logger.error(f"CSV missing 'Symbol'/'Series'. Found: {list(df.columns)}")
            return []

        # Slice before iterating — avoids scanning rows we'll discard
        df = df.iloc[start_from:]
        if limit:
            df = df.head(limit)

        symbols_data: List[Tuple[str, str]] = []
        for _, row in df.iterrows():
            symbol = str(row[symbol_col]).strip() if pd.notna(row[symbol_col]) else ''
            series = str(row[series_col]).strip() if pd.notna(row[series_col]) else ''

            if not symbol or symbol in _SKIP_SYMBOLS:
                continue

            trading_symbol = f"{symbol}-EQ" if series.upper() == 'EQ' else symbol
            symbols_data.append((symbol, trading_symbol))

        logger.info(
            f"Loaded {len(symbols_data)} symbols "
            f"(start_from={start_from}, limit={limit})"
        )
        return symbols_data

    except Exception as e:
        logger.error(f"Failed to read CSV '{csv_path}': {e}")
        return []


# =============================================================================
# DATAFRAME CONSTRUCTION
# =============================================================================

def build_dataframe(candles: List, symbol: str) -> pd.DataFrame:
    """
    Convert raw candle list to a UTC-aware DataFrame with native datetime dtype.

    Timestamps are stored as pandas Timestamps with UTC tzinfo — NOT as strings.
    This guarantees:
      * Monotonic order validated here is the same property stored in memory.
      * Downstream stage 2 can sort/filter without re-parsing string values.
      * When written to CSV via write_csv_with_checksum(), the format is
        declared in the run manifest so consumers know the exact format.
    """
    df = pd.DataFrame(
        candles,
        columns=["datetime", "open", "high", "low", "close", "volume"],
    )

    # Parse to UTC-aware Timestamps (source feed is NSE/IST; Angel One returns
    # ISO 8601 strings which pd.to_datetime with utc=True normalises to UTC)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.sort_values('datetime').reset_index(drop=True)

    if not df['datetime'].is_monotonic_increasing:
        raise ValueError(
            f"{symbol}: timestamps are not monotonically increasing after sort — "
            "possible duplicate or reversed candles from API"
        )

    return df


# =============================================================================
# POST-FETCH VALIDATION
# =============================================================================

def validate_candles(df: pd.DataFrame, symbol: str) -> List[str]:
    """
    Basic sanity checks on freshly fetched candle data.

    Returns a list of warning strings (empty list = all clear).

    Gap detection counts every anomalous gap instance (not just the maximum)
    so that tickers with many small gaps (suspended/stale stocks) are caught,
    while a single legitimate holiday cluster of up to MAX_EXPECTED_CALENDAR_GAP_DAYS
    does not produce false positives.
    """
    warnings: List[str] = []

    if len(df) == 0:
        warnings.append("zero rows returned")
        return warnings

    # OHLCV relationships
    if (df['high'] < df['low']).any():
        warnings.append(f"{(df['high'] < df['low']).sum()} row(s) where high < low")

    if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
        warnings.append("non-positive price(s) detected")

    if (df['volume'] < 0).any():
        warnings.append("negative volume detected")

    # Gap analysis — count every anomalous gap, not just the maximum
    diffs = df['datetime'].diff().dropna()
    anomalous = diffs[diffs.dt.days > MAX_EXPECTED_CALENDAR_GAP_DAYS]
    if not anomalous.empty:
        max_gap = int(anomalous.dt.days.max())
        warnings.append(
            f"{len(anomalous)} gap(s) > {MAX_EXPECTED_CALENDAR_GAP_DAYS} calendar days "
            f"(largest: {max_gap} days) — possible suspension or stale data"
        )

    return warnings


# =============================================================================
# ATOMIC FILE WRITE WITH INTEGRITY CHECK
# =============================================================================

def write_csv_with_checksum(df: pd.DataFrame, output_file: Path) -> str:
    """
    Write DataFrame to CSV atomically and return SHA-256 hex digest.

    Atomicity: os.replace() is atomic on POSIX and Windows (Python 3.3+),
    unlike Path.rename() which raises FileExistsError on Windows when the
    destination exists.

    Integrity check: compare written file byte-length against in-memory bytes.
    This is immune to embedded-newline fields (unlike a line-count check) and
    catches truncated writes (e.g. disk-full) that would silently corrupt
    downstream stages.

    Datetime serialisation: UTC-aware Timestamps are formatted as
    ISO 8601 with explicit '+00:00' offset so CSV consumers can parse
    unambiguously. The exact format is declared in the run manifest.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df_out = df.copy()
    if pd.api.types.is_datetime64_any_dtype(df_out['datetime']):
        df_out['datetime'] = df_out['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')

    csv_bytes = df_out.to_csv(index=False).encode('utf-8')
    checksum  = hashlib.sha256(csv_bytes).hexdigest()

    tmp_file = output_file.with_suffix('.tmp')
    try:
        tmp_file.write_bytes(csv_bytes)
        os.replace(tmp_file, output_file)   # atomic on POSIX + Windows
    except Exception:
        tmp_file.unlink(missing_ok=True)
        raise

    # Byte-length verification (line-count is fragile with embedded newlines)
    written_size = output_file.stat().st_size
    if written_size != len(csv_bytes):
        raise IOError(
            f"Write verification failed for '{output_file.name}': "
            f"expected {len(csv_bytes)} bytes, got {written_size}"
        )

    return checksum


# =============================================================================
# MANIFEST  (schema declaration + run parameters + per-file checksums)
# =============================================================================

_DATETIME_FORMAT   = '%Y-%m-%dT%H:%M:%S+00:00'
_DATETIME_TIMEZONE = 'UTC'


def write_manifest(
    as_of_date: datetime,
    historical_days: int,
    exchange: str,
) -> Path:
    """
    Write a JSON manifest declaring run parameters and CSV schema.

    The manifest lets any downstream stage reconstruct the exact fetch
    window, timezone, and datetime format without hardcoding assumptions
    or re-inferring from file contents.

    Checksums are populated incrementally by update_manifest_checksum()
    so the manifest is useful even after a partial run.
    """
    manifest = {
        "schema_version":    1,
        "script":            Path(__file__).name,
        "produced_at_utc":   datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        "parameters": {
            "as_of_date":      str(as_of_date.date()),
            "historical_days": historical_days,
            "exchange":        exchange,
        },
        "csv_schema": {
            "datetime_column":   "datetime",
            "datetime_format":   _DATETIME_FORMAT,
            "datetime_timezone": _DATETIME_TIMEZONE,
            "columns":           ["datetime", "open", "high", "low", "close", "volume"],
            "notes": (
                "Timestamps are UTC. "
                "Source feed is NSE (IST = UTC+5:30); "
                "conversion to UTC applied at fetch time by pd.to_datetime(utc=True)."
            ),
        },
        "checksums": {},    # populated by update_manifest_checksum()
    }

    manifest_file = STOCKS_DIR / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))
    return manifest_file


def update_manifest_checksum(symbol_stem: str, checksum: str) -> None:
    """
    Append a SHA-256 checksum entry for one ticker to the manifest.

    The checksum is also persisted here (not only logged) so later runs
    can verify file integrity without re-downloading.
    """
    manifest_file = STOCKS_DIR / "manifest.json"
    if not manifest_file.exists():
        return
    manifest = json.loads(manifest_file.read_text())
    manifest["checksums"][symbol_stem] = checksum
    manifest_file.write_text(json.dumps(manifest, indent=2))


# =============================================================================
# EXISTING FILE CHECK
# =============================================================================

def get_existing_stems(stocks_dir: Path) -> set:
    """Return sanitised stems of already-downloaded CSV files."""
    if not stocks_dir.exists():
        return set()
    return {f.stem for f in stocks_dir.glob("*.csv")}


# =============================================================================
# FETCH ONE TICKER  (orchestrator — each concern is a separately testable unit)
# =============================================================================

def fetch_single_ticker(
    client: AngelOneClient,
    symbol: str,
    trading_symbol: str,
    as_of_date: datetime,
    historical_days: int = DEFAULT_HISTORICAL_DAYS,
    exchange: str = DEFAULT_EXCHANGE,
) -> Tuple[bool, str, int]:
    """
    Orchestrate the full fetch pipeline for a single ticker.

    Steps (each delegated to a testable unit function):
      1. Token resolution      → client.get_token_from_symbol()
      2. Date window           → timedelta arithmetic on pinned as_of_date
      3. Candle fetch          → client.fetch_historical_data()
      4. DataFrame construction → build_dataframe()
      5. Validation            → validate_candles()
      6. Atomic write          → write_csv_with_checksum()
      7. Manifest update       → update_manifest_checksum()

    Returns:
        (success: bool, message: str, row_count: int)
    """
    # 1. Token — exact match only
    token = client.get_token_from_symbol(trading_symbol, exchange)
    if not token:
        return False, f"Exact token not found for '{trading_symbol}'", 0

    # 2. Date window (both ends timezone-aware)
    to_date   = as_of_date
    from_date = to_date - timedelta(days=historical_days)

    # 3. Candles
    candles = client.fetch_historical_data(token, exchange, from_date, to_date)
    if candles is None:
        return False, "API returned no data after retries", 0

    # 4. Build DataFrame
    try:
        df = build_dataframe(candles, symbol)
    except Exception as e:
        return False, f"DataFrame construction failed: {e}", 0

    # 5. Validate (warnings only — do not drop rows here; that is stage 2's job)
    for warning in validate_candles(df, symbol):
        logger.warning(f"{symbol}: {warning}")

    if len(df) == 0:
        return False, "Zero valid rows after parsing", 0

    # 6. Write atomically
    output_file = STOCKS_DIR / f"{sanitise_symbol(symbol)}.csv"
    try:
        checksum = write_csv_with_checksum(df, output_file)
    except Exception as e:
        return False, f"File write failed: {e}", 0

    logger.info(
        f"{symbol}: {len(df)} rows → {output_file.name} "
        f"(SHA-256: {checksum[:16]}…)"
    )

    # 7. Persist checksum in manifest
    update_manifest_checksum(sanitise_symbol(symbol), checksum)

    return True, "SUCCESS", len(df)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Fetch historical OHLCV data from Angel One API (deterministic)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 1_fetch.py
  python3 1_fetch.py --as-of-date 2024-01-01   # recommended for backtests
  python3 1_fetch.py --limit 50 --skip-existing
  python3 1_fetch.py --start-from 100 --verbose
        """,
    )
    parser.add_argument('--csv-file', default=str(DEFAULT_CSV_PATH))
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--start-from', type=int, default=0)
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--days', type=int, default=DEFAULT_HISTORICAL_DAYS)
    parser.add_argument('--exchange', default=DEFAULT_EXCHANGE)
    parser.add_argument('--requests-per-minute', type=int,
                        default=DEFAULT_REQUESTS_PER_MINUTE)
    parser.add_argument(
        '--as-of-date', default=None, metavar='YYYY-MM-DD',
        help=(
            'Pin fetch end-date (YYYY-MM-DD) for reproducibility. '
            'If omitted, uses today at NSE close — non-reproducible, '
            'NOT recommended for backtests.'
        ),
    )
    parser.add_argument('--verbose', action='store_true',
                        help='Enable DEBUG-level logging')

    args = parser.parse_args()

    # Logging is set up here, not at module level, so importing this module
    # in a test suite does not create log files or directories.
    global logger
    logger = setup_logging(verbose=args.verbose)

    # ------------------------------------------------------------------
    # Resolve as-of date
    # ------------------------------------------------------------------
    if args.as_of_date:
        try:
            naive = datetime.strptime(args.as_of_date, '%Y-%m-%d')
            # Attach NSE close-of-business time (15:30 IST)
            as_of_date = naive.replace(hour=15, minute=30, tzinfo=NSE_TZ)
        except ValueError:
            print(f"ERROR: --as-of-date must be YYYY-MM-DD, got '{args.as_of_date}'")
            sys.exit(1)
    else:
        as_of_date = datetime.now(tz=NSE_TZ).replace(
            hour=15, minute=30, second=0, microsecond=0
        )
        logger.warning(
            "No --as-of-date supplied. Fetch window ends at today "
            f"({as_of_date.date()} IST). Results WILL differ across run days. "
            "Pass --as-of-date YYYY-MM-DD to pin the window for reproducibility."
        )

    print("\n" + "=" * 60)
    print("STEP 1: FETCH DATA FROM ANGEL ONE API")
    print("=" * 60)
    print(f"  As-of date  : {as_of_date.date()} IST")
    print(f"  Look-back   : {args.days} calendar days")
    print(f"  Output      : {STOCKS_DIR}")

    # ------------------------------------------------------------------
    # Read symbols
    # ------------------------------------------------------------------
    if not Path(args.csv_file).exists():
        print(f"\nERROR: CSV not found: {args.csv_file}")
        sys.exit(1)

    print(f"\n  Reading symbols from: {args.csv_file}")
    symbols_data = read_symbols_from_csv(args.csv_file, args.limit, args.start_from)

    if not symbols_data:
        print("\nERROR: No symbols loaded from CSV")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Skip existing (uses same sanitise_symbol as writer — no mismatch)
    # ------------------------------------------------------------------
    if args.skip_existing:
        existing = get_existing_stems(STOCKS_DIR)
        before   = len(symbols_data)
        symbols_data = [
            (sym, ts) for sym, ts in symbols_data
            if sanitise_symbol(sym) not in existing
        ]
        print(f"  Skipped {before - len(symbols_data)} existing, "
              f"{len(symbols_data)} remaining")

    if not symbols_data:
        print("\nAll symbols already downloaded.")
        sys.exit(0)

    estimated_minutes = (len(symbols_data) * 2) / args.requests_per_minute
    print(f"\n  Symbols to fetch : {len(symbols_data)}")
    print(f"  Estimated time   : ~{estimated_minutes:.1f} minutes\n")

    # ------------------------------------------------------------------
    # Write manifest before fetching so it exists even on early failure
    # ------------------------------------------------------------------
    manifest_file = write_manifest(
        as_of_date=as_of_date,
        historical_days=args.days,
        exchange=args.exchange,
    )
    logger.info(f"Manifest written: {manifest_file}")

    # ------------------------------------------------------------------
    # Connect
    # ------------------------------------------------------------------
    client = AngelOneClient(requests_per_minute=args.requests_per_minute)
    if not client.login():
        print("\nERROR: Login failed. Check credentials in angel.env.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Fetch loop
    # ------------------------------------------------------------------
    successful: List[dict] = []
    failed:     List[dict] = []

    try:
        for i, (symbol, trading_symbol) in enumerate(symbols_data, start=1):
            print(f"  [{i}/{len(symbols_data)}] {symbol}…", end=" ", flush=True)

            ok, msg, rows = fetch_single_ticker(
                client, symbol, trading_symbol,
                as_of_date=as_of_date,
                historical_days=args.days,
                exchange=args.exchange,
            )

            record = {
                'symbol':    symbol,
                'rows':      rows,
                'status':    'success' if ok else 'failed',
                'message':   msg,
                'timestamp': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            }
            (successful if ok else failed).append(record)
            print(f"✓ ({rows} rows)" if ok else f"✗ ({msg})")

    finally:
        client.logout()

    # ------------------------------------------------------------------
    # Persist download report — BOTH successes and failures (fix: v1 lost failures)
    # ------------------------------------------------------------------
    report_df   = pd.DataFrame(successful + failed)
    report_file = REPORT_DIR / "download_report.csv"
    report_df.to_csv(report_file, index=False)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("FETCH SUMMARY")
    print("-" * 60)
    print(f"  Total processed : {len(symbols_data)}")
    print(f"  ✓ Successful    : {len(successful)}")
    print(f"  ✗ Failed        : {len(failed)}")

    if failed:
        print("\n  Failed tickers (first 20):")
        for rec in failed[:20]:
            print(f"    - {rec['symbol']}: {rec['message']}")
        if len(failed) > 20:
            print(f"    … and {len(failed) - 20} more (see download_report.csv)")

    print(f"\n  Report   : {report_file}")
    print(f"  Manifest : {manifest_file}")
    print(f"  Stocks   : {STOCKS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
