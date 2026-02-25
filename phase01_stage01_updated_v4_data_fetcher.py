#!/usr/bin/env python3
"""
phases/phase1/1_fetch.py
========================

Step 1: Fetch historical OHLCV data from Angel One API.

Reads tickers from Nifty 500 CSV file and downloads historical data.
All output paths derive from PIPELINE_STAGE_ID — decoupled from filename.

Design principles:
  * Stage 1 collects maximally; Stage 2 cleans. No data is discarded here
    except on unrecoverable API failures. Duplicate timestamps are de-duped
    with a warning rather than raising.
  * Every output is auditable: manifest.json declares schema + run params,
    download_report.csv records every ticker outcome (success AND failure),
    and per-file SHA-256 checksums are persisted in the manifest.
  * Temporal integrity: --as-of-date pins the fetch window; all datetimes
    are UTC-aware in memory and written as ISO 8601+00:00 on disk.

Usage:
    python3 1_fetch.py
    python3 1_fetch.py --as-of-date 2024-01-01   # required for reproducibility
    python3 1_fetch.py --limit 50 --skip-existing
    python3 1_fetch.py --start-from 100 --verbose

Output (all under <script_dir>/<PIPELINE_STAGE_ID>_output/):
    temp_stocks/<symbol>.csv        -- OHLCV, UTC timestamps (ISO 8601+00:00)
    temp_stocks/manifest.json       -- run parameters, CSV schema, checksums
    report/download_report.csv      -- per-ticker outcome audit log
    logs/fetch_<timestamp>.log      -- full execution log
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
# PIPELINE IDENTITY  -- decouples output paths from the script filename
# =============================================================================
# Rename the script (e.g. to "1_fetch.py") without changing where outputs land.

PIPELINE_STAGE_ID = "phase01_stage01"

_SCRIPT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = _SCRIPT_DIR / f"{PIPELINE_STAGE_ID}_output"
STOCKS_DIR = OUTPUT_DIR / "temp_stocks"
LOGS_DIR   = OUTPUT_DIR / "logs"
REPORT_DIR = OUTPUT_DIR / "report"

for _d in (OUTPUT_DIR, STOCKS_DIR, LOGS_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DEFAULT_CSV_PATH = _SCRIPT_DIR / "nifty_500" / "ind_nifty500list.csv"

# =============================================================================
# ENVIRONMENT VARIABLES
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
    print(f"ERROR: Missing dependency -- {_e}")
    print("Run: pip install smartapi-python pyotp")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_HISTORICAL_DAYS     = 2000
DEFAULT_EXCHANGE            = "NSE"
DEFAULT_REQUESTS_PER_MINUTE = 120   # 2 calls/second -- under Angel One's 10/s limit

# NSE timezone (IST = UTC+5:30).
# Fixed-offset is correct: IST has not observed DST since 1945 and our
# look-back of DEFAULT_HISTORICAL_DAYS (~5.5 years) is entirely post-1945.
# For Python 3.9+ and longer look-backs, prefer:
#   from zoneinfo import ZoneInfo; NSE_TZ = ZoneInfo("Asia/Kolkata")
# which serialises as the self-documenting IANA name "Asia/Kolkata".
NSE_TZ = timezone(timedelta(hours=5, minutes=30))

# Gap detection: NSE worst-case holiday cluster ~= Diwali + surrounding weekends
# (~12 consecutive calendar days). Gaps beyond this threshold are anomalous.
MAX_EXPECTED_CALENDAR_GAP_DAYS = 15

# Known non-stock sentinel values in NSE CSV exports.
# DUMMYHDLVR: a placeholder row sometimes injected by NSE data providers to
# signal a delivery-only dummy instrument with no real OHLCV data.
_SKIP_SYMBOLS = frozenset({'nan', 'DUMMYHDLVR'})

# Rate-limit error detection: multi-word phrases only.
# Avoids false positives on innocent messages containing "rate"
# (e.g. "exchange rate", "interest rate update").
# HTTP status codes matched with surrounding spaces to avoid substring matches.
_RATE_LIMIT_PHRASES = (
    'rate limit',
    'too many requests',
    'access denied',
    'http 502',
    ' 502 ',
    'http 429',
    ' 429 ',
)

# CSV datetime format -- UTC, ISO 8601 with explicit offset.
# Declared here and in the manifest so consumers never guess the format.
_DATETIME_FORMAT   = '%Y-%m-%dT%H:%M:%S+00:00'
_DATETIME_TIMEZONE = 'UTC'

# Manifest schema version -- increment when the manifest structure changes.
# Changelog:
#   v1 (initial): as_of_date, historical_days, exchange, csv_schema, checksums,
#                 status ("in_progress" | "complete" | "failed"), completed_at_utc
MANIFEST_SCHEMA_VERSION = 1

# =============================================================================
# LOGGING  (setup deferred to main() -- no side-effects on import/test)
# =============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure root logger. Call from main() only, never at module level."""
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


# Placeholder; reassigned in main() after argument parsing.
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

    Single canonical implementation -- used by both the writer and the
    existing-file checker to prevent skip-existing mismatches.
    """
    s = symbol.replace('&', 'and').replace(' ', '_')
    s = re.sub(r'[<>:"/\\|?*]', '', s)
    return s.lower()

# =============================================================================
# RATE-LIMIT DETECTION
# =============================================================================

def _is_rate_limit_error(msg: str) -> bool:
    """True if msg signals a rate-limit or gateway error."""
    m = msg.lower()
    return any(phrase in m for phrase in _RATE_LIMIT_PHRASES)

# =============================================================================
# ANGEL ONE API CLIENT
# =============================================================================

class AngelOneClient:
    """
    Angel One SmartAPI client with rate limiting, retry/backoff, and
    strict exact-match token resolution.

    Key guarantees
    --------------
    * Rate limiting uses time.monotonic() -- immune to wall-clock adjustments.
    * Token resolution is EXACT-MATCH ONLY. No fuzzy fallback that could
      silently return a different ticker's data.
    * fetch_historical_data() requires timezone-aware datetimes so the fetch
      window is unambiguous and reproducible across run days.
    """

    def __init__(self, requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE):
        self.api_key      = os.getenv('ANGEL_API_KEY')
        self.secret_key   = os.getenv('ANGEL_SECRET_KEY')
        self.username     = os.getenv('ANGEL_USERNAME')
        self.mpin         = os.getenv('ANGEL_MPIN')
        self.totp_secret  = os.getenv('ANGEL_TOTP_SECRET')
        self._check_credentials()

        self.obj            = None
        self.is_logged_in   = False
        self.min_interval   = 60.0 / requests_per_minute
        self._last_req_mono = 0.0

    # ------------------------------------------------------------------
    # Credentials
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
            print(
                f"\n{'='*60}\n"
                f"ERROR: Missing environment variables:\n{lines}\n"
                f"Create angel.env next to this script with the above keys.\n"
                f"{'='*60}"
            )
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
    # TOTP  (RFC 4648 Section 6 -- Base32 padding)
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_totp_secret(secret: str) -> str:
        """
        Normalise a Base32 TOTP secret per RFC 4648 Section 6.

        Valid Base32 strings must be a multiple of 8 characters.
        Generic formula: padding = (8 - len % 8) % 8
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
    # Token resolution -- EXACT MATCH ONLY
    # ------------------------------------------------------------------

    def get_token_from_symbol(
        self,
        trading_symbol: str,
        exchange: str = DEFAULT_EXCHANGE,
        max_retries: int = 3,
    ) -> Optional[str]:
        """
        Return the API token for exactly ``trading_symbol`` on ``exchange``.

        Returns None (never raises) if the symbol is not found.
        No fuzzy fallback -- returning the wrong token is a silent data-
        corruption bug worse than a clean failure.
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
                            logger.info(f"Token resolved: {trading_symbol} -> {token}")
                            return token

                    candidates = [i.get('tradingsymbol') for i in result['data'][:5]]
                    logger.error(
                        f"No exact match for '{trading_symbol}'. "
                        f"API candidates: {candidates}. "
                        f"Skipping -- will NOT use an approximate match."
                    )
                    return None

                error_msg = result.get('message', '')
                if _is_rate_limit_error(error_msg):
                    backoff = (attempt + 1) * 5
                    logger.warning(f"Rate limit (token search). Waiting {backoff}s...")
                    time.sleep(backoff)
                    continue

                logger.error(f"searchScrip error for '{trading_symbol}': {error_msg}")
                return None

            except Exception as e:
                if _is_rate_limit_error(str(e)):
                    backoff = (attempt + 1) * 10
                    logger.warning(f"Rate limit exception (token). Waiting {backoff}s...")
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
        fetch window is unambiguous and reproducible across run days.
        """
        if from_date.tzinfo is None or to_date.tzinfo is None:
            raise ValueError(
                "from_date and to_date must be timezone-aware. "
                "Pass datetime objects with tzinfo (e.g. NSE_TZ or timezone.utc)."
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
                    f"Fetching {symbol_token}: {from_date.date()} -> {to_date.date()} "
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
                    logger.warning(f"Rate limit (candle fetch). Waiting {backoff}s...")
                    time.sleep(backoff)
                    continue

                logger.error(f"getCandleData error: {error_msg}")
                return None

            except Exception as e:
                if _is_rate_limit_error(str(e)):
                    backoff = (attempt + 1) * 15
                    logger.warning(f"Rate limit exception (candle). Waiting {backoff}s...")
                    time.sleep(backoff)
                elif attempt < max_retries - 1:
                    time.sleep(3)
                else:
                    logger.error(f"fetch_historical_data failed: {e}")
                    return None

        return None


# =============================================================================
# CSV READING  (fully vectorised -- no iterrows)
# =============================================================================

def read_symbols_from_csv(
    csv_path: str,
    limit: Optional[int] = None,
    start_from: int = 0,
) -> List[Tuple[str, str]]:
    """
    Parse an NSE Nifty 500 CSV and return (symbol, trading_symbol) pairs.

    Fully vectorised: uses iloc slicing and pandas string operations.
    No iterrows() or manual row skipping.
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

        # Slice before any processing
        df = df.iloc[start_from:]
        if limit:
            df = df.head(limit)

        symbols = df[symbol_col].astype(str).str.strip()
        series  = df[series_col].astype(str).str.strip()

        valid_mask = ~symbols.isin(_SKIP_SYMBOLS) & symbols.ne('') & symbols.ne('nan')
        symbols = symbols[valid_mask].reset_index(drop=True)
        series  = series[valid_mask].reset_index(drop=True)

        # Vectorised trading symbol construction
        trading = symbols.where(
            series.str.upper() != 'EQ',
            symbols + '-EQ',
        )

        result = list(zip(symbols.tolist(), trading.tolist()))
        logger.info(
            f"Loaded {len(result)} symbols "
            f"(start_from={start_from}, limit={limit})"
        )
        return result

    except Exception as e:
        logger.error(f"Failed to read CSV '{csv_path}': {e}")
        return []


# =============================================================================
# DATAFRAME CONSTRUCTION
# =============================================================================

def build_dataframe(candles: List, symbol: str) -> pd.DataFrame:
    """
    Convert raw candle list to a UTC-aware DataFrame.

    POSTCONDITION: df['datetime'] is UTC-aware pd.Timestamp dtype,
    sorted ascending, with duplicate timestamps removed (keeping first).

    Stage 1 collects maximally: duplicate candles from the API are de-duplicated
    with a warning rather than raising, so a ticker with one bad API candle is
    not permanently skipped. Raising here would be the wrong gate -- Stage 2
    owns data quality decisions beyond what is strictly necessary for parsing.
    """
    df = pd.DataFrame(
        candles,
        columns=["datetime", "open", "high", "low", "close", "volume"],
    )

    # Parse to UTC-aware Timestamps (Angel One returns ISO 8601 strings)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.sort_values('datetime').reset_index(drop=True)

    # De-duplicate: warn and keep first occurrence rather than raising
    dupes = df['datetime'].duplicated()
    if dupes.any():
        logger.warning(
            f"{symbol}: {dupes.sum()} duplicate timestamp(s) from API -- "
            f"keeping first occurrence, discarding rest"
        )
        df = df[~dupes].reset_index(drop=True)

    return df


# =============================================================================
# POST-FETCH VALIDATION
# =============================================================================

def validate_candles(df: pd.DataFrame, symbol: str) -> List[str]:
    """
    Sanity checks on freshly fetched candle data.

    Returns a list of warning strings (empty = all clear).
    This function NEVER modifies df -- Stage 2 owns remediation.

    OHLCV completeness: all six relationships checked
      (open/close within [low, high], high >= low, positive prices, non-negative volume).

    Gap detection: counts every anomalous gap instance so suspended stocks
    with many small gaps are caught alongside single large holiday clusters.

    PRECONDITION: df['datetime'] must be UTC-aware pd.Timestamp dtype
    (as returned by build_dataframe()).
    """
    warnings_out: List[str] = []

    if len(df) == 0:
        warnings_out.append("zero rows returned")
        return warnings_out

    # Complete OHLCV relationship checks (all six bounds)
    invalid_ohlc = (
        (df['close'] > df['high']) |
        (df['close'] < df['low'])  |
        (df['open']  > df['high']) |
        (df['open']  < df['low'])  |
        (df['high']  < df['low'])
    )
    if invalid_ohlc.any():
        warnings_out.append(
            f"{invalid_ohlc.sum()} row(s) with invalid OHLC relationships "
            f"(open/close outside [low, high] or high < low)"
        )

    if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
        warnings_out.append("non-positive price(s) detected")

    if (df['volume'] < 0).any():
        warnings_out.append("negative volume detected")

    # Gap analysis: count each anomalous instance, not just the maximum
    diffs     = df['datetime'].diff().dropna()
    anomalous = diffs[diffs.dt.days > MAX_EXPECTED_CALENDAR_GAP_DAYS]
    if not anomalous.empty:
        max_gap = int(anomalous.dt.days.max())
        warnings_out.append(
            f"{len(anomalous)} gap(s) > {MAX_EXPECTED_CALENDAR_GAP_DAYS} calendar days "
            f"(largest: {max_gap} days) -- possible suspension or stale data"
        )

    return warnings_out


# =============================================================================
# ATOMIC FILE WRITE WITH INTEGRITY CHECK
# =============================================================================

def write_csv_with_checksum(df: pd.DataFrame, output_file: Path) -> str:
    """
    Write DataFrame to CSV atomically and return SHA-256 hex digest.

    PRECONDITION: df['datetime'] must be UTC-aware pd.Timestamp dtype.
    validate_candles() must be called before this function so that the
    validated in-memory representation is what gets written to disk.

    Atomicity: os.replace() is atomic on POSIX and Windows (Python 3.3+),
    unlike Path.rename() which raises FileExistsError on Windows.

    Integrity: byte-length comparison guards against disk-full truncation.
    Line-count comparison is fragile with embedded newlines in field values.

    Datetime serialisation: UTC-aware Timestamps formatted as ISO 8601
    with explicit '+00:00' offset. Format and timezone declared in manifest.
    """
    assert pd.api.types.is_datetime64tz_dtype(df['datetime']), (
        "build_dataframe() postcondition violated: "
        "df['datetime'] must be UTC-aware before calling write_csv_with_checksum(). "
        "Ensure validate_candles() was called on the same df before this call."
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    df_out = df.copy()
    df_out['datetime'] = df_out['datetime'].dt.strftime(_DATETIME_FORMAT)

    csv_bytes = df_out.to_csv(index=False).encode('utf-8')
    checksum  = hashlib.sha256(csv_bytes).hexdigest()

    tmp_file = output_file.with_suffix('.tmp')
    try:
        tmp_file.write_bytes(csv_bytes)
        os.replace(tmp_file, output_file)   # atomic on POSIX + Windows
    except Exception:
        tmp_file.unlink(missing_ok=True)
        raise

    written_size = output_file.stat().st_size
    if written_size != len(csv_bytes):
        raise IOError(
            f"Write verification failed for '{output_file.name}': "
            f"expected {len(csv_bytes)} bytes, got {written_size}"
        )

    return checksum


# =============================================================================
# MANIFEST  (written after login; updated atomically per-ticker)
# =============================================================================

def _write_json_atomic(path: Path, data: dict) -> None:
    """Write a dict to a JSON file atomically via os.replace()."""
    tmp = path.with_suffix('.json.tmp')
    try:
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def write_manifest(
    as_of_date: datetime,
    historical_days: int,
    exchange: str,
    status: str = "in_progress",
) -> Path:
    """
    Write (or overwrite) the JSON manifest for this run.

    Called AFTER login succeeds so the manifest only exists when a real
    fetch run has started. The 'status' field lets consumers distinguish
    in-progress runs from completed ones without parsing the download report.

    Schema version changelog:
      v1 (current): pipeline_stage_id, script, produced_at_utc, status,
                    parameters, csv_schema, checksums, completed_at_utc
    """
    manifest = {
        "schema_version":    MANIFEST_SCHEMA_VERSION,
        "pipeline_stage_id": PIPELINE_STAGE_ID,
        "script":            Path(__file__).name,
        "produced_at_utc":   datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "status":            status,   # "in_progress" | "complete" | "failed"
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
                "Timestamps are UTC. Source feed is NSE (IST = UTC+5:30); "
                "conversion to UTC applied at fetch time via pd.to_datetime(utc=True)."
            ),
        },
        "checksums": {},    # populated by update_manifest_checksum() per-ticker
    }

    manifest_file = STOCKS_DIR / "manifest.json"
    _write_json_atomic(manifest_file, manifest)
    return manifest_file


def update_manifest_checksum(symbol_stem: str, checksum: str) -> None:
    """
    Atomically append a SHA-256 checksum entry to the manifest.

    Uses read-modify-write with atomic os.replace() write to prevent
    manifest corruption from concurrent processes.

    Logs a warning and returns (never raises) on any failure so a single
    disk hiccup does not abort the entire fetch loop via the finally block.
    """
    manifest_file = STOCKS_DIR / "manifest.json"
    try:
        if not manifest_file.exists():
            logger.warning(
                f"Manifest not found when writing checksum for '{symbol_stem}' -- "
                "checksum will not be persisted. File was written correctly."
            )
            return

        manifest = json.loads(manifest_file.read_text())
        manifest["checksums"][symbol_stem] = checksum
        _write_json_atomic(manifest_file, manifest)

    except Exception as e:
        # Never raise -- a checksum write failure must not abort the fetch loop
        logger.warning(
            f"Failed to persist checksum for '{symbol_stem}': {e}. "
            "CSV file was written correctly; only the manifest entry is missing."
        )


def finalise_manifest(status: str) -> None:
    """
    Update manifest status to 'complete' or 'failed' at run end.

    Allows downstream consumers to detect interrupted runs without
    inspecting the download report.
    """
    manifest_file = STOCKS_DIR / "manifest.json"
    try:
        if not manifest_file.exists():
            return
        manifest = json.loads(manifest_file.read_text())
        manifest["status"] = status
        manifest["completed_at_utc"] = datetime.now(timezone.utc).strftime(
            '%Y-%m-%dT%H:%M:%SZ'
        )
        _write_json_atomic(manifest_file, manifest)
    except Exception as e:
        logger.warning(f"Failed to finalise manifest status to '{status}': {e}")


# =============================================================================
# EXISTING FILE CHECK
# =============================================================================

def get_existing_stems(stocks_dir: Path) -> set:
    """Return sanitised stems of already-downloaded CSV files."""
    if not stocks_dir.exists():
        return set()
    return {f.stem for f in stocks_dir.glob("*.csv")}


# =============================================================================
# FETCH ONE TICKER  (orchestrator -- each concern is a separately testable unit)
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

    Steps (each delegated to a testable unit):
      1. Token resolution      -> client.get_token_from_symbol()
      2. Date window           -> timedelta arithmetic on pinned as_of_date
      3. Candle fetch          -> client.fetch_historical_data()
      4. DataFrame             -> build_dataframe()  [de-dupes with warning]
      5. Validation            -> validate_candles() [warns, never modifies df]
      6. Atomic write          -> write_csv_with_checksum()
      7. Manifest update       -> update_manifest_checksum() [warns, never raises]

    Returns:
        (success: bool, message: str, row_count: int)
    """
    # 1. Token -- exact match only
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

    # 4. Build DataFrame (de-dupes timestamps; never raises on data quality)
    try:
        df = build_dataframe(candles, symbol)
    except Exception as e:
        return False, f"DataFrame construction failed: {e}", 0

    # 5. Validate (logs warnings; does NOT modify df -- Stage 2 owns remediation)
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
        f"{symbol}: {len(df)} rows -> {output_file.name} "
        f"(SHA-256: {checksum[:8]}...)"   # 8 hex chars = 32 bits, standard for log display
    )

    # 7. Persist checksum (warns on failure, never raises)
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
            'Pin fetch end-date for reproducibility. '
            'If omitted, uses today at NSE close -- results WILL differ '
            'across run days. Required for backtests.'
        ),
    )
    parser.add_argument('--verbose', action='store_true',
                        help='Enable DEBUG-level logging')

    args = parser.parse_args()

    # Deferred logging setup -- importing this module in tests does not create
    # log files or directories.
    global logger
    logger = setup_logging(verbose=args.verbose)

    # ------------------------------------------------------------------
    # Resolve as-of date
    # ------------------------------------------------------------------
    if args.as_of_date:
        try:
            naive = datetime.strptime(args.as_of_date, '%Y-%m-%d')
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
            "Pass --as-of-date YYYY-MM-DD to pin the window."
        )

    print("\n" + "=" * 60)
    print("STEP 1: FETCH DATA FROM ANGEL ONE API")
    print("=" * 60)
    print(f"  Stage ID    : {PIPELINE_STAGE_ID}")
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
    # Skip existing (uses same sanitise_symbol as writer -- no mismatch)
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
    # Login first -- manifest is written AFTER login succeeds so it only
    # exists when a real fetch run has actually started.
    # ------------------------------------------------------------------
    client = AngelOneClient(requests_per_minute=args.requests_per_minute)
    if not client.login():
        print("\nERROR: Login failed. Check credentials in angel.env.")
        sys.exit(1)

    manifest_file = write_manifest(
        as_of_date=as_of_date,
        historical_days=args.days,
        exchange=args.exchange,
        status="in_progress",
    )
    logger.info(f"Manifest written: {manifest_file}")

    # ------------------------------------------------------------------
    # Fetch loop
    # ------------------------------------------------------------------
    successful: List[dict] = []
    failed:     List[dict] = []
    run_status = "failed"   # updated to "complete" on clean exit

    try:
        for i, (symbol, trading_symbol) in enumerate(symbols_data, start=1):
            print(f"  [{i}/{len(symbols_data)}] {symbol}...", end=" ", flush=True)

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
                'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            }
            (successful if ok else failed).append(record)
            print(f"OK ({rows} rows)" if ok else f"FAIL ({msg})")

        run_status = "complete"

    finally:
        client.logout()
        finalise_manifest(run_status)

    # ------------------------------------------------------------------
    # Persist download report -- BOTH successes and failures
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
    print(f"  Successful      : {len(successful)}")
    print(f"  Failed          : {len(failed)}")
    print(f"  Run status      : {run_status}")

    if failed:
        print("\n  Failed tickers (first 20):")
        for rec in failed[:20]:
            print(f"    - {rec['symbol']}: {rec['message']}")
        if len(failed) > 20:
            print(f"    ... and {len(failed) - 20} more (see download_report.csv)")

    print(f"\n  Report   : {report_file}")
    print(f"  Manifest : {manifest_file}")
    print(f"  Stocks   : {STOCKS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
