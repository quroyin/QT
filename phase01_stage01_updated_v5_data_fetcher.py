#!/usr/bin/env python3
"""
phases/phase1/1_fetch.py
========================

Step 1: Fetch historical OHLCV data from Angel One API.

Minimum Python version: 3.8 (uses f-strings, walrus operator is avoided,
from __future__ import annotations for PEP 563 postponed evaluation).

All output paths derive from PIPELINE_STAGE_ID -- decoupled from filename.

Design principles:
  * Stage 1 collects maximally; Stage 2 cleans. No data is discarded here
    except on unrecoverable API failures. Duplicate timestamps are de-duped
    with a warning rather than raising.
  * Every output is auditable: manifest.json declares schema + run params
    (including validation thresholds), download_report.csv records every
    ticker outcome (success AND failure), and per-file SHA-256 checksums
    are persisted atomically in the manifest.
  * Temporal integrity: --as-of-date pins the fetch window; all datetimes
    are UTC-aware in memory and written as ISO 8601+00:00 on disk.

Usage:
    python3 1_fetch.py
    python3 1_fetch.py --as-of-date 2024-01-01   # required for reproducibility
    python3 1_fetch.py --limit 50 --skip-existing
    python3 1_fetch.py --start-from 100 --verbose

Output (all under <script_dir>/phase01_stage01_output/):
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
# Rename this script (e.g. to "1_fetch.py") without changing output locations.

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
#
# Minimum version note: this file requires Python 3.8+.
# For Python 3.9+ with look-backs beyond ~5.5 years, prefer:
#   from zoneinfo import ZoneInfo; NSE_TZ = ZoneInfo("Asia/Kolkata")
# which uses the IANA tz database and serialises as the self-documenting
# name "Asia/Kolkata". The fixed-offset below is correct for all dates
# post-1945 (IST has not observed DST since then), and our default look-back
# of DEFAULT_HISTORICAL_DAYS (~5.5 years) is entirely within that range.
NSE_TZ = timezone(timedelta(hours=5, minutes=30))

# Gap detection threshold included in the manifest so consumers know which
# threshold was applied without reading source code.
# Justification: NSE worst-case holiday cluster ~= Diwali + surrounding
# weekends (~12 consecutive calendar days). 15 gives a 3-day margin.
MAX_EXPECTED_CALENDAR_GAP_DAYS = 15

# Known non-stock sentinel values in NSE CSV exports.
# DUMMYHDLVR: placeholder row sometimes injected by NSE data providers to
# signal a delivery-only dummy instrument with no real OHLCV data.
_SKIP_SYMBOLS = frozenset({'nan', 'DUMMYHDLVR'})

# Rate-limit error detection.
# We match on the JSON message body since SmartConnect abstracts the HTTP
# layer and we cannot reliably inspect raw HTTP status codes. Bare numeric
# strings ('429', '502', '503') are included because Angel One returns
# messages like {"message": "Error code 429 returned"} and {"errorCode": "429"}.
# A false positive (throttling when we didn't need to) is far less dangerous
# than a missed rate-limit that hammers the API.
_RATE_LIMIT_PHRASES = (
    'rate limit',
    'too many requests',
    'access denied',
    '429',
    '502',
    '503',
)

# CSV datetime format -- UTC, ISO 8601 with explicit offset.
# Declared here and in the manifest so consumers never guess the format.
_DATETIME_FORMAT   = '%Y-%m-%dT%H:%M:%S+00:00'
_DATETIME_TIMEZONE = 'UTC'

# Manifest schema version.
# Changelog:
#   v1 (current): pipeline_stage_id (str, required), script (str, required),
#     produced_at_utc (str ISO8601, required), status (str, required),
#     completed_at_utc (str ISO8601, optional -- set by finalise_manifest),
#     parameters.as_of_date (str YYYY-MM-DD, required),
#     parameters.historical_days (int, required),
#     parameters.exchange (str, required),
#     parameters.max_gap_days (int, required),
#     csv_schema.datetime_column (str, required),
#     csv_schema.datetime_format (str, required),
#     csv_schema.datetime_timezone (str, required),
#     csv_schema.columns (list[str], required),
#     csv_schema.notes (str, required),
#     checksums (dict[str, str], required -- populated incrementally)
MANIFEST_SCHEMA_VERSION = 1

# =============================================================================
# LOGGING  (setup deferred to main() -- no side-effects on import/test)
# =============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Configure the root logger and return the module logger.

    Must be called from main() only. Calling at module level would create
    log files when the module is imported by test runners.

    Because getLogger(__name__) inherits from the root logger, configuring
    the root logger here automatically propagates to the module-level
    placeholder -- no `global logger` reassignment required.
    """
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


# Module-level logger. Inherits from root -- setup_logging() in main()
# configures the root logger which propagates here automatically.
# No `global` reassignment needed.
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
    """
    True if msg signals a rate-limit or gateway error.

    Matches against _RATE_LIMIT_PHRASES in the JSON message body.
    Bare numeric codes ('429', '502', '503') are included because Angel One
    embeds them in message strings rather than surfacing HTTP status codes
    through the SmartConnect abstraction layer.
    """
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

        df = df.iloc[start_from:]
        if limit:
            df = df.head(limit)

        symbols = df[symbol_col].astype(str).str.strip()
        series  = df[series_col].astype(str).str.strip()

        valid_mask = ~symbols.isin(_SKIP_SYMBOLS) & symbols.ne('') & symbols.ne('nan')
        symbols = symbols[valid_mask].reset_index(drop=True)
        series  = series[valid_mask].reset_index(drop=True)

        # pd.Series.where(cond, other):
        #   keeps original value where cond is True
        #   substitutes `other`         where cond is False
        # Condition: series is NOT 'EQ' -> keep bare symbol (no suffix).
        # Else (series IS 'EQ')         -> substitute symbol + '-EQ'.
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

    POSTCONDITION:
      * df['datetime'] is UTC-aware pd.Timestamp dtype (datetime64[ns, UTC]).
      * Rows are sorted ascending by datetime.
      * Duplicate datetimes are removed: the first occurrence in post-sort
        (i.e. post-UTC-normalisation) order is kept. "First" is deterministic
        because the sort is stable and the UTC conversion is bijective.
        For a quant pipeline: if two API candles share a timestamp after UTC
        normalisation, both values are equally suspect; keeping the first
        is a conservative, auditable choice.

    Design: duplicate candles are de-duplicated with a warning rather than
    raising. Stage 1 collects maximally; Stage 2 owns data quality decisions.
    """
    df = pd.DataFrame(
        candles,
        columns=["datetime", "open", "high", "low", "close", "volume"],
    )

    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.sort_values('datetime').reset_index(drop=True)

    dupes = df['datetime'].duplicated(keep='first')
    if dupes.any():
        logger.warning(
            f"{symbol}: {dupes.sum()} duplicate timestamp(s) from API -- "
            f"keeping first occurrence (post-UTC-sort order), discarding rest"
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

    OHLCV completeness: all six OHLC bounds are checked:
      close within [low, high], open within [low, high], high >= low.

    Gap detection: counts every anomalous gap so suspended stocks with many
    small gaps are caught alongside a single large holiday cluster.

    PRECONDITION: df['datetime'] must be UTC-aware pd.Timestamp dtype
    (as returned by build_dataframe()).
    """
    warnings_out: List[str] = []

    if len(df) == 0:
        warnings_out.append("zero rows returned")
        return warnings_out

    # All six OHLC relationship bounds
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

    PRECONDITION: df['datetime'] must be UTC-aware pd.Timestamp dtype,
    as produced by build_dataframe(). The assert below enforces this contract
    explicitly so that a future call-order change surfaces immediately rather
    than silently writing non-UTC data.

    Atomicity: os.replace() is atomic on POSIX and Windows (Python 3.3+).
    Integrity: byte-length comparison guards against disk-full truncation.
    Datetime serialisation: formatted as _DATETIME_FORMAT; declared in manifest.
    """
    assert pd.api.types.is_datetime64tz_dtype(df['datetime']), (
        "Precondition violated: df['datetime'] must be UTC-aware pd.Timestamp. "
        "Call build_dataframe() before write_csv_with_checksum()."
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    df_out = df.copy()
    df_out['datetime'] = df_out['datetime'].dt.strftime(_DATETIME_FORMAT)

    csv_bytes = df_out.to_csv(index=False).encode('utf-8')
    checksum  = hashlib.sha256(csv_bytes).hexdigest()

    tmp_file = output_file.parent / (output_file.name + '.tmp')
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


def write_csv_atomic(path: Path, content: bytes) -> None:
    """
    Write raw bytes to a file atomically via os.replace().
    Used for the download report CSV to match the same integrity guarantee
    as write_csv_with_checksum().
    """
    tmp = path.parent / (path.name + '.tmp')
    try:
        tmp.write_bytes(content)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# =============================================================================
# MANIFEST  (written after login; updated atomically per-ticker)
# =============================================================================

def _write_json_atomic(path: Path, data: dict) -> None:
    """
    Write a dict to a JSON file atomically via os.replace().

    Temp file is placed in the same directory as the target so that
    os.replace() is guaranteed to be atomic (same filesystem).

    Naming: path.parent / (path.name + '.tmp') is explicit and avoids
    Path.with_suffix() which replaces the last extension component and
    can surprise readers unfamiliar with that behaviour.
    """
    tmp = path.parent / (path.name + '.tmp')
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

    MAX_EXPECTED_CALENDAR_GAP_DAYS is included in parameters so consumers
    can reproduce the validation threshold without reading source code.

    Schema version changelog: see MANIFEST_SCHEMA_VERSION above.
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
            "max_gap_days":    MAX_EXPECTED_CALENDAR_GAP_DAYS,
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

    Uses read-modify-write with _write_json_atomic() (os.replace()) to
    prevent manifest corruption from concurrent processes.

    Logs a warning and returns (never raises) on any failure so a disk
    hiccup does not abort the fetch loop via the finally block.
    """
    manifest_file = STOCKS_DIR / "manifest.json"
    try:
        if not manifest_file.exists():
            logger.warning(
                f"Manifest not found when writing checksum for '{symbol_stem}' -- "
                "checksum will not be persisted. CSV file was written correctly."
            )
            return

        manifest = json.loads(manifest_file.read_text())
        manifest["checksums"][symbol_stem] = checksum
        _write_json_atomic(manifest_file, manifest)

    except Exception as e:
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
# PER-TICKER TIMEOUT  (cross-platform via concurrent.futures)
# =============================================================================

def _fetch_with_timeout(
    client: AngelOneClient,
    symbol: str,
    trading_symbol: str,
    as_of_date: datetime,
    historical_days: int,
    exchange: str,
    timeout_seconds: int = 120,
) -> Tuple[bool, str, int]:
    """
    Run fetch_single_ticker with a wall-clock timeout.

    Uses concurrent.futures.ThreadPoolExecutor for cross-platform support.
    signal.SIGALRM would work on POSIX only; ThreadPoolExecutor works on
    both POSIX and Windows.

    If the ticker does not complete within timeout_seconds, returns a
    failure tuple so the main loop continues rather than hanging indefinitely.

    Worst-case without timeout: token retries (3 x 10s) + candle retries
    (3 x 15s) = ~75s per failed ticker. For 500 tickers with 50 failures
    this adds ~62 minutes of dead wait time.
    """
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            fetch_single_ticker,
            client, symbol, trading_symbol,
            as_of_date, historical_days, exchange,
        )
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            logger.error(
                f"{symbol}: fetch timed out after {timeout_seconds}s -- "
                "skipping ticker"
            )
            return False, f"Timed out after {timeout_seconds}s", 0
        except Exception as e:
            logger.error(f"{symbol}: unexpected error in fetch thread: {e}")
            return False, f"Unexpected error: {e}", 0


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
    parser.add_argument(
        '--ticker-timeout', type=int, default=120,
        help='Wall-clock timeout in seconds per ticker (default: 120). '
             'Prevents a stalled API call from hanging the entire run.'
    )
    parser.add_argument('--verbose', action='store_true',
                        help='Enable DEBUG-level logging')

    args = parser.parse_args()

    # setup_logging() configures the root logger, which the module-level
    # logger inherits from automatically -- no `global` reassignment needed.
    setup_logging(verbose=args.verbose)

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
    print(f"  Stage ID      : {PIPELINE_STAGE_ID}")
    print(f"  As-of date    : {as_of_date.date()} IST")
    print(f"  Look-back     : {args.days} calendar days")
    print(f"  Ticker timeout: {args.ticker_timeout}s")
    print(f"  Output        : {STOCKS_DIR}")

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
    # Login first -- manifest written AFTER login so it only exists
    # when a real fetch run has actually started.
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
    run_status = "failed"   # updated to "complete" on clean loop exit

    try:
        for i, (symbol, trading_symbol) in enumerate(symbols_data, start=1):
            print(f"  [{i}/{len(symbols_data)}] {symbol}...", end=" ", flush=True)

            ok, msg, rows = _fetch_with_timeout(
                client, symbol, trading_symbol,
                as_of_date=as_of_date,
                historical_days=args.days,
                exchange=args.exchange,
                timeout_seconds=args.ticker_timeout,
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
    # Persist download report atomically -- BOTH successes and failures.
    # Same atomic write pattern as CSV and manifest files.
    # ------------------------------------------------------------------
    report_df    = pd.DataFrame(successful + failed)
    report_bytes = report_df.to_csv(index=False).encode('utf-8')
    report_file  = REPORT_DIR / "download_report.csv"
    write_csv_atomic(report_file, report_bytes)

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
