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

# Updated to point to the ETF list relative to the script location
DEFAULT_CSV_PATH = _SCRIPT_DIR / "list" / "eq_etfseclist.csv"

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
NSE_TZ = timezone(timedelta(hours=5, minutes=30))

# Gap detection: NSE worst-case holiday cluster ~= Diwali + surrounding weekends
MAX_EXPECTED_CALENDAR_GAP_DAYS = 15

# Known non-stock sentinel values in NSE CSV exports.
_SKIP_SYMBOLS = frozenset({'nan', 'DUMMYHDLVR'})

# Rate-limit error detection
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
_DATETIME_FORMAT   = '%Y-%m-%dT%H:%M:%S+00:00'
_DATETIME_TIMEZONE = 'UTC'

# Manifest schema version
MANIFEST_SCHEMA_VERSION = 1

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure root logger. Call from main() only."""
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

logger: logging.Logger = logging.getLogger(__name__)

# =============================================================================
# SYMBOL SANITISATION
# =============================================================================

def sanitise_symbol(symbol: str) -> str:
    """Convert a raw ticker to a safe, lowercase filesystem stem."""
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
    """Angel One SmartAPI client with rate limiting and retry logic."""

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

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_req_mono
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def _update_request_time(self) -> None:
        self._last_req_mono = time.monotonic()

    @staticmethod
    def _fix_totp_secret(secret: str) -> str:
        """Normalise a Base32 TOTP secret per RFC 4648 Section 6."""
        if not secret:
            raise ValueError("TOTP secret is empty")
        secret  = re.sub(r'[^A-Z2-7]', '', secret.upper())
        padding = (8 - len(secret) % 8) % 8
        return secret + '=' * padding

    def _generate_totp(self) -> str:
        return pyotp.TOTP(self._fix_totp_secret(self.totp_secret)).now()

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

    def get_token_from_symbol(
        self,
        trading_symbol: str,
        exchange: str = DEFAULT_EXCHANGE,
        max_retries: int = 3,
    ) -> Optional[str]:
        """Return the API token for exactly ``trading_symbol`` on ``exchange``."""
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

    def fetch_historical_data(
        self,
        symbol_token: str,
        exchange: str,
        from_date: datetime,
        to_date: datetime,
        interval: str = "ONE_DAY",
        max_retries: int = 3,
    ) -> Optional[List]:
        """Fetch OHLCV candles."""
        if from_date.tzinfo is None or to_date.tzinfo is None:
            raise ValueError("from_date and to_date must be timezone-aware.")

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
# CSV READING
# =============================================================================

def read_symbols_from_csv(
    csv_path: str,
    limit: Optional[int] = None,
    start_from: int = 0,
) -> List[Tuple[str, str]]:
    """
    Parse an NSE CSV and return (symbol, trading_symbol) pairs.
    
    Handles CSVs with or without a 'Series' column. If missing, defaults to 'EQ'.
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"CSV columns: {list(df.columns)}")

        col_lower  = {c.lower(): c for c in df.columns}
        symbol_col = col_lower.get('symbol')
        series_col = col_lower.get('series')

        if not symbol_col:
            logger.error(f"CSV missing 'Symbol'. Found: {list(df.columns)}")
            return []

        # Slice before any processing
        df = df.iloc[start_from:]
        if limit:
            df = df.head(limit)

        symbols = df[symbol_col].astype(str).str.strip()

        # Handle missing Series column (common in ETF lists)
        # Default to 'EQ' (Equity) so we construct symbol-EQ
        if not series_col:
            logger.info("No 'Series' column found in CSV. Defaulting all instruments to 'EQ'.")
            series = pd.Series(['EQ'] * len(df), index=df.index)
        else:
            series = df[series_col].astype(str).str.strip()

        valid_mask = ~symbols.isin(_SKIP_SYMBOLS) & symbols.ne('') & symbols.ne('nan')
        symbols = symbols[valid_mask].reset_index(drop=True)
        series  = series[valid_mask].reset_index(drop=True)

        # Vectorised trading symbol construction
        # If series is NOT 'EQ', keep symbol as is. If 'EQ', append '-EQ'.
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
    """Convert raw candle list to a UTC-aware DataFrame."""
    df = pd.DataFrame(
        candles,
        columns=["datetime", "open", "high", "low", "close", "volume"],
    )

    # Parse to UTC-aware Timestamps
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.sort_values('datetime').reset_index(drop=True)

    # De-duplicate
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
    """Sanity checks on freshly fetched candle data."""
    warnings_out: List[str] = []

    if len(df) == 0:
        warnings_out.append("zero rows returned")
        return warnings_out

    invalid_ohlc = (
        (df['close'] > df['high']) |
        (df['close'] < df['low'])  |
        (df['open']  > df['high']) |
        (df['open']  < df['low'])  |
        (df['high']  < df['low'])
    )
    if invalid_ohlc.any():
        warnings_out.append(
            f"{invalid_ohlc.sum()} row(s) with invalid OHLC relationships"
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
            f"(largest: {max_gap} days)"
        )

    return warnings_out


# =============================================================================
# ATOMIC FILE WRITE WITH INTEGRITY CHECK
# =============================================================================

def write_csv_with_checksum(df: pd.DataFrame, output_file: Path) -> str:
    """Write DataFrame to CSV atomically and return SHA-256 hex digest."""
    assert pd.api.types.is_datetime64tz_dtype(df['datetime']), (
        "df['datetime'] must be UTC-aware before calling write_csv_with_checksum()."
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    df_out = df.copy()
    df_out['datetime'] = df_out['datetime'].dt.strftime(_DATETIME_FORMAT)

    csv_bytes = df_out.to_csv(index=False).encode('utf-8')
    checksum  = hashlib.sha256(csv_bytes).hexdigest()

    tmp_file = output_file.with_suffix('.tmp')
    try:
        tmp_file.write_bytes(csv_bytes)
        os.replace(tmp_file, output_file)
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
# MANIFEST
# =============================================================================

def _write_json_atomic(path: Path, data: dict) -> None:
    """Write a dict to a JSON file atomically."""
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
    """Write (or overwrite) the JSON manifest for this run."""
    manifest = {
        "schema_version":    MANIFEST_SCHEMA_VERSION,
        "pipeline_stage_id": PIPELINE_STAGE_ID,
        "script":            Path(__file__).name,
        "produced_at_utc":   datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "status":            status,
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
        },
        "checksums": {},
    }

    manifest_file = STOCKS_DIR / "manifest.json"
    _write_json_atomic(manifest_file, manifest)
    return manifest_file


def update_manifest_checksum(symbol_stem: str, checksum: str) -> None:
    """Atomically append a SHA-256 checksum entry to the manifest."""
    manifest_file = STOCKS_DIR / "manifest.json"
    try:
        if not manifest_file.exists():
            logger.warning(f"Manifest not found when writing checksum for '{symbol_stem}'")
            return

        manifest = json.loads(manifest_file.read_text())
        manifest["checksums"][symbol_stem] = checksum
        _write_json_atomic(manifest_file, manifest)

    except Exception as e:
        logger.warning(f"Failed to persist checksum for '{symbol_stem}': {e}")


def finalise_manifest(status: str) -> None:
    """Update manifest status to 'complete' or 'failed' at run end."""
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
# FETCH ONE TICKER
# =============================================================================

def fetch_single_ticker(
    client: AngelOneClient,
    symbol: str,
    trading_symbol: str,
    as_of_date: datetime,
    historical_days: int = DEFAULT_HISTORICAL_DAYS,
    exchange: str = DEFAULT_EXCHANGE,
) -> Tuple[bool, str, int]:
    """Orchestrate the full fetch pipeline for a single ticker."""
    token = client.get_token_from_symbol(trading_symbol, exchange)
    if not token:
        return False, f"Exact token not found for '{trading_symbol}'", 0

    to_date   = as_of_date
    from_date = to_date - timedelta(days=historical_days)

    candles = client.fetch_historical_data(token, exchange, from_date, to_date)
    if candles is None:
        return False, "API returned no data after retries", 0

    try:
        df = build_dataframe(candles, symbol)
    except Exception as e:
        return False, f"DataFrame construction failed: {e}", 0

    for warning in validate_candles(df, symbol):
        logger.warning(f"{symbol}: {warning}")

    if len(df) == 0:
        return False, "Zero valid rows after parsing", 0

    output_file = STOCKS_DIR / f"{sanitise_symbol(symbol)}.csv"
    try:
        checksum = write_csv_with_checksum(df, output_file)
    except Exception as e:
        return False, f"File write failed: {e}", 0

    logger.info(f"{symbol}: {len(df)} rows -> {output_file.name} (SHA-256: {checksum[:8]}...)")
    update_manifest_checksum(sanitise_symbol(symbol), checksum)

    return True, "SUCCESS", len(df)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Fetch historical OHLCV data from Angel One API (deterministic)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help='Pin fetch end-date for reproducibility.',
    )
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    global logger
    logger = setup_logging(verbose=args.verbose)

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
        logger.warning("No --as-of-date supplied. Fetch window ends at today.")

    print("\n" + "=" * 60)
    print("STEP 1: FETCH DATA FROM ANGEL ONE API")
    print("=" * 60)
    print(f"  Stage ID    : {PIPELINE_STAGE_ID}")
    print(f"  As-of date  : {as_of_date.date()} IST")
    print(f"  Output      : {STOCKS_DIR}")

    if not Path(args.csv_file).exists():
        print(f"\nERROR: CSV not found: {args.csv_file}")
        sys.exit(1)

    print(f"\n  Reading symbols from: {args.csv_file}")
    symbols_data = read_symbols_from_csv(args.csv_file, args.limit, args.start_from)

    if not symbols_data:
        print("\nERROR: No symbols loaded from CSV")
        sys.exit(1)

    if args.skip_existing:
        existing = get_existing_stems(STOCKS_DIR)
        before   = len(symbols_data)
        symbols_data = [
            (sym, ts) for sym, ts in symbols_data
            if sanitise_symbol(sym) not in existing
        ]
        print(f"  Skipped {before - len(symbols_data)} existing")

    if not symbols_data:
        print("\nAll symbols already downloaded.")
        sys.exit(0)

    print(f"\n  Symbols to fetch : {len(symbols_data)}")

    client = AngelOneClient(requests_per_minute=args.requests_per_minute)
    if not client.login():
        print("\nERROR: Login failed.")
        sys.exit(1)

    write_manifest(
        as_of_date=as_of_date,
        historical_days=args.days,
        exchange=args.exchange,
        status="in_progress",
    )

    successful: List[dict] = []
    failed:     List[dict] = []
    run_status = "failed"

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

    report_df   = pd.DataFrame(successful + failed)
    report_file = REPORT_DIR / "download_report.csv"
    report_df.to_csv(report_file, index=False)

    print("\n" + "-" * 60)
    print("FETCH SUMMARY")
    print("-" * 60)
    print(f"  Total processed : {len(symbols_data)}")
    print(f"  Successful      : {len(successful)}")
    print(f"  Failed          : {len(failed)}")
    print(f"  Report   : {report_file}")
    print("=" * 60)

def get_existing_stems(stocks_dir: Path) -> set:
    """Return sanitised stems of already-downloaded CSV files."""
    if not stocks_dir.exists():
        return set()
    return {f.stem for f in stocks_dir.glob("*.csv")}

if __name__ == "__main__":
    main()
