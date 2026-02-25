#!/usr/bin/env python3
"""
phases/phase1/1_fetch.py
========================

Step 1: Fetch historical OHLCV data from Angel One API.

Reads tickers from Nifty 500 CSV file and downloads historical data.
All output paths are derived dynamically from __file__ — never hardcoded.

Usage:
    # Default (fetch all tickers from default CSV)
    python3 phases/phase1/1_fetch.py

    # Custom CSV file
    python3 phases/phase1/1_fetch.py --csv-file /path/to/tickers.csv

    # Limit number of tickers
    python3 phases/phase1/1_fetch.py --limit 50

    # Resume from specific index
    python3 phases/phase1/1_fetch.py --start-from 100

    # Skip already downloaded
    python3 phases/phase1/1_fetch.py --skip-existing

    # Pin fetch window for reproducibility (REQUIRED for backtests)
    python3 phases/phase1/1_fetch.py --as-of-date 2024-01-01

Output:
    <script_dir>/phase01_stage01_data_fetcher_output/temp_stocks/*.csv
    <script_dir>/phase01_stage01_data_fetcher_output/report/download_report.csv
    <script_dir>/phase01_stage01_data_fetcher_output/logs/fetch_<timestamp>.log
"""

import os
import sys
import re
import time
import hashlib
import logging
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Tuple

# =============================================================================
# DYNAMIC OUTPUT PATHS  (fix: no hardcoded paths, no parent.parent.parent)
# =============================================================================

_SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT_STEM = Path(os.path.basename(__file__)).stem          # e.g. "phase01_stage01_data_fetcher"

OUTPUT_DIR  = _SCRIPT_DIR / f"{_SCRIPT_STEM}_output"
STOCKS_DIR  = OUTPUT_DIR / "temp_stocks"
LOGS_DIR    = OUTPUT_DIR / "logs"
REPORT_DIR  = OUTPUT_DIR / "report"

for _d in (OUTPUT_DIR, STOCKS_DIR, LOGS_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Default CSV lives next to the script (sibling directory)
DEFAULT_CSV_PATH = _SCRIPT_DIR / "nifty_500" / "ind_nifty500list.csv"

# =============================================================================
# LOAD ENVIRONMENT VARIABLES
# =============================================================================

_env_locations = [
    _SCRIPT_DIR / "angel.env",
    _SCRIPT_DIR / ".env",
    OUTPUT_DIR / ".env",
]

_env_loaded = False
for _env_path in _env_locations:
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
        _env_loaded = True
        break

if not _env_loaded:
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
DEFAULT_REQUESTS_PER_MINUTE = 120   # 2 calls/second — safely under Angel One's 10/s limit

# NSE timezone (IST = UTC+5:30) — stored explicitly; never stripped silently
NSE_TZ = timezone(timedelta(hours=5, minutes=30))

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """Setup logging to timestamped file AND console. Returns module logger."""
    log_file = LOGS_DIR / f"fetch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()

# =============================================================================
# SYMBOL NAME SANITISATION  (single source of truth — fix: was duplicated)
# =============================================================================

def sanitise_symbol(symbol: str) -> str:
    """
    Convert a raw ticker symbol to a safe filesystem stem.

    Rules (applied in order):
      1. Replace '&' with 'and'
      2. Replace spaces with '_'
      3. Strip characters illegal on Windows/Linux/macOS: < > : " / \\ | ? *
      4. Lower-case

    This function is the ONE canonical implementation — used both when
    writing files and when checking for existing files (fix: was duplicated
    with slightly different logic in two places, causing skip-existing mismatches).
    """
    s = symbol.replace('&', 'and').replace(' ', '_')
    s = re.sub(r'[<>:"/\\|?*]', '', s)
    return s.lower()

# =============================================================================
# ANGEL ONE API CLIENT
# =============================================================================

class AngelOneClient:
    """
    Angel One SmartAPI client with rate limiting, retry/backoff, and
    strict token resolution (no wrong-ticker fallback).

    Key design decisions
    --------------------
    * Rate limiting is enforced via min_interval between requests.
    * Exponential backoff on 429/502/rate-limit responses.
    * Token resolution is EXACT-match only — if the symbol is not found
      verbatim, the function returns None rather than silently returning
      an unrelated ticker's token (fix: was silently falling back to
      the first -EQ result in search, which could be wrong ticker).
    * TOTP padding is documented with reference to RFC 4648 §6.
    """

    def __init__(self, requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE):
        self.api_key      = os.getenv('ANGEL_API_KEY')
        self.secret_key   = os.getenv('ANGEL_SECRET_KEY')
        self.username     = os.getenv('ANGEL_USERNAME')
        self.mpin         = os.getenv('ANGEL_MPIN')
        self.totp_secret  = os.getenv('ANGEL_TOTP_SECRET')
        self._check_credentials()

        self.obj          = None
        self.is_logged_in = False

        self.requests_per_minute = requests_per_minute
        self.min_interval        = 60.0 / requests_per_minute
        self.last_request_time   = 0.0

    # ------------------------------------------------------------------
    # Credential validation
    # ------------------------------------------------------------------

    def _check_credentials(self) -> None:
        missing = [k for k, v in {
            'ANGEL_API_KEY':    self.api_key,
            'ANGEL_SECRET_KEY': self.secret_key,
            'ANGEL_USERNAME':   self.username,
            'ANGEL_MPIN':       self.mpin,
            'ANGEL_TOTP_SECRET': self.totp_secret,
        }.items() if not v]

        if missing:
            print("\n" + "=" * 60)
            print("ERROR: Missing environment variables:")
            for var in missing:
                print(f"  - {var}")
            print("\nCreate angel.env next to this script with the above keys.")
            print("=" * 60)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.monotonic() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def _update_request_time(self) -> None:
        self.last_request_time = time.monotonic()

    # ------------------------------------------------------------------
    # TOTP
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_totp_secret(secret: str) -> str:
        """
        Normalise a Base32 TOTP secret (RFC 4648 §6).

        Angel One issues 26-character Base32 secrets. Valid Base32 strings
        must be a multiple of 8 characters (each group of 8 chars decodes
        to 5 bytes). A 26-char string is padded to 32 chars with 6 '='
        characters (26 + 6 = 32 = 4 × 8). For any other length the padding
        is computed generically as: pad = (8 - len % 8) % 8.
        """
        if not secret:
            raise ValueError("TOTP secret is empty")

        # Keep only valid Base32 alphabet (A-Z and 2-7)
        secret = re.sub(r'[^A-Z2-7]', '', secret.upper())

        padding = (8 - len(secret) % 8) % 8
        secret += '=' * padding
        return secret

    def _generate_totp(self) -> str:
        return pyotp.TOTP(self._fix_totp_secret(self.totp_secret)).now()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def login(self, max_retries: int = 3, retry_delay: int = 5) -> bool:
        if self.is_logged_in and self.obj:
            return True

        self.obj = SmartConnect(api_key=self.api_key)

        for attempt in range(max_retries):
            try:
                logger.info(f"Login attempt {attempt + 1}/{max_retries}")
                otp = self._generate_totp()
                logger.info(f"Generated TOTP: {otp}")

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

                logger.warning(f"Login failed: {data.get('message', 'Unknown error')}")

            except Exception as e:
                logger.error(f"Login attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay}s…")
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
                    logger.warning(f"Termination error: {result.get('message')}")
        except Exception as e:
            logger.error(f"Logout error: {e}")
        finally:
            self.is_logged_in = False

    # ------------------------------------------------------------------
    # Token resolution — EXACT MATCH ONLY (fix: removed wrong-ticker fallback)
    # ------------------------------------------------------------------

    def get_token_from_symbol(
        self,
        trading_symbol: str,
        exchange: str = DEFAULT_EXCHANGE,
        max_retries: int = 3,
    ) -> Optional[str]:
        """
        Return the token for *exactly* ``trading_symbol`` on ``exchange``.

        No fuzzy fallback: if the API does not return an item whose
        ``tradingsymbol`` field equals ``trading_symbol`` verbatim, the
        method returns ``None``.  This prevents silently fetching a
        different stock's data.
        """
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                result = self.obj.searchScrip(exchange, trading_symbol)
                self._update_request_time()

                if result.get('status') and result.get('data'):
                    # Strict exact-match only
                    for item in result['data']:
                        if item.get('tradingsymbol', '') == trading_symbol:
                            token = item['symboltoken']
                            logger.info(f"Token resolved: {trading_symbol} → {token}")
                            return token

                    # No exact match found
                    logger.error(
                        f"Symbol '{trading_symbol}' not found in search results. "
                        f"Candidates: {[i.get('tradingsymbol') for i in result['data'][:5]]}. "
                        f"Skipping — will NOT use an approximate match."
                    )
                    return None

                error_msg = result.get('message', '')
                if _is_rate_limit_error(error_msg):
                    backoff = (attempt + 1) * 5
                    logger.warning(f"Rate limit hit (token search). Waiting {backoff}s…")
                    time.sleep(backoff)
                    continue

                logger.error(f"searchScrip error for {trading_symbol}: {error_msg}")
                return None

            except Exception as e:
                if _is_rate_limit_error(str(e)):
                    backoff = (attempt + 1) * 10
                    logger.warning(f"Rate limit exception. Waiting {backoff}s…")
                    time.sleep(backoff)
                elif attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    logger.error(f"Token resolution failed for {trading_symbol}: {e}")
                    return None

        return None

    # ------------------------------------------------------------------
    # Historical data
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
        Fetch OHLCV candles.  Both ``from_date`` and ``to_date`` must be
        timezone-aware (IST/UTC) so the fetch window is unambiguous.
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
# HELPER
# =============================================================================

def _is_rate_limit_error(msg: str) -> bool:
    """True if the message indicates a rate-limit or gateway error."""
    msg_lower = msg.lower()
    return any(k in msg_lower for k in ('rate', 'access denied', '502', '429'))


# =============================================================================
# CSV READING
# =============================================================================

def read_symbols_from_csv(
    csv_path: str,
    limit: Optional[int] = None,
    start_from: int = 0,
) -> List[Tuple[str, str]]:
    """
    Parse an NSE Nifty 500 CSV file and return (symbol, trading_symbol) pairs.

    Args:
        csv_path:   Path to CSV.
        limit:      Maximum symbols to return (None = all).
        start_from: Skip the first N rows (for resuming).

    Returns:
        List of (symbol, trading_symbol) e.g. [("RELIANCE", "RELIANCE-EQ"), …]
    """
    import pandas as pd  # local import keeps top-level imports minimal

    try:
        df = pd.read_csv(csv_path)
        logger.info(f"CSV columns: {list(df.columns)}")

        # Case-insensitive column discovery
        col_lower = {c.lower(): c for c in df.columns}
        symbol_col = col_lower.get('symbol')
        series_col = col_lower.get('series')

        if not symbol_col or not series_col:
            logger.error(f"CSV missing 'Symbol' or 'Series' columns. Found: {list(df.columns)}")
            return []

        symbols_data: List[Tuple[str, str]] = []

        for idx, row in df.iterrows():
            if idx < start_from:
                continue
            if limit and len(symbols_data) >= limit:
                break

            symbol = str(row[symbol_col]).strip() if pd.notna(row[symbol_col]) else ''
            series = str(row[series_col]).strip() if pd.notna(row[series_col]) else ''

            if not symbol or symbol in ('nan', 'DUMMYHDLVR'):
                continue

            trading_symbol = f"{symbol}-EQ" if series.upper() == 'EQ' else symbol
            symbols_data.append((symbol, trading_symbol))

        logger.info(f"Read {len(symbols_data)} symbols (start_from={start_from}, limit={limit})")
        return symbols_data

    except Exception as e:
        logger.error(f"Failed to read CSV {csv_path}: {e}")
        return []


# =============================================================================
# FETCH SINGLE TICKER  (fix: decomposed; timezone-aware; post-fetch validation)
# =============================================================================

def _build_dataframe(candles: List, symbol: str) -> "pd.DataFrame":
    """
    Convert raw candle list to a validated, sorted DataFrame.

    Timestamps are stored as timezone-aware UTC strings (ISO 8601) so that
    the downstream cleaner can reconstruct the exact instant unambiguously.
    The NSE feed delivers IST; we convert to UTC and store as a string to
    avoid silent tz-stripping when writing CSV.

    Fix: previously tz was silently stripped with dt.tz_convert(None),
    leaving timestamps ambiguous.  Now we retain UTC offset in the column
    value itself as a string representation, and document the conversion.
    """
    import pandas as pd  # local import

    df = pd.DataFrame(candles, columns=["datetime", "open", "high", "low", "close", "volume"])

    # Parse with tz awareness
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)   # normalise to UTC
    # Store as ISO 8601 string with explicit offset so CSV consumers can
    # parse without ambiguity, even if they don't know the original TZ.
    df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')

    df = df.sort_values('datetime').reset_index(drop=True)
    return df


def _validate_candles(df: "pd.DataFrame", symbol: str) -> List[str]:
    """
    Run basic sanity checks on a freshly fetched candle DataFrame.
    Returns a list of warning strings (empty = all clear).

    Fix: previously there was zero post-fetch validation.
    """
    import pandas as pd  # local import
    warnings: List[str] = []

    # 1. Row count
    if len(df) == 0:
        warnings.append("zero rows returned")
        return warnings

    # 2. OHLCV sanity
    if (df['high'] < df['low']).any():
        count = (df['high'] < df['low']).sum()
        warnings.append(f"{count} rows where high < low")

    if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
        warnings.append("non-positive price(s) detected")

    if (df['volume'] < 0).any():
        warnings.append("negative volume detected")

    # 3. Ascending order
    datetimes = pd.to_datetime(df['datetime'])
    if not datetimes.is_monotonic_increasing:
        warnings.append("timestamps are not strictly ascending after sort")

    # 4. Large gap detection (>10 calendar days without a candle)
    diffs = datetimes.diff().dropna()
    max_gap = diffs.max()
    if pd.notna(max_gap) and max_gap.days > 10:
        warnings.append(f"largest gap between candles: {max_gap.days} days")

    return warnings


def _write_csv_with_checksum(df: "pd.DataFrame", output_file: Path) -> str:
    """
    Write DataFrame to CSV and return SHA-256 of the written file.

    Fix: previously there was no post-write integrity check; a disk-full
    event could produce a truncated file with no warning.
    """
    import pandas as pd  # local import

    output_file.parent.mkdir(parents=True, exist_ok=True)
    csv_bytes = df.to_csv(index=False).encode('utf-8')

    # Write atomically via a temp file to avoid partial writes
    tmp_file = output_file.with_suffix('.tmp')
    try:
        tmp_file.write_bytes(csv_bytes)
        tmp_file.rename(output_file)          # atomic on POSIX
    except Exception:
        tmp_file.unlink(missing_ok=True)
        raise

    # Verify row count
    written_rows = sum(1 for _ in output_file.open()) - 1   # subtract header
    if written_rows != len(df):
        raise IOError(
            f"Write verification failed: expected {len(df)} rows, "
            f"got {written_rows} in {output_file}"
        )

    checksum = hashlib.sha256(csv_bytes).hexdigest()
    return checksum


def fetch_single_ticker(
    client: AngelOneClient,
    symbol: str,
    trading_symbol: str,
    as_of_date: datetime,
    historical_days: int = DEFAULT_HISTORICAL_DAYS,
    exchange: str = DEFAULT_EXCHANGE,
) -> Tuple[bool, str, int]:
    """
    Fetch and persist data for one ticker.

    Args:
        client:          Authenticated AngelOneClient.
        symbol:          Base symbol, e.g. "RELIANCE".
        trading_symbol:  API symbol, e.g. "RELIANCE-EQ".
        as_of_date:      Pinned end-date (timezone-aware) for reproducibility.
        historical_days: Look-back window in calendar days.
        exchange:        Exchange code.

    Returns:
        (success, message, row_count)
    """
    import pandas as pd  # local import

    # 1. Resolve token — exact match only
    token = client.get_token_from_symbol(trading_symbol, exchange)
    if not token:
        return False, f"Exact token not found for '{trading_symbol}'", 0

    # 2. Compute deterministic date window
    to_date   = as_of_date
    from_date = to_date - timedelta(days=historical_days)
    logger.info(f"{symbol}: window {from_date.date()} → {to_date.date()}")

    # 3. Fetch candles
    candles = client.fetch_historical_data(token, exchange, from_date, to_date)
    if candles is None:
        return False, "API returned no data after retries", 0

    # 4. Build DataFrame
    try:
        df = _build_dataframe(candles, symbol)
    except Exception as e:
        return False, f"DataFrame construction failed: {e}", 0

    # 5. Post-fetch validation
    validation_warnings = _validate_candles(df, symbol)
    for w in validation_warnings:
        logger.warning(f"{symbol}: {w}")

    if len(df) == 0:
        return False, "Zero valid rows after parsing", 0

    # 6. Write CSV with integrity check
    output_file = STOCKS_DIR / f"{sanitise_symbol(symbol)}.csv"
    try:
        checksum = _write_csv_with_checksum(df, output_file)
        logger.info(f"{symbol}: saved {len(df)} rows → {output_file.name} (SHA-256: {checksum[:12]}…)")
    except Exception as e:
        return False, f"File write failed: {e}", 0

    return True, "SUCCESS", len(df)


# =============================================================================
# EXISTING FILE CHECK  (fix: uses the same sanitise_symbol as writer)
# =============================================================================

def get_existing_stems(stocks_dir: Path) -> set:
    """Return set of already-downloaded symbol stems (sanitised names)."""
    if not stocks_dir.exists():
        return set()
    return {f.stem for f in stocks_dir.glob("*.csv")}


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
  python3 1_fetch.py --as-of-date 2024-01-01   # pin for reproducibility
  python3 1_fetch.py --limit 50 --skip-existing
  python3 1_fetch.py --start-from 100
        """,
    )
    parser.add_argument('--csv-file', default=str(DEFAULT_CSV_PATH),
                        help='Path to Nifty 500 CSV file')
    parser.add_argument('--limit', type=int, default=None,
                        help='Max tickers to process')
    parser.add_argument('--start-from', type=int, default=0,
                        help='Start from CSV row index (for resuming)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip tickers that already have a CSV file')
    parser.add_argument('--days', type=int, default=DEFAULT_HISTORICAL_DAYS,
                        help='Historical look-back in calendar days')
    parser.add_argument('--exchange', default=DEFAULT_EXCHANGE,
                        help='Exchange code (default: NSE)')
    parser.add_argument('--requests-per-minute', type=int,
                        default=DEFAULT_REQUESTS_PER_MINUTE,
                        help='API rate limit (default: 120 = 2/s)')
    parser.add_argument('--as-of-date', default=None,
                        metavar='YYYY-MM-DD',
                        help=(
                            'Pin the fetch end-date for reproducibility. '
                            'If omitted, uses today (non-reproducible — '
                            'not recommended for backtests).'
                        ))
    parser.add_argument('--verbose', action='store_true',
                        help='Enable DEBUG-level logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Resolve as-of date (fix: was always datetime.now() — non-deterministic)
    # ------------------------------------------------------------------
    if args.as_of_date:
        try:
            as_of_naive = datetime.strptime(args.as_of_date, '%Y-%m-%d')
            # Attach NSE close-of-business time (15:30 IST)
            as_of_date = as_of_naive.replace(
                hour=15, minute=30, tzinfo=NSE_TZ
            )
        except ValueError:
            print(f"ERROR: --as-of-date must be YYYY-MM-DD, got '{args.as_of_date}'")
            sys.exit(1)
    else:
        # Non-pinned: today at NSE close (warn loudly)
        as_of_date = datetime.now(tz=NSE_TZ).replace(hour=15, minute=30, second=0, microsecond=0)
        logger.warning(
            "No --as-of-date supplied. Fetch window ends at today's date "
            f"({as_of_date.date()}). Results will differ on different run days. "
            "Pass --as-of-date YYYY-MM-DD to pin the window."
        )

    print("\n" + "=" * 60)
    print("STEP 1: FETCH DATA FROM ANGEL ONE API")
    print("=" * 60)
    print(f"  As-of date  : {as_of_date.date()} (IST)")
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
    # Skip existing (fix: uses same sanitise_symbol as writer)
    # ------------------------------------------------------------------
    if args.skip_existing:
        existing = get_existing_stems(STOCKS_DIR)
        before = len(symbols_data)
        symbols_data = [
            (sym, ts) for sym, ts in symbols_data
            if sanitise_symbol(sym) not in existing
        ]
        skipped = before - len(symbols_data)
        print(f"  Skipped {skipped} already-downloaded, {len(symbols_data)} remaining")

    if not symbols_data:
        print("\nAll symbols already downloaded.")
        sys.exit(0)

    # Estimate runtime
    estimated_calls   = len(symbols_data) * 2   # token lookup + candle fetch
    estimated_minutes = estimated_calls / args.requests_per_minute
    print(f"\n  Symbols to fetch : {len(symbols_data)}")
    print(f"  Estimated time   : ~{estimated_minutes:.1f} minutes")
    print()

    # ------------------------------------------------------------------
    # Connect to Angel One
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

            ts_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            if ok:
                successful.append({
                    'symbol':    symbol,
                    'rows':      rows,
                    'status':    'success',
                    'message':   msg,
                    'timestamp': ts_now,
                })
                print(f"✓ ({rows} rows)")
            else:
                failed.append({
                    'symbol':    symbol,
                    'rows':      0,
                    'status':    'failed',
                    'message':   msg,
                    'timestamp': ts_now,
                })
                print(f"✗ ({msg})")

    finally:
        client.logout()

    # ------------------------------------------------------------------
    # Persist download report — includes BOTH successes and failures
    # (fix: previously only successes were written)
    # ------------------------------------------------------------------
    import pandas as pd  # local import

    all_records = successful + failed
    report_df = pd.DataFrame(all_records)
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
        print("\n  Failed tickers:")
        for rec in failed[:20]:
            print(f"    - {rec['symbol']}: {rec['message']}")
        if len(failed) > 20:
            print(f"    … and {len(failed) - 20} more (see download_report.csv)")

    print(f"\n  Report : {report_file}")
    print(f"  Stocks : {STOCKS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
