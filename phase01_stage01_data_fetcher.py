#!/usr/bin/env python3
"""
phases/phase1/1_fetch.py
========================

Step 1: Fetch historical data from Angel One API.

Reads tickers from Nifty 500 CSV file and downloads historical OHLCV data.

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

Output:
    artifacts/phase_1_data/temp_stocks/*.csv (individual stock files)
"""

import os
import sys
import re
import time
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pyotp

# =============================================================================
# LOAD ENVIRONMENT VARIABLES
# =============================================================================

# Try multiple .env file locations
env_locations = [
    Path(__file__).parent / "angel.env",      # phases/phase1/angel.env
    Path(__file__).parent / ".env",           # phases/phase1/.env
    PROJECT_ROOT / ".env",                     # project_root/.env
    PROJECT_ROOT / "angel.env",                # project_root/angel.env
]

env_loaded = False
for env_path in env_locations:
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
            print(f"[INFO] Loaded environment from: {env_path}")
            env_loaded = True
            break
        except ImportError:
            # dotenv not installed, try manual loading
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
            print(f"[INFO] Loaded environment from: {env_path}")
            env_loaded = True
            break

if not env_loaded:
    print("[WARNING] No .env file found. Looking for system environment variables.")

# Angel One SmartAPI
try:
    from SmartApi import SmartConnect
    ANGEL_API_AVAILABLE = True
except ImportError:
    ANGEL_API_AVAILABLE = False
    print("ERROR: SmartAPI not installed. Run: pip install smartapi-python")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default paths
DEFAULT_CSV_PATH = PROJECT_ROOT / "phases" / "phase1" / "nifty_500" / "ind_nifty500list.csv"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "phase_1_data" / "temp_stocks"
LOGS_DIR = PROJECT_ROOT / "phases" / "phase1" / "logs"

# Historical data settings
DEFAULT_HISTORICAL_DAYS = 2000
DEFAULT_EXCHANGE = "NSE"

# Rate limiting
# Angel One API: 10 calls/second limit
# We use 2 calls/second to be safe (well under limit)
DEFAULT_REQUESTS_PER_MINUTE = 120  # 2 calls per second

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Setup logging to file and console."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    log_file = LOGS_DIR / f"fetch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# ANGEL ONE API CLIENT
# =============================================================================

class AngelOneClient:
    """
    Angel One API client with rate limiting and retry logic.
    
    Features:
        - Automatic TOTP generation
        - Rate limiting with configurable limits
        - Exponential backoff on failures
        - Session management
    """
    
    def __init__(self, requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE):
        """
        Initialize client.
        
        Args:
            requests_per_minute: Rate limit for API calls
        """
        # Get credentials from environment
        self.api_key = os.getenv('ANGEL_API_KEY')
        self.secret_key = os.getenv('ANGEL_SECRET_KEY')
        self.username = os.getenv('ANGEL_USERNAME')
        self.mpin = os.getenv('ANGEL_MPIN')
        self.totp_secret = os.getenv('ANGEL_TOTP_SECRET')
        
        # Validate credentials
        self._check_credentials()
        
        logger.info(f"Initialized with username: {self.username}")
        
        # Connection state
        self.obj = None
        self.is_logged_in = False
        
        # Rate limiting
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0
    
    def _check_credentials(self):
        """Validate that all required credentials are present."""
        missing = []
        
        if not self.api_key:
            missing.append("ANGEL_API_KEY")
        if not self.secret_key:
            missing.append("ANGEL_SECRET_KEY")
        if not self.username:
            missing.append("ANGEL_USERNAME")
        if not self.mpin:
            missing.append("ANGEL_MPIN")
        if not self.totp_secret:
            missing.append("ANGEL_TOTP_SECRET")
        
        if missing:
            print("\n" + "=" * 60)
            print("ERROR: Missing environment variables:")
            for var in missing:
                print(f"  - {var}")
            print("\nCreate an angel.env file with:")
            print("  ANGEL_API_KEY=your_key")
            print("  ANGEL_SECRET_KEY=your_secret")
            print("  ANGEL_USERNAME=your_username")
            print("  ANGEL_MPIN=your_mpin")
            print("  ANGEL_TOTP_SECRET=your_totp_secret")
            print("=" * 60)
            sys.exit(1)
    
    def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
    
    def _update_request_time(self):
        """Update last request timestamp."""
        self.last_request_time = time.time()
    
    def _fix_totp_secret(self, secret: str) -> str:
        """
        Fix Base32 padding issues in TOTP secret.
        
        Angel One TOTP secrets are often 26 characters,
        which need padding to be valid Base32 (32 chars).
        """
        if not secret:
            raise ValueError("TOTP secret is empty")
        
        # Remove spaces, hyphens, special characters
        secret = re.sub(r'[^A-Z2-7]', '', secret.upper())
        
        # Add padding for 26-char keys (common for Angel One)
        if len(secret) == 26:
            secret += '======'
        else:
            # Generic padding to multiple of 8
            padding_needed = (8 - len(secret) % 8) % 8
            secret += '=' * padding_needed
        
        return secret
    
    def _generate_totp(self) -> str:
        """Generate current TOTP code."""
        try:
            fixed_secret = self._fix_totp_secret(self.totp_secret)
            totp = pyotp.TOTP(fixed_secret)
            return totp.now()
        except Exception as e:
            logger.error(f"TOTP generation failed: {e}")
            raise
    
    def login(self, max_retries: int = 3, retry_delay: int = 5) -> bool:
        """
        Login to Angel One API.
        
        Args:
            max_retries: Number of login attempts
            retry_delay: Seconds between retries
            
        Returns:
            True if login successful
        """
        if self.is_logged_in and self.obj:
            logger.info("Already logged in")
            return True
        
        self.obj = SmartConnect(api_key=self.api_key)
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Login attempt {attempt + 1}/{max_retries}")
                current_otp = self._generate_totp()
                logger.info(f"Generated TOTP: {current_otp}")
                
                # Try different parameter combinations based on SmartAPI version
                try:
                    data = self.obj.generateSession(
                        self.username,
                        self.mpin,
                        current_otp
                    )
                except TypeError as e:
                    if "'clientCode'" in str(e):
                        data = self.obj.generateSession(
                            self.username,
                            self.mpin,
                            current_otp,
                            self.username
                        )
                    else:
                        raise
                
                if data.get('status') and data.get('message') == 'SUCCESS':
                    self.is_logged_in = True
                    logger.info("Login successful")
                    return True
                else:
                    error_msg = data.get('message', 'Unknown error')
                    logger.warning(f"Login failed: {error_msg}")
                    
            except Exception as e:
                logger.error(f"Login attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error("Max login retries exceeded")
                    return False
        
        return False
    
    def logout(self):
        """Logout from Angel One API."""
        try:
            if self.obj and self.is_logged_in:
                result = self.obj.terminateSession(self.username)
                if result.get('status'):
                    logger.info("Session terminated successfully")
                else:
                    logger.warning(f"Session termination failed: {result.get('message')}")
        except Exception as e:
            logger.error(f"Error during logout: {e}")
        finally:
            self.is_logged_in = False
    
    def get_token_from_symbol(
        self,
        trading_symbol: str,
        exchange: str = DEFAULT_EXCHANGE,
        max_retries: int = 3
    ) -> Optional[str]:
        """
        Get token for a trading symbol with retry logic.
        
        Args:
            trading_symbol: Symbol with -EQ suffix (e.g., "RELIANCE-EQ")
            exchange: Exchange name (default: NSE)
            max_retries: Number of retry attempts
            
        Returns:
            Token string or None if not found
        """
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                
                logger.debug(f"Searching for {trading_symbol} (attempt {attempt + 1}/{max_retries})")
                search_result = self.obj.searchScrip(exchange, trading_symbol)
                
                self._update_request_time()
                
                if search_result.get('status') and search_result.get('data'):
                    # Look for exact match first
                    for item in search_result['data']:
                        if item.get('tradingsymbol', '') == trading_symbol:
                            token = item.get('symboltoken', '')
                            logger.info(f"Found exact match: {trading_symbol} with token {token}")
                            return token
                    
                    # Fall back to any equity symbol
                    for item in search_result['data']:
                        ts = item.get('tradingsymbol', '')
                        if '-EQ' in ts:
                            token = item.get('symboltoken', '')
                            logger.info(f"Found alternative: {ts} with token {token}")
                            return token
                
                # Check for rate limit error
                error_msg = search_result.get('message', '')
                if 'rate' in error_msg.lower() or 'access denied' in error_msg.lower():
                    logger.warning(f"Rate limit hit during search for {trading_symbol}")
                    if attempt < max_retries - 1:
                        backoff_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                        logger.info(f"Waiting {backoff_time}s before retry...")
                        time.sleep(backoff_time)
                        continue
                
                logger.error(f"Could not find token for {trading_symbol}: {error_msg}")
                return None
                
            except Exception as e:
                error_str = str(e)
                logger.error(f"Error getting token for {trading_symbol} (attempt {attempt + 1}): {error_str}")
                
                # Check for rate limit error
                if 'rate' in error_str.lower() or 'access denied' in error_str.lower() or '502' in error_str:
                    if attempt < max_retries - 1:
                        backoff_time = (attempt + 1) * 10  # 10, 20, 30 seconds
                        logger.info(f"Rate limit error. Waiting {backoff_time}s...")
                        time.sleep(backoff_time)
                        continue
                elif attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                
                return None
        
        return None
    
    def fetch_historical_data(
        self,
        symbol_token: str,
        exchange: str,
        from_date: datetime,
        to_date: datetime,
        interval: str = "ONE_DAY",
        max_retries: int = 3
    ) -> Optional[List]:
        """
        Fetch historical candle data with retry logic.
        
        Args:
            symbol_token: Token from get_token_from_symbol
            exchange: Exchange name
            from_date: Start date
            to_date: End date
            interval: Candle interval (default: ONE_DAY)
            max_retries: Number of retry attempts
            
        Returns:
            List of candles or None
        """
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                
                params = {
                    "exchange": exchange,
                    "symboltoken": symbol_token,
                    "interval": interval,
                    "fromdate": from_date.strftime('%Y-%m-%d %H:%M'),
                    "todate": to_date.strftime('%Y-%m-%d %H:%M')
                }
                
                logger.info(f"Fetching data from {from_date.date()} to {to_date.date()} (attempt {attempt + 1}/{max_retries})")
                historical_data = self.obj.getCandleData(params)
                
                self._update_request_time()
                
                if historical_data.get('status') and historical_data.get('message') == 'SUCCESS':
                    candles = historical_data.get('data', [])
                    logger.info(f"Retrieved {len(candles)} candles")
                    return candles
                else:
                    error_msg = historical_data.get('message', 'Unknown error')
                    logger.error(f"API returned error: {error_msg}")
                    
                    # Check for rate limit errors
                    if 'rate' in error_msg.lower() or 'access denied' in error_msg.lower():
                        if attempt < max_retries - 1:
                            backoff_time = (attempt + 1) * 10
                            logger.warning(f"Rate limit hit. Waiting {backoff_time}s...")
                            time.sleep(backoff_time)
                            continue
                    
                    return None
                    
            except Exception as e:
                error_str = str(e)
                logger.error(f"Error fetching historical data (attempt {attempt + 1}): {error_str}")
                
                # Check for rate limit error
                if 'rate' in error_str.lower() or 'access denied' in error_str.lower() or '502' in error_str:
                    if attempt < max_retries - 1:
                        backoff_time = (attempt + 1) * 15  # 15, 30, 45 seconds
                        logger.warning(f"Rate limit error. Waiting {backoff_time}s...")
                        time.sleep(backoff_time)
                        continue
                elif attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                
                return None
        
        return None


# =============================================================================
# CSV READING
# =============================================================================

def read_symbols_from_csv(
    csv_path: str,
    limit: Optional[int] = None,
    start_from: int = 0
) -> List[Tuple[str, str]]:
    """
    Read symbols from Nifty 500 CSV file.
    
    Args:
        csv_path: Path to the CSV file
        limit: Maximum number of symbols to read (None for all)
        start_from: Start from this index (for resuming)
    
    Returns:
        List of tuples (symbol, trading_symbol)
    """
    try:
        symbols_data = []
        
        # Read CSV file
        df = pd.read_csv(csv_path)
        logger.info(f"CSV columns: {list(df.columns)}")
        
        # Find required columns (case-insensitive)
        required_columns = ['Symbol', 'Series']
        column_map = {}
        
        for col in df.columns:
            for req in required_columns:
                if req.lower() in col.lower():
                    column_map[req] = col
                    break
        
        if len(column_map) != len(required_columns):
            logger.error(f"CSV file missing required columns. Found: {list(df.columns)}")
            print(f"\nERROR: CSV must have 'Symbol' and 'Series' columns")
            print(f"Found columns: {list(df.columns)}")
            return []
        
        symbol_col = column_map['Symbol']
        series_col = column_map['Series']
        
        logger.info(f"Using columns: Symbol='{symbol_col}', Series='{series_col}'")
        
        # Extract symbols
        for index, row in df.iterrows():
            if index < start_from:
                continue
            
            if limit and len(symbols_data) >= limit:
                break
            
            symbol = str(row[symbol_col]).strip() if pd.notna(row[symbol_col]) else None
            series = str(row[series_col]).strip() if pd.notna(row[series_col]) else None
            
            # Skip invalid symbols
            if not symbol or symbol == 'nan' or symbol == 'DUMMYHDLVR':
                continue
            
            # Create trading symbol
            if series and series.upper() == 'EQ':
                trading_symbol = f"{symbol}-EQ"
            else:
                trading_symbol = symbol
            
            symbols_data.append((symbol, trading_symbol))
        
        logger.info(f"Read {len(symbols_data)} symbols from CSV file")
        
        if symbols_data:
            logger.info(f"First 10 symbols: {[s[0] for s in symbols_data[:10]]}")
        
        return symbols_data
        
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return []


# =============================================================================
# FETCH LOGIC
# =============================================================================

def fetch_single_ticker(
    client: AngelOneClient,
    symbol: str,
    trading_symbol: str,
    output_dir: Path,
    historical_days: int = DEFAULT_HISTORICAL_DAYS,
    exchange: str = DEFAULT_EXCHANGE
) -> Tuple[bool, str, int]:
    """
    Fetch data for a single ticker.
    
    Args:
        client: AngelOneClient instance
        symbol: Base symbol (e.g., "RELIANCE")
        trading_symbol: Symbol with suffix (e.g., "RELIANCE-EQ")
        output_dir: Directory to save CSV files
        historical_days: Number of days of historical data
        exchange: Exchange name
        
    Returns:
        Tuple of (success, message, rows)
    """
    try:
        # Get token
        token = client.get_token_from_symbol(trading_symbol, exchange, max_retries=3)
        
        if not token:
            return (False, "Token not found after retries", 0)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=historical_days)
        
        # Fetch historical data
        candles = client.fetch_historical_data(
            token, exchange, start_date, end_date, max_retries=3
        )
        
        if not candles:
            return (False, "No data retrieved after retries", 0)
        
        # Create DataFrame
        df = pd.DataFrame(
            candles,
            columns=["datetime", "open", "high", "low", "close", "volume"]
        )
        
        # Parse datetime
        try:
            df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%dT%H:%M:%S%z')
        except:
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        
        df['datetime'] = df['datetime'].dt.tz_convert(None)
        df.sort_values('datetime', inplace=True)
        
        # Clean symbol name for filename
        clean_symbol = symbol.lower()
        clean_symbol = re.sub(r'[&<>:"/\\|?*]', '', clean_symbol)
        clean_symbol = clean_symbol.replace('&', 'and').replace(' ', '_')
        
        # Save to CSV
        output_file = output_dir / f"{clean_symbol}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Data saved to {output_file}")
        
        return (True, "Success", len(df))
        
    except Exception as e:
        return (False, str(e), 0)


def get_existing_files(output_dir: Path) -> set:
    """Get set of already downloaded symbol names."""
    if not output_dir.exists():
        return set()
    
    return {f.stem for f in output_dir.glob("*.csv")}


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fetch historical data for NSE stocks from Angel One API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fetch all tickers from default CSV
    python3 phases/phase1/1_fetch.py
    
    # Custom CSV file
    python3 phases/phase1/1_fetch.py --csv-file /path/to/tickers.csv
    
    # Limit to 50 tickers
    python3 phases/phase1/1_fetch.py --limit 50
    
    # Resume from index 100
    python3 phases/phase1/1_fetch.py --start-from 100
    
    # Skip already downloaded
    python3 phases/phase1/1_fetch.py --skip-existing
    
    # Combine options
    python3 phases/phase1/1_fetch.py --skip-existing --limit 100
        """
    )
    
    parser.add_argument(
        '--csv-file', type=str, default=str(DEFAULT_CSV_PATH),
        help=f'Path to CSV file containing symbols (default: {DEFAULT_CSV_PATH})'
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Limit number of symbols to process (default: all)'
    )
    parser.add_argument(
        '--start-from', type=int, default=0,
        help='Start processing from this index (for resuming)'
    )
    parser.add_argument(
        '--skip-existing', action='store_true',
        help='Skip symbols that already have data files'
    )
    parser.add_argument(
        '--days', type=int, default=DEFAULT_HISTORICAL_DAYS,
        help=f'Number of days of historical data (default: {DEFAULT_HISTORICAL_DAYS})'
    )
    parser.add_argument(
        '--exchange', type=str, default=DEFAULT_EXCHANGE,
        help=f'Exchange (default: {DEFAULT_EXCHANGE})'
    )
    parser.add_argument(
        '--requests-per-minute', type=int, default=DEFAULT_REQUESTS_PER_MINUTE,
        help=f'Max requests per minute (default: {DEFAULT_REQUESTS_PER_MINUTE})'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 60)
    print("STEP 1: FETCH DATA FROM ANGEL ONE API")
    print("=" * 60)
    
    # Check CSV file
    if not os.path.exists(args.csv_file):
        print(f"\nERROR: CSV file not found: {args.csv_file}")
        print("\nPlease provide the correct path using --csv-file")
        print(f"Default path: {DEFAULT_CSV_PATH}")
        sys.exit(1)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Read symbols from CSV
    print(f"\n  Reading symbols from: {args.csv_file}")
    symbols_data = read_symbols_from_csv(
        args.csv_file,
        limit=args.limit,
        start_from=args.start_from
    )
    
    if not symbols_data:
        print("\nERROR: No symbols to process")
        sys.exit(1)
    
    # Skip existing if requested
    if args.skip_existing:
        existing = get_existing_files(OUTPUT_DIR)
        
        if existing:
            filtered = []
            for symbol, trading_symbol in symbols_data:
                clean_name = re.sub(r'[&<>:"/\\|?*]', '', symbol.lower())
                clean_name = clean_name.replace('&', 'and').replace(' ', '_')
                
                if clean_name in existing:
                    logger.info(f"Skipping {symbol} (already exists)")
                else:
                    filtered.append((symbol, trading_symbol))
            
            symbols_data = filtered
            print(f"  Skipped {len(existing)} existing files, {len(symbols_data)} remaining")
    
    if not symbols_data:
        print("\nAll symbols already downloaded. Use without --skip-existing to re-download.")
        sys.exit(0)
    
    # Print settings
    print(f"\n  Symbols to fetch: {len(symbols_data)}")
    print(f"  Historical days: {args.days}")
    print(f"  Exchange: {args.exchange}")
    print(f"  Rate limit: {args.requests_per_minute} requests/minute")
    print(f"  Output: {OUTPUT_DIR}")
    
    # Estimate time (2 API calls per ticker: search + fetch)
    estimated_calls = len(symbols_data) * 2
    estimated_minutes = estimated_calls / args.requests_per_minute
    print(f"  Estimated time: ~{estimated_minutes:.1f} minutes")
    print()
    
    # Initialize client
    client = AngelOneClient(requests_per_minute=args.requests_per_minute)
    
    # Login
    if not client.login():
        print("\nERROR: Login failed. Check your credentials.")
        sys.exit(1)
    
    try:
        # Fetch each ticker
        successful = []
        failed = []
        
        for i, (symbol, trading_symbol) in enumerate(symbols_data):
            print(f"  [{i+1}/{len(symbols_data)}] {symbol}...", end=" ", flush=True)
            
            success, message, rows = fetch_single_ticker(
                client, symbol, trading_symbol,
                OUTPUT_DIR, args.days, args.exchange
            )
            
            if success:
                successful.append(symbol)
                print(f"✓ ({rows} rows)")
            else:
                failed.append((symbol, message))
                print(f"✗ ({message})")
        
        # Summary
        print("\n" + "-" * 60)
        print("FETCH SUMMARY")
        print("-" * 60)
        print(f"  Total processed: {len(symbols_data)}")
        print(f"  ✓ Successful: {len(successful)}")
        print(f"  ✗ Failed: {len(failed)}")
        
        if successful:
            print(f"\n  First 10 successful: {', '.join(successful[:10])}")
            if len(successful) > 10:
                print(f"  ... and {len(successful) - 10} more")
        
        if failed:
            print(f"\n  Failed tickers:")
            for symbol, error in failed[:10]:
                print(f"    - {symbol}: {error}")
            if len(failed) > 10:
                print(f"    ... and {len(failed) - 10} more")
        
        # Save download report
        if successful:
            report_dir = OUTPUT_DIR / "report"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_file = report_dir / "download_report.csv"
            
            report_df = pd.DataFrame({
                'symbol': successful,
                'status': 'success',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            report_df.to_csv(report_file, index=False)
            print(f"\n  📋 Report saved: {report_file}")
        
        print(f"\n  📁 Output directory: {OUTPUT_DIR}")
        print("=" * 60)
        
    finally:
        client.logout()


if __name__ == "__main__":
    main()
