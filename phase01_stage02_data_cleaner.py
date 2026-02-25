#!/usr/bin/env python3
"""
phases/phase1/2_clean.py
========================

Step 2: Clean and merge individual stock CSV files into a single parquet file.

This script:
1. Loads all CSV files from the temp_stocks directory
2. Standardizes column names and data types
3. Validates OHLCV consistency (low ≤ high, open within [low, high])
4. Removes only the rows that fail consistency checks and generates a report
5. Handles missing values and removes duplicates
6. Saves cleaned data to parquet format

Usage:
    python3 phases/phase1/2_clean.py

Input:
    artifacts/phase_1_data/temp_stocks/*.csv (individual stock files)

Output:
    artifacts/phase_1_data/merged_data.parquet
    artifacts/phase_1_data/deleted_rows_report.csv
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input/output paths
INPUT_DIR = PROJECT_ROOT / "artifacts" / "phase_1_data" / "temp_stocks"
OUTPUT_FILE = PROJECT_ROOT / "artifacts" / "phase_1_data" / "merged_data.parquet"
REPORT_FILE = PROJECT_ROOT / "artifacts" / "phase_1_data" / "deleted_rows_report.csv"

# Column name mappings (CSV → Standardized)
COLUMN_MAPPINGS = {
    'datetime': 'date',
    'timestamp': 'date',
    'time': 'date',
    'Datetime': 'date',
    'Timestamp': 'date',
    'Date': 'date',
}

# Expected columns after standardization
EXPECTED_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']

# Missing value handling
FILL_METHOD = "ffill"  # Options: "ffill", "bfill", "drop"
DROP_ROWS_WITH_ALL_NAN = True

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging() -> logging.Logger:
    """Setup logging for this script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# HELPER FUNCTIONS (existing)
# =============================================================================

def get_csv_files(input_dir: Path) -> List[Path]:
    """Get all CSV files from input directory."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    csv_files = list(input_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if f.parent.name != "report"]
    return sorted(csv_files)

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase and rename common variants."""
    df = df.rename(columns=COLUMN_MAPPINGS)
    df.columns = df.columns.str.lower()
    return df

def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Parse datetime column to proper datetime type."""
    if 'date' not in df.columns:
        raise ValueError("DataFrame must have a 'date' column")
    
    for fmt in [None, '%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S']:
        try:
            if fmt is None:
                df['date'] = pd.to_datetime(df['date'])
            else:
                df['date'] = pd.to_datetime(df['date'], format=fmt)
            break
        except (ValueError, TypeError):
            continue
    
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        try:
            df['date'] = pd.to_datetime(df['date'], utc=True)
        except:
            raise ValueError(f"Could not parse datetime column. Sample values: {df['date'].head()}")
    
    if hasattr(df['date'].dtype, 'tz') and df['date'].dt.tz is not None:
        df['date'] = df['date'].dt.tz_convert(None)
    
    return df

def validate_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all OHLCV columns exist and are numeric."""
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in ohlcv_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def clean_single_file(file_path: Path, ticker_name: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Clean a single CSV file."""
    try:
        if ticker_name is None:
            ticker_name = file_path.stem.upper()
        
        df = pd.read_csv(file_path)
        if df.empty:
            logger.warning(f"Empty file: {file_path.name}")
            return None
        
        df = standardize_columns(df)
        df = parse_datetime(df)
        df = validate_ohlcv_columns(df)
        df['ticker'] = ticker_name
        df = df[EXPECTED_COLUMNS]
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {e}")
        return None

# =============================================================================
# UPDATED: OHLCV CONSISTENCY CHECK (remove only bad rows)
# =============================================================================

def validate_ohlcv_consistency(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Check OHLCV data consistency and remove only the violating rows.
    
    Rules:
    1. low <= high (for rows where both are non-null)
    2. open >= low and open <= high (for rows where open, low, high are non-null)
    
    Returns:
        - filtered DataFrame with violating rows removed
        - report DataFrame with per‑ticker summary of removed rows
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Initialize masks for violations
    violation_mask = pd.Series(False, index=df.index)
    
    # Rule 1: low > high
    mask_rule1 = df['low'].notna() & df['high'].notna()
    low_high_viol = mask_rule1 & (df['low'] > df['high'])
    violation_mask |= low_high_viol
    
    # Rule 2: open outside [low, high]
    mask_rule2 = df['open'].notna() & df['low'].notna() & df['high'].notna()
    open_outside = mask_rule2 & ((df['open'] < df['low']) | (df['open'] > df['high']))
    violation_mask |= open_outside
    
    # If no violations, return unchanged
    if not violation_mask.any():
        return df, pd.DataFrame()
    
    # Separate good and bad rows
    bad_rows = df[violation_mask].copy()
    good_rows = df[~violation_mask].copy()
    
    # Generate per‑ticker summary report
    report_rows = []
    for ticker, group in bad_rows.groupby('ticker'):
        reasons = []
        count = len(group)
        # Determine which rules were broken in this ticker (for reporting)
        # We can use the first row to see reason (since all rows in group are from same ticker)
        # But for simplicity, we'll just record that violations occurred.
        # More detailed: we could count per rule, but we'll keep it simple.
        reasons.append(f"{count} row(s) with OHLCV inconsistency")
        # (If we wanted per‑rule counts we'd need to re‑evaluate)
        first_date = group['date'].min()
        report_rows.append({
            'ticker': ticker,
            'reason': '; '.join(reasons),
            'violation_count': count,
            'first_violation_date': first_date.strftime('%Y-%m-%d') if pd.notna(first_date) else None
        })
    
    report_df = pd.DataFrame(report_rows)
    
    # Log summary
    logger.info(f"Removed {len(bad_rows)} rows with OHLCV inconsistencies across {len(report_df)} tickers")
    
    return good_rows, report_df

# =============================================================================
# EXISTING CLEANING FUNCTIONS (unchanged)
# =============================================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the DataFrame."""
    initial_rows = len(df)
    initial_tickers = df['ticker'].nunique()
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    
    if DROP_ROWS_WITH_ALL_NAN:
        mask = df[ohlcv_cols].notna().any(axis=1)
        df = df[mask].reset_index(drop=True)
    
    if FILL_METHOD == "ffill":
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
        df[ohlcv_cols] = df.groupby('ticker')[ohlcv_cols].transform(lambda x: x.ffill())
    elif FILL_METHOD == "bfill":
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
        df[ohlcv_cols] = df.groupby('ticker')[ohlcv_cols].transform(lambda x: x.bfill())
    elif FILL_METHOD == "drop":
        df = df.dropna(subset=ohlcv_cols).reset_index(drop=True)
    
    df = df.dropna(subset=['close']).reset_index(drop=True)
    
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        logger.info(f"Dropped {dropped_rows} rows with missing values")
    logger.info(f"Missing value handling: {initial_tickers} → {df['ticker'].nunique()} tickers")
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows based on (date, ticker)."""
    initial_rows = len(df)
    df = df.drop_duplicates(subset=['date', 'ticker'], keep='first').reset_index(drop=True)
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate rows")
    return df

def get_data_summary(df: pd.DataFrame) -> dict:
    """Generate summary statistics for the cleaned data."""
    return {
        'total_rows': len(df),
        'total_tickers': df['ticker'].nunique(),
        'date_start': df['date'].min().strftime('%Y-%m-%d'),
        'date_end': df['date'].max().strftime('%Y-%m-%d'),
        'memory_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for data cleaning."""
    
    print("\n" + "=" * 60)
    print("STEP 2: CLEAN AND MERGE DATA")
    print("=" * 60)
    
    if not INPUT_DIR.exists():
        print(f"\nERROR: Input directory not found: {INPUT_DIR}")
        print("\nRun 1_fetch.py first to download data.")
        sys.exit(1)
    
    print(f"\n  Input: {INPUT_DIR}")
    print(f"  Output: {OUTPUT_FILE}")
    
    # Get CSV files
    print("\n  Loading files...")
    try:
        csv_files = get_csv_files(INPUT_DIR)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    
    if not csv_files:
        print(f"\nERROR: No CSV files found in {INPUT_DIR}")
        sys.exit(1)
    
    logger.info(f"Found {len(csv_files)} CSV files")
    
    # Load and clean each file
    all_dfs = []
    failed_files = []
    
    for i, file_path in enumerate(csv_files):
        df = clean_single_file(file_path)
        if df is not None:
            all_dfs.append(df)
        else:
            failed_files.append(file_path.name)
    
    if not all_dfs:
        print("\nERROR: No valid CSV files could be loaded")
        sys.exit(1)
    
    # Concatenate all DataFrames
    print(f"  Loaded {sum(len(df) for df in all_dfs)} rows from {len(all_dfs)} tickers")
    merged_df = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    
    # -------------------------------------------------------------------------
    # UPDATED: OHLCV Consistency Check (remove only bad rows)
    # -------------------------------------------------------------------------
    print("  Checking OHLCV consistency...")
    initial_row_count = len(merged_df)
    merged_df, report_df = validate_ohlcv_consistency(merged_df)
    removed_rows = initial_row_count - len(merged_df)
    
    if not report_df.empty:
        # Save report
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(REPORT_FILE, index=False)
        logger.info(f"Removed {removed_rows} rows with OHLCV inconsistencies. Report saved to {REPORT_FILE}")
        print(f"  Removed {removed_rows} rows due to OHLCV inconsistencies.")
    else:
        print("  All rows passed OHLCV consistency checks.")
    
    # Continue with standard cleaning
    print("  Handling missing values...")
    merged_df = handle_missing_values(merged_df)
    
    print("  Removing duplicates...")
    merged_df = remove_duplicates(merged_df)
    
    print("  Sorting data...")
    merged_df = merged_df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # Summary
    summary = get_data_summary(merged_df)
    
    print("  Saving merged data...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(OUTPUT_FILE, engine='pyarrow', compression='snappy', index=False)
    
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    
    print("\n" + "-" * 60)
    print("CLEAN SUMMARY")
    print("-" * 60)
    print(f"  Total rows: {summary['total_rows']:,}")
    print(f"  Total tickers: {summary['total_tickers']}")
    print(f"  Date range: {summary['date_start']} to {summary['date_end']}")
    print(f"  File size: {file_size_mb:.2f} MB")
    
    if removed_rows > 0:
        print(f"\n  Removed inconsistent rows: {removed_rows} (see {REPORT_FILE.name})")
    
    if failed_files:
        print(f"\n  Failed files ({len(failed_files)}):")
        for f in failed_files[:5]:
            print(f"    - {f}")
        if len(failed_files) > 5:
            print(f"    ... and {len(failed_files) - 5} more")
    
    print(f"\n  Output: {OUTPUT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    main()
