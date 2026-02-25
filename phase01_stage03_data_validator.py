#!/usr/bin/env python3
"""
phases/phase1/3_validate.py
===========================

Step 3: Validate OHLCV data integrity.

Run:
    cd /Users/kuro/python_project/angel1
    python3 phases/phase1/3_validate.py

Input:
    artifacts/phase_1_data/merged_data.parquet

Output:
    artifacts/phase_1_data/output.parquet (final validated data)
    artifacts/phase_1_data/validation_report.json
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = PROJECT_ROOT / "artifacts" / "phase_1_data" / "merged_data.parquet"
OUTPUT_FILE = PROJECT_ROOT / "artifacts" / "phase_1_data" / "output.parquet"
REPORT_FILE = PROJECT_ROOT / "artifacts" / "phase_1_data" / "validation_report.json"

# Validation settings
MAX_PRICE_CHANGE_PCT = 0.50  # Flag if daily move > 50%

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION CHECKS
# =============================================================================

class OHLCVValidator:
    """Validates OHLCV data integrity."""
    
    def __init__(self):
        self.warnings = []
        self.errors = []
        self.rows_dropped = 0
    
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all validation checks.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Validated DataFrame with invalid rows removed
        """
        print("  Running validation checks...\n")
        
        original_len = len(df)
        df = df.copy()
        df['_failed'] = False
        
        # Check 1: OHLCV relationships
        self._check_ohlcv_relationships(df)
        
        # Check 2: Positive prices
        self._check_positive_prices(df)
        
        # Check 3: Non-negative volume
        self._check_volume(df)
        
        # Check 4: Duplicates
        df = self._check_duplicates(df)
        
        # Check 5: Price anomalies
        self._check_price_anomalies(df)
        
        # Remove failed rows
        df = df[~df['_failed']].copy()
        df.drop(columns=['_failed'], inplace=True)
        
        self.rows_dropped = original_len - len(df)
        
        return df
    
    def _check_ohlcv_relationships(self, df: pd.DataFrame):
        """Check OHLC price relationships."""
        checks = [
            ('high < low', df['high'] < df['low']),
            ('high < open', df['high'] < df['open']),
            ('high < close', df['high'] < df['close']),
            ('low > open', df['low'] > df['open']),
            ('low > close', df['low'] > df['close']),
        ]
        
        for name, mask in checks:
            if mask.any():
                count = mask.sum()
                self.warnings.append(f"{name} in {count} rows")
                df.loc[mask, '_failed'] = True
                print(f"    ⚠ {name}: {count} rows")
    
    def _check_positive_prices(self, df: pd.DataFrame):
        """Check all prices are positive."""
        for col in ['open', 'high', 'low', 'close']:
            mask = df[col] <= 0
            if mask.any():
                count = mask.sum()
                self.errors.append(f"{col} <= 0 in {count} rows")
                df.loc[mask, '_failed'] = True
                print(f"    ✗ {col} <= 0: {count} rows")
    
    def _check_volume(self, df: pd.DataFrame):
        """Check volume is non-negative."""
        mask = df['volume'] < 0
        if mask.any():
            count = mask.sum()
            self.errors.append(f"volume < 0 in {count} rows")
            df.loc[mask, '_failed'] = True
            print(f"    ✗ volume < 0: {count} rows")
    
    def _check_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for duplicate (ticker, date) pairs."""
        duplicates = df.duplicated(subset=['ticker', 'date'], keep=False)
        
        if duplicates.any():
            count = duplicates.sum()
            self.warnings.append(f"{count} duplicate (ticker, date) pairs")
            print(f"    ⚠ Duplicates: {count} rows")
            
            # Keep first
            df = df.drop_duplicates(subset=['ticker', 'date'], keep='first')
        
        return df
    
    def _check_price_anomalies(self, df: pd.DataFrame):
        """Check for extreme price movements."""
        df_sorted = df.sort_values(['ticker', 'date'])
        df['_daily_return'] = df_sorted.groupby('ticker')['close'].pct_change()
        
        extreme_mask = df['_daily_return'].abs() > MAX_PRICE_CHANGE_PCT
        
        if extreme_mask.any():
            # Count per ticker
            extreme_counts = df[extreme_mask].groupby('ticker').size()
            
            for ticker, count in extreme_counts.items():
                self.warnings.append(
                    f"{ticker}: {count} days with >{MAX_PRICE_CHANGE_PCT:.0%} move"
                )
            
            print(f"    ⚠ Extreme moves: {extreme_mask.sum()} rows")
            print(f"       (not dropped - may be legitimate volatility)")
        
        df.drop(columns=['_daily_return'], inplace=True)
    
    def get_report(self) -> dict:
        """Get validation report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "warnings": self.warnings,
            "errors": self.errors,
            "warning_count": len(self.warnings),
            "error_count": len(self.errors),
            "rows_dropped": self.rows_dropped,
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("STEP 3: VALIDATE OHLCV DATA")
    print("=" * 60)
    
    # Check input
    if not INPUT_FILE.exists():
        print(f"\nERROR: Input file not found: {INPUT_FILE}")
        print("Run 2_clean.py first")
        sys.exit(1)
    
    print(f"\n  Input: {INPUT_FILE}")
    print(f"  Output: {OUTPUT_FILE}")
    print()
    
    # Load data
    print("  Loading data...")
    df = pd.read_parquet(INPUT_FILE)
    
    print(f"  Loaded {len(df):,} rows from {df['ticker'].nunique()} tickers")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print()
    
    # Validate
    validator = OHLCVValidator()
    df = validator.validate(df)
    
    # Save output
    print("\n  Saving validated data...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, engine='pyarrow', index=False)
    
    # Save report
    report = validator.get_report()
    report["final_rows"] = len(df)
    report["final_tickers"] = df['ticker'].nunique()
    report["date_range"] = {
        "min": str(df['date'].min()),
        "max": str(df['date'].max()),
    }
    
    with open(REPORT_FILE, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Summary
    print("\n" + "-" * 60)
    print("VALIDATION SUMMARY")
    print("-" * 60)
    print(f"  Warnings: {report['warning_count']}")
    print(f"  Errors: {report['error_count']}")
    print(f"  Rows dropped: {report['rows_dropped']}")
    print(f"  Final rows: {len(df):,}")
    print(f"  Final tickers: {df['ticker'].nunique()}")
    
    # File size
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    
    print(f"\n  Output: {OUTPUT_FILE}")
    print(f"  Report: {REPORT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
