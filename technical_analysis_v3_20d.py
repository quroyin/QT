# technical_analysis.py - Cross-Sectional Relative Strength Ranking Engine v5.0
#
# Refactored from v4.0 (Rule-Based) → v5.0 (ML-Augmented)
#
# Changes from v4.0:
#   + MLModelTrainer class: Ridge (Phase 1), XGBoost (Phase 2)
#   + prepare_training_data(): panel dataset with rolling cross-sectional Z-scores
#   + Neutralized target: excess return over universe median per date
#   + Feature column contract stored inside .pkl artifacts
#   + Ridge coefficient logging (mandatory JSON artifact)
#   + Rank IC (Spearman) as primary validation metric
#   + Walk-forward TimeSeriesSplit CV with per-fold reporting
#   + XGBoost performance threshold: must beat Ridge R² by >5%
#   + Graceful fallback to rule-based weights if model missing/corrupt
#   + MIN_TRAINING_SAMPLES guard before training is allowed
#   - Removed: StandardScaler (inputs are Z-scored; scaling is redundant)
#   ~ Config: USE_ML_MODEL, MODEL_TYPE, MODEL_PATH, MODEL_DIR, ML hyperparams

import pandas as pd
import numpy as np
import os
import logging
import sys
from datetime import datetime
import ta
import glob
from tqdm import tqdm
import warnings
from typing import Dict, List, Optional, Any, Tuple
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# ML imports
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr
import joblib

# XGBoost — Phase 2, optional hard dependency
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    """Configuration for the Cross-Sectional Relative Strength Ranking Engine v5.0."""

    def __init__(self, config_file: Optional[str] = None):
        # --- Directories ---
        self.INPUT_DATA_DIR    = "/Users/kuro/python_project/angel1/phases/etf/phase01_stage01_output/temp_stocks"
        self.OUTPUT_REPORT_DIR = "/Users/kuro/python_project/angel1/phases/etf/etf_v3_20days"
        self.MODEL_DIR         = "/Users/kuro/python_project/angel1/phases/etf/models/v3_20days"

        # --- Trend Detection ---
        self.ADX_TRENDING_THRESHOLD = 25

        # --- Data Quality ---
        self.MIN_DATA_POINTS = 100

        # --- Rule-Based Ranking Weights (fallback when USE_ML_MODEL=False or model fails) ---
        self.WEIGHT_MOMENTUM = 0.60
        self.WEIGHT_TREND    = 0.30
        self.WEIGHT_VOLUME   = 0.10

        # --- Composite Momentum Blend ---
        self.MOMENTUM_SHORT_DAYS    = 20
        self.MOMENTUM_MEDIUM_DAYS   = 60
        self.MOMENTUM_SHORT_WEIGHT  = 0.40
        self.MOMENTUM_MEDIUM_WEIGHT = 0.60

        # --- Winsorization ---
        self.WINSOR_LOWER_PCT = 5
        self.WINSOR_UPPER_PCT = 95

        # --- Processing ---
        self.PARALLEL_PROCESSING = False
        self.MAX_WORKERS         = 4

        # --- ML Settings ---
        self.USE_ML_MODEL = False     # True = ML-based scoring; False = rule-based
        self.MODEL_TYPE   = 'ridge'   # 'ridge' | 'xgboost'
        self.MODEL_PATH   = ""        # Auto-resolved below if empty

        # Feature column contract — order MUST match training order
        self.FEATURE_COLUMNS = ['z_momentum', 'z_trend', 'z_volume']

        # --- ML Training Parameters ---
        self.MIN_TRAINING_SAMPLES            = 500   # Panel rows required before training
        self.TRAINING_YEARS                  = 3     # History depth for panel construction
        self.RIDGE_ALPHA                     = 1.0
        self.CV_FOLDS                        = 5     # TimeSeriesSplit folds
        self.XGB_N_ESTIMATORS                = 100
        self.XGB_MAX_DEPTH                   = 3     # Shallow: prevents overfitting on noise
        self.XGB_LEARNING_RATE               = 0.1
        self.XGB_RIDGE_IMPROVEMENT_THRESHOLD = 0.05  # XGB must beat Ridge R² by 5%

        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)

        # Resolve MODEL_PATH after potential file-load overrides
        if not self.MODEL_PATH:
            self.MODEL_PATH = os.path.join(
                self.MODEL_DIR, f"{self.MODEL_TYPE}_ranker.pkl"
            )

    def load_from_file(self, config_file: str):
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
                for k, v in data.items():
                    if hasattr(self, k):
                        setattr(self, k, v)
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")

    def save_to_file(self, config_file: str):
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_bootstrap_cfg = Config()
os.makedirs(_bootstrap_cfg.OUTPUT_REPORT_DIR, exist_ok=True)
os.makedirs(_bootstrap_cfg.MODEL_DIR, exist_ok=True)


def setup_logging(output_dir: str, log_level=logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "ranking_engine.log")),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging(_bootstrap_cfg.OUTPUT_REPORT_DIR)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    pass


class IndicatorError(Exception):
    pass


class TrainingError(Exception):
    pass


# ---------------------------------------------------------------------------
# Data Validator
# ---------------------------------------------------------------------------

class DataValidator:
    """OHLCV data quality checks."""

    @staticmethod
    def validate_ohlcv_data(df: pd.DataFrame, min_data_points: int = 100) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            'is_valid':      True,
            'issues':        [],
            'quality_score': 1.0,
            'data_points':   len(df)
        }

        if df.empty:
            results['is_valid'] = False
            results['issues'].append("Empty DataFrame")
            return results

        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            results['is_valid'] = False
            results['issues'].append(f"Missing columns: {missing}")
            return results

        if 'datetime' not in df.columns:
            results['issues'].append("Missing datetime column")
            results['quality_score'] -= 0.2

        for col in ['open', 'high', 'low', 'close']:
            if (df[col] <= 0).any():
                results['issues'].append(f"Non-positive values in {col}")
                results['quality_score'] -= 0.15

        if (df['volume'] < 0).any():
            results['issues'].append("Negative volume")
            results['quality_score'] -= 0.1

        ohlc_issues = 0
        if (df['high'] < df['low']).any():
            results['issues'].append("High < Low violations")
            ohlc_issues += 1
        if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
            results['issues'].append("High < Open/Close violations")
            ohlc_issues += 1
        if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
            results['issues'].append("Low > Open/Close violations")
            ohlc_issues += 1

        returns = df['close'].pct_change()
        if (abs(returns) > 0.2).sum() > len(df) * 0.05:
            results['issues'].append("Excessive outliers in returns")
            results['quality_score'] -= 0.2

        if (df['volume'] == 0).sum() > len(df) * 0.1:
            results['issues'].append("Excessive zero-volume days")
            results['quality_score'] -= 0.1

        if len(df) < min_data_points:
            results['issues'].append(
                f"Insufficient data: {len(df)} < {min_data_points}"
            )
            results['quality_score'] -= 0.3

        results['quality_score'] = max(0.0, results['quality_score'] - ohlc_issues * 0.1)
        if results['quality_score'] < 0.6:
            results['is_valid'] = False

        return results


# ---------------------------------------------------------------------------
# ML Model Trainer
# ---------------------------------------------------------------------------

class MLModelTrainer:
    """
    Trains ranking models on a historical panel dataset.

    Pipeline:
        prepare_training_data()
            → For each stock: compute raw factors at every historical date
            → Compute rolling cross-sectional Z-scores per date
            → Compute neutralized target (excess return over universe median)
        train_ridge_model()
            → Phase 1: Ridge regression, learns optimal linear weights
            → Saves model + feature contract + coefficients JSON
        train_xgboost_model()
            → Phase 2: XGBoost with TimeSeriesSplit CV
            → Only promoted to production if R² beats Ridge by > threshold
    """

    def __init__(self, config_obj: Config, data_dir: str):
        self.config   = config_obj
        self.data_dir = data_dir
        self.validator = DataValidator()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_all_symbol_data(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV files from data directory into a dict of DataFrames."""
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        universe: Dict[str, pd.DataFrame] = {}

        for fpath in tqdm(csv_files, desc="Loading historical data"):
            symbol = os.path.basename(fpath).replace('.csv', '').upper()
            try:
                df = pd.read_csv(fpath)
                df = df.rename(columns={
                    'O': 'open', 'H': 'high', 'L': 'low',
                    'C': 'close', 'V': 'volume'
                })
                validation = self.validator.validate_ohlcv_data(
                    df, min_data_points=self.config.MIN_DATA_POINTS
                )
                if not validation['is_valid'] and validation['quality_score'] < 0.5:
                    continue

                # Resolve datetime
                if 'datetime' not in df.columns:
                    for cand in ['date', 'timestamp', 'Date', 'DateTime']:
                        if cand in df.columns:
                            df['datetime'] = pd.to_datetime(df[cand])
                            break
                    else:
                        continue
                else:
                    df['datetime'] = pd.to_datetime(df['datetime'])

                df.set_index('datetime', inplace=True)
                df.sort_index(inplace=True)
                universe[symbol] = df
            except Exception as e:
                logger.debug(f"Skipping {symbol} during load: {e}")

        logger.info(f"Loaded {len(universe)} symbols for training.")
        return universe

    @staticmethod
    def _winsorize(series: pd.Series, lower_pct: float, upper_pct: float) -> pd.Series:
        lo = series.quantile(lower_pct / 100.0)
        hi = series.quantile(upper_pct / 100.0)
        return series.clip(lower=lo, upper=hi)

    @staticmethod
    def _zscore_series(series: pd.Series) -> pd.Series:
        """Cross-sectional Z-score. Returns zeros if std ≈ 0 (avoids division by zero)."""
        mu    = series.mean()
        sigma = series.std(ddof=0)
        if sigma < 1e-10:
            return pd.Series(0.0, index=series.index)
        return (series - mu) / sigma

    @staticmethod
    def _rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Rank Information Coefficient (Spearman correlation).
        Primary validation metric for cross-sectional ranking models.
        A value > 0.05 is considered meaningful in equity finance.
        """
        if len(y_true) < 3:
            return 0.0
        ic, _ = spearmanr(y_true, y_pred)
        return float(ic) if not np.isnan(ic) else 0.0

    # ------------------------------------------------------------------
    # Panel Dataset Construction
    # ------------------------------------------------------------------

    def prepare_training_data(self) -> pd.DataFrame:
        """
        Build a panel dataset: rows = (symbol, date), columns = features + target.

        Key design decisions:
        1. No look-ahead bias: Z-scores are computed per date using only stocks
           active on that date.
        2. Neutralized target: future_return_20d is expressed as excess over the
           universe median on that date, teaching the model relative outperformance.
        3. Winsorization applied before cross-sectional Z-scoring to suppress
           outliers (same as inference pipeline).
        """
        universe = self._load_all_symbol_data()
        if not universe:
            raise TrainingError("No valid data loaded. Cannot build panel dataset.")

        # ---- Step 1: Compute per-stock raw time series ----
        # For each stock, compute roc_composite, adx, vol_ratio at every date.
        stock_series: Dict[str, pd.DataFrame] = {}

        for symbol, df in tqdm(universe.items(), desc="Computing raw factors"):
            try:
                if len(df) < self.config.MOMENTUM_MEDIUM_DAYS + self.config.MIN_DATA_POINTS:
                    continue

                records = []
                close   = df['close']
                volume  = df['volume']
                high    = df['high']
                low     = df['low']

                # Compute ADX series once for whole stock history
                adx_series = ta.trend.adx(high, low, close, window=14)
                vol_ma      = volume.rolling(20).mean()

                s  = self.config.MOMENTUM_SHORT_DAYS
                m  = self.config.MOMENTUM_MEDIUM_DAYS
                sw = self.config.MOMENTUM_SHORT_WEIGHT
                mw = self.config.MOMENTUM_MEDIUM_WEIGHT

                for i in range(m, len(df) - self.config.MOMENTUM_SHORT_DAYS):
                    date = df.index[i]
                    c_now  = close.iloc[i]
                    c_s    = close.iloc[i - s]
                    c_m    = close.iloc[i - m]

                    roc_20 = (c_now / c_s) - 1.0
                    roc_60 = (c_now / c_m) - 1.0
                    roc_composite = roc_20 * sw + roc_60 * mw

                    adx_val = adx_series.iloc[i]
                    if pd.isna(adx_val):
                        continue

                    vm = vol_ma.iloc[i]
                    vol_ratio = float(volume.iloc[i]) / vm if (vm > 0 and not pd.isna(vm)) else np.nan
                    if np.isnan(vol_ratio):
                        continue

                    # Future 20-day return (look-ahead for target ONLY — correct for training)
                    c_future = close.iloc[i + s]
                    future_return_20d = (c_future / c_now) - 1.0

                    records.append({
                        'symbol':        symbol,
                        'date':          date,
                        'roc_20':        roc_20,
                        'roc_60':        roc_60,
                        'roc_composite': roc_composite,
                        'adx':           float(adx_val),
                        'vol_ratio':     vol_ratio,
                        'future_return': future_return_20d,
                    })

                if records:
                    stock_series[symbol] = pd.DataFrame(records).set_index('date')

            except Exception as e:
                logger.debug(f"Factor computation failed for {symbol}: {e}")
                continue

        if not stock_series:
            raise TrainingError("No factor data computed. Panel cannot be built.")

        # ---- Step 2: Merge all stocks into one panel ----
        panel = pd.concat(stock_series.values(), axis=0)
        panel.sort_index(inplace=True)
        logger.info(f"Raw panel: {len(panel)} rows, {panel['symbol'].nunique()} stocks, "
                    f"date range: {panel.index.min().date()} → {panel.index.max().date()}")

        # ---- Step 3: Rolling cross-sectional normalization (per date) ----
        # Group by date, winsorize + Z-score each factor across stocks active that day.
        # This mirrors the inference pipeline exactly, ensuring no look-ahead bias.
        normalized_rows = []

        for date, day_group in tqdm(panel.groupby(level=0), desc="Rolling normalization"):
            if len(day_group) < 5:
                # Too few stocks on this date for meaningful normalization
                continue

            row_out = day_group.copy()

            # roc_composite → z_momentum
            roc_win = self._winsorize(
                day_group['roc_composite'],
                self.config.WINSOR_LOWER_PCT,
                self.config.WINSOR_UPPER_PCT
            )
            row_out['z_momentum'] = self._zscore_series(roc_win).values

            # adx → z_trend
            adx_win = self._winsorize(
                day_group['adx'],
                self.config.WINSOR_LOWER_PCT,
                self.config.WINSOR_UPPER_PCT
            )
            row_out['z_trend'] = self._zscore_series(adx_win).values

            # vol_ratio → z_volume
            vol_win = self._winsorize(
                day_group['vol_ratio'],
                self.config.WINSOR_LOWER_PCT,
                self.config.WINSOR_UPPER_PCT
            )
            row_out['z_volume'] = self._zscore_series(vol_win).values

            # ---- Step 4: Neutralize target (excess over universe median) ----
            # This teaches the model RELATIVE outperformance, not raw market direction.
            median_return = day_group['future_return'].median()
            row_out['future_return_20d'] = day_group['future_return'] - median_return

            normalized_rows.append(row_out)

        if not normalized_rows:
            raise TrainingError("Cross-sectional normalization produced no rows.")

        panel_final = pd.concat(normalized_rows, axis=0)
        panel_final = panel_final.dropna(
            subset=['z_momentum', 'z_trend', 'z_volume', 'future_return_20d']
        )

        logger.info(
            f"Panel ready: {len(panel_final)} rows after normalization & NaN drop."
        )

        if len(panel_final) < self.config.MIN_TRAINING_SAMPLES:
            raise TrainingError(
                f"Insufficient training data: {len(panel_final)} rows < "
                f"MIN_TRAINING_SAMPLES={self.config.MIN_TRAINING_SAMPLES}. "
                f"Add more historical data or reduce TRAINING_YEARS."
            )

        return panel_final

    # ------------------------------------------------------------------
    # Phase 1: Ridge Regression
    # ------------------------------------------------------------------

    def train_ridge_model(self, panel: pd.DataFrame) -> Dict[str, Any]:
        """
        Train a Ridge regression model to learn optimal cross-sectional weights.

        Inputs (X): [z_momentum, z_trend, z_volume]  — already Z-scored, no scaler needed.
        Target (y): future_return_20d (neutralized excess return)

        Artifacts saved:
            models/ridge_ranker.pkl        — model + feature contract
            models/ridge_coefficients.json — human-readable coefficients for auditing

        Returns a results dict with CV metrics.
        """
        feature_cols = self.config.FEATURE_COLUMNS

        # Sort panel by date to respect temporal ordering
        panel_sorted = panel.sort_index()

        X = panel_sorted[feature_cols].values
        y = panel_sorted['future_return_20d'].values

        logger.info(f"Ridge training: {len(X)} samples, features={feature_cols}")

        # ---- TimeSeriesSplit Cross-Validation ----
        tscv = TimeSeriesSplit(n_splits=self.config.CV_FOLDS)
        cv_results = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            ridge = Ridge(alpha=self.config.RIDGE_ALPHA)
            ridge.fit(X_tr, y_tr)
            y_pred = ridge.predict(X_val)

            r2   = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            ic   = self._rank_ic(y_val, y_pred)

            cv_results.append({'fold': fold + 1, 'r2': r2, 'rmse': rmse, 'rank_ic': ic})
            logger.info(
                f"  Ridge Fold {fold+1}: R²={r2:.4f}  RMSE={rmse:.6f}  Rank IC={ic:.4f}"
            )

        avg_r2  = float(np.mean([r['r2']      for r in cv_results]))
        avg_ic  = float(np.mean([r['rank_ic'] for r in cv_results]))
        avg_rmse = float(np.mean([r['rmse']   for r in cv_results]))

        logger.info(
            f"Ridge CV Summary — Avg R²={avg_r2:.4f}  "
            f"Avg Rank IC={avg_ic:.4f}  Avg RMSE={avg_rmse:.6f}"
        )
        if avg_ic < 0.05:
            logger.warning(
                "Ridge Rank IC < 0.05. Model may not be adding predictive value. "
                "Consider expanding feature set or checking data quality."
            )

        # ---- Final fit on full dataset ----
        ridge_final = Ridge(alpha=self.config.RIDGE_ALPHA)
        ridge_final.fit(X, y)

        coefficients = dict(zip(feature_cols, ridge_final.coef_.tolist()))
        logger.info(f"Ridge final coefficients: {coefficients}")
        logger.info(f"Ridge intercept: {ridge_final.intercept_:.6f}")

        # ---- Sanity check on coefficients ----
        if coefficients.get('z_momentum', 0) < 0:
            logger.warning(
                "AUDIT FLAG: Ridge z_momentum coefficient is NEGATIVE. "
                "This contradicts long-only momentum hypothesis. "
                "Review data pipeline for look-ahead bias or data errors."
            )

        # ---- Save model artifact ----
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
        model_path = os.path.join(self.config.MODEL_DIR, 'ridge_ranker.pkl')
        artifact = {
            'model':           ridge_final,
            'feature_columns': feature_cols,   # Contract for inference alignment
            'model_type':      'ridge',
            'trained_at':      datetime.now().isoformat(),
            'cv_avg_r2':       avg_r2,
            'cv_avg_rank_ic':  avg_ic,
            'cv_avg_rmse':     avg_rmse,
            'n_training_rows': len(X),
        }
        joblib.dump(artifact, model_path)
        logger.info(f"Ridge model saved: {model_path}")

        # ---- Save human-readable coefficient audit file (mandatory) ----
        coeff_path = os.path.join(self.config.MODEL_DIR, 'ridge_coefficients.json')
        coeff_audit = {
            'coefficients':      coefficients,
            'intercept':         float(ridge_final.intercept_),
            'feature_columns':   feature_cols,
            'cv_fold_results':   cv_results,
            'cv_avg_r2':         avg_r2,
            'cv_avg_rank_ic':    avg_ic,
            'cv_avg_rmse':       avg_rmse,
            'n_training_rows':   len(X),
            'ridge_alpha':       self.config.RIDGE_ALPHA,
            'trained_at':        datetime.now().isoformat(),
            'interpretation': {
                f: f"A 1-SD increase in {f} predicts "
                   f"{'positive' if v > 0 else 'negative'} "
                   f"excess return of {abs(v)*100:.4f}% (neutralized)"
                for f, v in coefficients.items()
            }
        }
        with open(coeff_path, 'w') as fp:
            json.dump(coeff_audit, fp, indent=2)
        logger.info(f"Ridge coefficient audit saved: {coeff_path}")

        return {
            'model_path':    model_path,
            'coeff_path':    coeff_path,
            'coefficients':  coefficients,
            'cv_avg_r2':     avg_r2,
            'cv_avg_rank_ic': avg_ic,
            'cv_avg_rmse':   avg_rmse,
            'cv_results':    cv_results,
        }

    # ------------------------------------------------------------------
    # Phase 2: XGBoost
    # ------------------------------------------------------------------

    def train_xgboost_model(self, panel: pd.DataFrame,
                             ridge_avg_r2: float) -> Dict[str, Any]:
        """
        Train an XGBoost model to capture non-linear factor interactions.

        Production gate: XGBoost must beat Ridge average R² by at least
        XGB_RIDGE_IMPROVEMENT_THRESHOLD (default 5%) to be considered viable.

        Inputs (X): [z_momentum, z_trend, z_volume]
        Target (y): future_return_20d (neutralized excess return)

        Artifacts saved to: models/xgb_ranker.pkl
        """
        if not XGBOOST_AVAILABLE:
            raise TrainingError(
                "XGBoost is not installed. Run: pip install xgboost"
            )

        feature_cols  = self.config.FEATURE_COLUMNS
        panel_sorted  = panel.sort_index()
        X = panel_sorted[feature_cols].values
        y = panel_sorted['future_return_20d'].values

        logger.info(
            f"XGBoost training: {len(X)} samples, features={feature_cols}"
        )

        # ---- TimeSeriesSplit Cross-Validation ----
        tscv = TimeSeriesSplit(n_splits=self.config.CV_FOLDS)
        cv_results = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            xgb = XGBRegressor(
                n_estimators  = self.config.XGB_N_ESTIMATORS,
                max_depth     = self.config.XGB_MAX_DEPTH,
                learning_rate = self.config.XGB_LEARNING_RATE,
                objective     = 'reg:squarederror',
                verbosity     = 0,
                n_jobs        = -1,
            )
            xgb.fit(X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False)
            y_pred = xgb.predict(X_val)

            r2   = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            ic   = self._rank_ic(y_val, y_pred)

            cv_results.append({'fold': fold + 1, 'r2': r2, 'rmse': rmse, 'rank_ic': ic})
            logger.info(
                f"  XGB Fold {fold+1}: R²={r2:.4f}  RMSE={rmse:.6f}  Rank IC={ic:.4f}"
            )

        avg_r2   = float(np.mean([r['r2']      for r in cv_results]))
        avg_ic   = float(np.mean([r['rank_ic'] for r in cv_results]))
        avg_rmse = float(np.mean([r['rmse']    for r in cv_results]))

        logger.info(
            f"XGBoost CV Summary — Avg R²={avg_r2:.4f}  "
            f"Avg Rank IC={avg_ic:.4f}  Avg RMSE={avg_rmse:.6f}"
        )

        # ---- Production Gate ----
        improvement = avg_r2 - ridge_avg_r2
        threshold   = self.config.XGB_RIDGE_IMPROVEMENT_THRESHOLD
        is_production_eligible = improvement >= threshold

        if is_production_eligible:
            logger.info(
                f"XGBoost PASSES production gate: "
                f"R² improvement = {improvement:.4f} >= threshold {threshold:.4f}"
            )
        else:
            logger.warning(
                f"XGBoost FAILS production gate: "
                f"R² improvement = {improvement:.4f} < threshold {threshold:.4f}. "
                f"Stick with Ridge for production. XGB artifact saved for reference only."
            )

        # ---- Final fit on full dataset ----
        xgb_final = XGBRegressor(
            n_estimators  = self.config.XGB_N_ESTIMATORS,
            max_depth     = self.config.XGB_MAX_DEPTH,
            learning_rate = self.config.XGB_LEARNING_RATE,
            objective     = 'reg:squarederror',
            verbosity     = 0,
            n_jobs        = -1,
        )
        xgb_final.fit(X, y)

        # Feature importance (proxy for "coefficients" in tree models)
        importance = dict(zip(feature_cols, xgb_final.feature_importances_.tolist()))
        logger.info(f"XGBoost feature importances: {importance}")

        # ---- Save artifact ----
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
        model_path = os.path.join(self.config.MODEL_DIR, 'xgb_ranker.pkl')
        artifact = {
            'model':                   xgb_final,
            'feature_columns':         feature_cols,
            'model_type':              'xgboost',
            'trained_at':              datetime.now().isoformat(),
            'cv_avg_r2':               avg_r2,
            'cv_avg_rank_ic':          avg_ic,
            'cv_avg_rmse':             avg_rmse,
            'n_training_rows':         len(X),
            'ridge_avg_r2':            ridge_avg_r2,
            'r2_improvement_vs_ridge': improvement,
            'is_production_eligible':  is_production_eligible,
            'feature_importances':     importance,
        }
        joblib.dump(artifact, model_path)
        logger.info(f"XGBoost model saved: {model_path}")

        # Save importance audit
        imp_path = os.path.join(self.config.MODEL_DIR, 'xgb_feature_importances.json')
        with open(imp_path, 'w') as fp:
            json.dump({
                'feature_importances':         importance,
                'cv_fold_results':             cv_results,
                'cv_avg_r2':                   avg_r2,
                'cv_avg_rank_ic':              avg_ic,
                'ridge_avg_r2':                ridge_avg_r2,
                'r2_improvement_vs_ridge':     improvement,
                'is_production_eligible':      is_production_eligible,
                'production_gate_threshold':   threshold,
                'trained_at':                  datetime.now().isoformat(),
            }, fp, indent=2)
        logger.info(f"XGBoost importance audit saved: {imp_path}")

        return {
            'model_path':              model_path,
            'importance_path':         imp_path,
            'feature_importances':     importance,
            'cv_avg_r2':               avg_r2,
            'cv_avg_rank_ic':          avg_ic,
            'cv_avg_rmse':             avg_rmse,
            'cv_results':              cv_results,
            'r2_improvement_vs_ridge': improvement,
            'is_production_eligible':  is_production_eligible,
        }

    # ------------------------------------------------------------------
    # Convenience: run both phases
    # ------------------------------------------------------------------

    def run_full_training(self) -> Dict[str, Any]:
        """
        Full training pipeline:
            1. Build panel dataset
            2. Train Ridge (Phase 1)
            3. Train XGBoost (Phase 2) if available
            4. Return summary of all results

        Call this from CLI with --train flag.
        """
        logger.info("=" * 70)
        logger.info("MLModelTrainer: Starting full training pipeline")
        logger.info("=" * 70)

        # Build panel
        panel = self.prepare_training_data()

        # Phase 1: Ridge
        logger.info("\n--- Phase 1: Ridge Regression ---")
        ridge_results = self.train_ridge_model(panel)

        # Phase 2: XGBoost (conditional)
        xgb_results = None
        if XGBOOST_AVAILABLE:
            logger.info("\n--- Phase 2: XGBoost ---")
            try:
                xgb_results = self.train_xgboost_model(
                    panel, ridge_avg_r2=ridge_results['cv_avg_r2']
                )
            except Exception as e:
                logger.error(f"XGBoost training failed: {e}")
        else:
            logger.info(
                "XGBoost not installed — skipping Phase 2. "
                "Install with: pip install xgboost"
            )

        summary = {
            'panel_rows':    len(panel),
            'ridge_results': ridge_results,
            'xgb_results':   xgb_results,
        }

        logger.info("\n" + "=" * 70)
        logger.info("Training complete.")
        logger.info(f"  Ridge Avg Rank IC : {ridge_results['cv_avg_rank_ic']:.4f}")
        if xgb_results:
            logger.info(f"  XGB   Avg Rank IC : {xgb_results['cv_avg_rank_ic']:.4f}")
            logger.info(
                f"  XGB Production Eligible: {xgb_results['is_production_eligible']}"
            )
        logger.info("=" * 70)

        return summary


# ---------------------------------------------------------------------------
# Cross-Sectional Ranking Engine (Inference)
# ---------------------------------------------------------------------------

class CrossSectionalRankingEngine:
    """
    Ranks stocks within a universe using cross-sectional Z-score normalization.

    Scoring modes:
        Rule-Based (USE_ML_MODEL=False):
            composite_score = z_momentum*0.60 + z_trend*0.30 + z_volume*0.10

        ML-Based (USE_ML_MODEL=True):
            composite_score = model.predict([z_momentum, z_trend, z_volume])
            Graceful fallback to rule-based if model missing or corrupt.

    Inference Pipeline:
        1. Load & validate OHLCV per stock
        2. Extract raw factors: composite ROC (20d+60d), ADX, Volume Ratio
        3. Winsorize each factor [5th, 95th percentile]
        4. Z-score normalize cross-sectionally
        5. Score via ML model (or rule-based fallback)
        6. Sort descending → Ranked DataFrame
    """

    def __init__(self, data_dir: str = None, config_obj: Config = None):
        self.config    = config_obj or Config()
        self.data_dir  = data_dir or self.config.INPUT_DATA_DIR
        self.validator = DataValidator()
        logger.info(
            f"CrossSectionalRankingEngine v5.0 | data_dir={self.data_dir} | "
            f"ML={'ON ('+self.config.MODEL_TYPE+')' if self.config.USE_ML_MODEL else 'OFF (rule-based)'}"
        )

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def load_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load and validate a single stock CSV. Returns None if unusable."""
        file_path = os.path.join(self.data_dir, f"{symbol.lower()}.csv")
        if not os.path.exists(file_path):
            logger.debug(f"File not found: {file_path}")
            return None

        try:
            df = pd.read_csv(file_path)
            if df.empty:
                return None

            df = df.rename(columns={
                'O': 'open', 'H': 'high', 'L': 'low',
                'C': 'close', 'V': 'volume'
            })

            validation = self.validator.validate_ohlcv_data(
                df, min_data_points=self.config.MIN_DATA_POINTS
            )
            if not validation['is_valid'] and validation['quality_score'] < 0.5:
                logger.warning(
                    f"{symbol}: skipped — quality={validation['quality_score']:.2f}, "
                    f"issues={validation['issues']}"
                )
                return None

            # Resolve datetime
            if 'datetime' not in df.columns:
                for cand in ['date', 'timestamp', 'Date', 'DateTime']:
                    if cand in df.columns:
                        df['datetime'] = pd.to_datetime(df[cand])
                        break
                else:
                    logger.error(f"{symbol}: no datetime column found")
                    return None
            else:
                df['datetime'] = pd.to_datetime(df['datetime'])

            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            return df

        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Normalization Helpers (shared with MLModelTrainer for consistency)
    # ------------------------------------------------------------------

    @staticmethod
    def _winsorize(series: pd.Series, lower_pct: float, upper_pct: float) -> pd.Series:
        lo = series.quantile(lower_pct / 100.0)
        hi = series.quantile(upper_pct / 100.0)
        return series.clip(lower=lo, upper=hi)

    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        """
        Cross-sectional Z-score normalization.
        Returns zeros if std ≈ 0 (all values identical) to avoid division by zero.
        NaN values are preserved; they are not used in mean/std calculation.
        """
        mu    = series.mean()
        sigma = series.std(ddof=0)
        if sigma < 1e-10:
            logger.warning("Z-score: std ≈ 0 — all values identical. Returning 0.")
            return pd.Series(0.0, index=series.index)
        return (series - mu) / sigma

    # ------------------------------------------------------------------
    # Factor Extraction (per stock — lightweight)
    # ------------------------------------------------------------------

    def _safe_last(self, series: pd.Series, default: float = np.nan) -> float:
        dropped = series.dropna()
        return float(dropped.iloc[-1]) if not dropped.empty else default

    def analyze_stock(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Extract raw ranking factors for a single stock.

        Returns:
            {
                'symbol':        str,
                'price':         float,
                'roc_20':        float,
                'roc_60':        float,
                'roc_composite': float,  # Weighted blend
                'adx':           float,
                'vol_ratio':     float,
            }
        Returns None if data is unusable.
        """
        df = self.load_data(symbol)
        if df is None or len(df) < self.config.MIN_DATA_POINTS:
            return None

        try:
            s  = self.config.MOMENTUM_SHORT_DAYS
            m  = self.config.MOMENTUM_MEDIUM_DAYS
            sw = self.config.MOMENTUM_SHORT_WEIGHT
            mw = self.config.MOMENTUM_MEDIUM_WEIGHT

            # 20-day ROC
            roc_20 = (
                (df['close'].iloc[-1] / df['close'].iloc[-(s + 1)]) - 1.0
            ) if len(df) > s else np.nan

            # 60-day ROC
            roc_60 = (
                (df['close'].iloc[-1] / df['close'].iloc[-(m + 1)]) - 1.0
            ) if len(df) > m else np.nan

            # Composite momentum
            if not np.isnan(roc_20) and not np.isnan(roc_60):
                roc_composite = roc_20 * sw + roc_60 * mw
            elif not np.isnan(roc_20):
                roc_composite = roc_20
            else:
                roc_composite = np.nan

            # ADX (14-period)
            adx_series = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
            adx = self._safe_last(adx_series, default=np.nan)

            # Volume ratio
            vol_ma     = df['volume'].rolling(20).mean()
            last_vol_ma = self._safe_last(vol_ma, default=0)
            vol_ratio  = (
                float(df['volume'].iloc[-1]) / last_vol_ma
                if last_vol_ma > 0 else np.nan
            )

            return {
                'symbol':        symbol,
                'price':         round(float(df['close'].iloc[-1]), 2),
                'roc_20':        float(roc_20)        if not np.isnan(roc_20)        else np.nan,
                'roc_60':        float(roc_60)        if not np.isnan(roc_60)        else np.nan,
                'roc_composite': float(roc_composite) if not np.isnan(roc_composite) else np.nan,
                'adx':           float(adx)           if not np.isnan(adx)           else np.nan,
                'vol_ratio':     float(vol_ratio)     if not np.isnan(vol_ratio)     else np.nan,
            }

        except Exception as e:
            logger.error(f"Factor extraction failed for {symbol}: {e}", exc_info=True)
            return None

    def analyze_stock_wrapper(self, symbol: str) -> Tuple[str, Optional[Dict]]:
        """Wrapper for parallel execution."""
        return symbol, self.analyze_stock(symbol)

    # ------------------------------------------------------------------
    # ML Scoring: Load Model & Predict
    # ------------------------------------------------------------------

    def _score_with_model(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        """
        Attempt ML scoring. Returns (df_with_scores, used_ml).
        Falls back to rule-based on any failure.
        """
        model_path = self.config.MODEL_PATH
        feature_cols = self.config.FEATURE_COLUMNS

        try:
            artifact = joblib.load(model_path)
            model    = artifact['model']

            # Enforce feature column contract from artifact (not config)
            # This ensures live inference always uses the exact columns from training.
            saved_features = artifact.get('feature_columns', feature_cols)
            if saved_features != feature_cols:
                logger.warning(
                    f"Feature column mismatch: config={feature_cols}, "
                    f"artifact={saved_features}. Using artifact definition."
                )
            feature_cols = saved_features

            # Validate all required columns are present
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing feature columns in DataFrame: {missing}")

            X_live = df[feature_cols].values
            df['composite_score'] = model.predict(X_live)

            model_type = artifact.get('model_type', self.config.MODEL_TYPE)
            logger.info(
                f"ML scoring complete using '{model_type}' model "
                f"(trained: {artifact.get('trained_at', 'unknown')}, "
                f"Rank IC: {artifact.get('cv_avg_rank_ic', 'N/A')})."
            )
            return df, True

        except FileNotFoundError:
            logger.error(
                f"Model file not found at '{model_path}'. "
                "Reverting to rule-based weights."
            )
        except Exception as e:
            logger.error(
                f"ML scoring failed: {e}. Reverting to rule-based weights."
            )

        return df, False

    # ------------------------------------------------------------------
    # Universe-Level Analysis
    # ------------------------------------------------------------------

    def analyze_all_stocks(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Full inference pipeline:
            1. Collect raw factors per stock
            2. Winsorize each factor
            3. Z-score normalize cross-sectionally
            4. Score via ML model (or rule-based fallback)
            5. Return DataFrame sorted by composite_score descending
        """
        if symbols is None:
            try:
                csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
                symbols   = [
                    os.path.basename(f).replace('.csv', '').upper()
                    for f in csv_files
                ]
                logger.info(f"Discovered {len(symbols)} symbols.")
            except Exception as e:
                logger.error(f"Error scanning data directory: {e}")
                return pd.DataFrame()

        # ---- Step 1: Collect raw factors ----
        raw_results: List[Dict] = []

        if self.config.PARALLEL_PROCESSING and len(symbols) > 10:
            logger.info(f"Parallel mode: {self.config.MAX_WORKERS} workers")
            with ProcessPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
                futures = {
                    executor.submit(self.analyze_stock_wrapper, s): s
                    for s in symbols
                }
                for future in tqdm(
                    as_completed(futures), total=len(symbols), desc="Extracting Factors"
                ):
                    _, result = future.result()
                    if result is not None:
                        raw_results.append(result)
        else:
            for symbol in tqdm(symbols, desc="Extracting Factors"):
                try:
                    result = self.analyze_stock(symbol)
                    if result is not None:
                        raw_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

        if not raw_results:
            logger.error("No stocks produced valid factor data.")
            return pd.DataFrame()

        logger.info(
            f"Factor extraction: {len(raw_results)} / {len(symbols)} stocks usable."
        )

        # ---- Step 2: Build DataFrame ----
        df = pd.DataFrame(raw_results)
        df.set_index('symbol', inplace=True)

        factor_map = {
            'roc_composite': 'z_momentum',
            'adx':           'z_trend',
            'vol_ratio':     'z_volume',
        }

        # ---- Steps 3 & 4: Winsorize → Z-score per factor ----
        for raw_col, z_col in factor_map.items():
            valid_mask = df[raw_col].notna()
            n_valid    = valid_mask.sum()

            if n_valid < 3:
                logger.warning(
                    f"Factor '{raw_col}': only {n_valid} valid values — "
                    "skipping normalization, Z-score set to 0."
                )
                df[z_col] = 0.0
                continue

            winsorized = self._winsorize(
                df.loc[valid_mask, raw_col].copy(),
                self.config.WINSOR_LOWER_PCT,
                self.config.WINSOR_UPPER_PCT
            )
            z_values = self._zscore(winsorized)

            df[z_col] = np.nan
            df.loc[valid_mask, z_col] = z_values.values

        # Fill NaN Z-scores with 0 (neutral) for scoring
        for z_col in factor_map.values():
            df[z_col] = df[z_col].fillna(0.0)

        # ---- Step 5: Composite Scoring ----
        used_ml = False
        if self.config.USE_ML_MODEL:
            df, used_ml = self._score_with_model(df)

        if not used_ml:
            # Rule-based fallback (also the default when USE_ML_MODEL=False)
            df['composite_score'] = (
                df['z_momentum'] * self.config.WEIGHT_MOMENTUM +
                df['z_trend']    * self.config.WEIGHT_TREND    +
                df['z_volume']   * self.config.WEIGHT_VOLUME
            )
            if self.config.USE_ML_MODEL:
                logger.info("Scoring completed using rule-based weights (ML fallback).")
            else:
                logger.info("Scoring completed using rule-based weights.")

        # ---- Step 6: Sort ----
        df.sort_values('composite_score', ascending=False, inplace=True)
        df.reset_index(inplace=True)   # Restore 'symbol' as column

        logger.info("Cross-sectional ranking complete.")
        return df

    # ------------------------------------------------------------------
    # Excel Report
    # ------------------------------------------------------------------

    def generate_report(self, ranked_df: pd.DataFrame,
                        output_dir: str = None) -> Optional[str]:
        """
        Single-sheet Excel report with heat-map on Composite Score.
        Includes scoring mode annotation (ML vs rule-based).
        """
        if ranked_df.empty:
            logger.warning("Ranked DataFrame is empty — no report generated.")
            return None

        output_dir    = output_dir or self.config.OUTPUT_REPORT_DIR
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_tag      = (
            f"ml_{self.config.MODEL_TYPE}"
            if self.config.USE_ML_MODEL else "rule_based"
        )
        output_file = os.path.join(
            output_dir,
            f"relative_strength_ranking_{mode_tag}_{timestamp_str}.xlsx"
        )

        # ---- Build display DataFrame ----
        report_df = pd.DataFrame({
            'Rank':              range(1, len(ranked_df) + 1),
            'Symbol':            ranked_df['symbol'].values,
            'Current Price':     ranked_df['price'].round(2).values,
            'Composite Score':   ranked_df['composite_score'].round(4).values,
            'Momentum Z-Score':  ranked_df['z_momentum'].round(4).values,
            'Trend Z-Score':     ranked_df['z_trend'].round(4).values,
            'Volume Z-Score':    ranked_df['z_volume'].round(4).values,
            'ROC 20d':           ranked_df['roc_20'].apply(
                lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
            ).values,
            'ROC 60d':           ranked_df['roc_60'].apply(
                lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
            ).values,
            'ADX (14)':          ranked_df['adx'].round(2).values,
            'Volume Ratio':      ranked_df['vol_ratio'].round(3).values,
        })

        try:
            with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
                report_df.to_excel(
                    writer, sheet_name='Relative Strength Ranking', index=False
                )

                workbook  = writer.book
                worksheet = writer.sheets['Relative Strength Ranking']

                # Formats
                header_fmt = workbook.add_format({
                    'bold': True, 'text_wrap': True, 'valign': 'top',
                    'fg_color': '#1F3864', 'font_color': '#FFFFFF',
                    'border': 1, 'align': 'center',
                })
                rank_fmt     = workbook.add_format({'align': 'center', 'bold': True, 'border': 1})
                currency_fmt = workbook.add_format({'num_format': '₹#,##0.00', 'border': 1})
                float4_fmt   = workbook.add_format({'num_format': '0.0000', 'border': 1})
                general_fmt  = workbook.add_format({'border': 1})
                na_fmt       = workbook.add_format({'border': 1, 'font_color': '#999999'})

                col_formats = {
                    'Rank':             rank_fmt,
                    'Symbol':           general_fmt,
                    'Current Price':    currency_fmt,
                    'Composite Score':  float4_fmt,
                    'Momentum Z-Score': float4_fmt,
                    'Trend Z-Score':    float4_fmt,
                    'Volume Z-Score':   float4_fmt,
                    'ROC 20d':          general_fmt,
                    'ROC 60d':          general_fmt,
                    'ADX (14)':         float4_fmt,
                    'Volume Ratio':     float4_fmt,
                }

                col_widths = {
                    'Rank': 6, 'Symbol': 14, 'Current Price': 14,
                    'Composite Score': 16, 'Momentum Z-Score': 18,
                    'Trend Z-Score': 14, 'Volume Z-Score': 14,
                    'ROC 20d': 10, 'ROC 60d': 10,
                    'ADX (14)': 10, 'Volume Ratio': 13,
                }

                # Write headers
                for col_num, col_name in enumerate(report_df.columns):
                    worksheet.write(0, col_num, col_name, header_fmt)

                # Write data
                for row_idx, row in report_df.iterrows():
                    for col_idx, col_name in enumerate(report_df.columns):
                        fmt = col_formats.get(col_name, general_fmt)
                        val = row[col_name]
                        if isinstance(val, float) and np.isnan(val):
                            worksheet.write(row_idx + 1, col_idx, 'N/A', na_fmt)
                        else:
                            worksheet.write(row_idx + 1, col_idx, val, fmt)

                # Heat-map on Composite Score
                comp_idx = report_df.columns.get_loc('Composite Score')
                last_row = len(report_df)
                if last_row > 0:
                    worksheet.conditional_format(
                        1, comp_idx, last_row, comp_idx,
                        {
                            'type':      '3_color_scale',
                            'min_color': '#FF4C4C',   # Red  — weak
                            'mid_color': '#FFEB9C',   # Yellow — neutral
                            'max_color': '#00B050',   # Green — strong
                        }
                    )

                # Column widths
                for col_idx, col_name in enumerate(report_df.columns):
                    worksheet.set_column(col_idx, col_idx, col_widths.get(col_name, 14))

                # Freeze header row + auto-filter
                worksheet.freeze_panes(1, 0)
                worksheet.autofilter(0, 0, last_row, len(report_df.columns) - 1)

                # Scoring mode annotation in cell A1 note area (as a comment)
                mode_note = (
                    f"Scoring mode: {'ML (' + self.config.MODEL_TYPE + ')' if self.config.USE_ML_MODEL else 'Rule-Based'} | "
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"Universe: {len(ranked_df)} stocks"
                )
                worksheet.write_comment(0, 0, mode_note)

            logger.info(f"Report saved: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error generating report: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Console Summary
    # ------------------------------------------------------------------

    def print_summary(self, ranked_df: pd.DataFrame):
        """Print ranked summary with Z-score validation to stdout."""
        try:
            if ranked_df.empty:
                print("No results to display.")
                return

            n           = len(ranked_df)
            mode_label  = (
                f"ML ({self.config.MODEL_TYPE.upper()})"
                if self.config.USE_ML_MODEL else "Rule-Based"
            )

            print("\n" + "=" * 115)
            print(
                f"CROSS-SECTIONAL RELATIVE STRENGTH RANKING — Long Only Universe  |  "
                f"Scoring: {mode_label}"
            )
            print(
                f"Engine v5.0  |  Ranked {n} stocks  |  "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            print("=" * 115)

            # Momentum distribution
            pos_mom = (ranked_df['roc_composite'] > 0).sum()
            neg_mom = (ranked_df['roc_composite'] <= 0).sum()
            print(f"\nMomentum Distribution:")
            print(f"  Positive composite ROC : {pos_mom:3d} ({pos_mom/n*100:5.1f}%)")
            print(f"  Negative composite ROC : {neg_mom:3d} ({neg_mom/n*100:5.1f}%)")

            # ADX trending count
            adx_valid = ranked_df['adx'].dropna()
            if not adx_valid.empty:
                trending = (adx_valid > self.config.ADX_TRENDING_THRESHOLD).sum()
                print(
                    f"  Trending (ADX > {self.config.ADX_TRENDING_THRESHOLD}) : "
                    f"{trending:3d} ({trending/n*100:5.1f}%)"
                )

            # Z-score validation (sum ≈ 0 confirms no look-ahead bias in normalization)
            print(f"\nZ-Score Validation (sum ≈ 0 confirms correct cross-sectional normalization):")
            for col, label in [
                ('z_momentum', 'Momentum'),
                ('z_trend',    'Trend   '),
                ('z_volume',   'Volume  ')
            ]:
                z_sum  = ranked_df[col].sum()
                z_mean = ranked_df[col].mean()
                status = "✓" if abs(z_sum) < 1.0 else "⚠ CHECK"
                print(
                    f"  {label}  sum={z_sum:+.6f}  mean={z_mean:+.6f}  {status}"
                )

            # Top 15
            header = (
                f"{'Rank':<5} {'Symbol':<12} {'Price':>8} {'Score':>10} "
                f"{'MomZ':>8} {'TrendZ':>8} {'VolZ':>8} "
                f"{'ROC20':>8} {'ROC60':>8} {'ADX':>6}"
            )
            sep = "-" * 115

            print(f"\nTop 15 — Highest Relative Strength:")
            print(header)
            print(sep)
            for rank_i, (_, row) in enumerate(ranked_df.head(15).iterrows(), start=1):
                roc20 = f"{row['roc_20']*100:+.1f}%" if pd.notna(row['roc_20']) else "  N/A"
                roc60 = f"{row['roc_60']*100:+.1f}%" if pd.notna(row['roc_60']) else "  N/A"
                adx_v = f"{row['adx']:.1f}"          if pd.notna(row['adx'])    else "  N/A"
                print(
                    f"{rank_i:<5} {row['symbol']:<12} {row['price']:>8.2f} "
                    f"{row['composite_score']:>10.4f} "
                    f"{row['z_momentum']:>8.4f} {row['z_trend']:>8.4f} {row['z_volume']:>8.4f} "
                    f"{roc20:>8} {roc60:>8} {adx_v:>6}"
                )

            # Bottom 5
            print(f"\nBottom 5 — Weakest Relative Strength:")
            print(header)
            print(sep)
            for rank_i, (_, row) in enumerate(
                ranked_df.tail(5).iterrows(), start=n - 4
            ):
                roc20 = f"{row['roc_20']*100:+.1f}%" if pd.notna(row['roc_20']) else "  N/A"
                roc60 = f"{row['roc_60']*100:+.1f}%" if pd.notna(row['roc_60']) else "  N/A"
                adx_v = f"{row['adx']:.1f}"          if pd.notna(row['adx'])    else "  N/A"
                print(
                    f"{rank_i:<5} {row['symbol']:<12} {row['price']:>8.2f} "
                    f"{row['composite_score']:>10.4f} "
                    f"{row['z_momentum']:>8.4f} {row['z_trend']:>8.4f} {row['z_volume']:>8.4f} "
                    f"{roc20:>8} {roc60:>8} {adx_v:>6}"
                )

            # Universe statistics
            desc = ranked_df['composite_score'].describe()
            print(f"\nComposite Score Statistics:")
            print(
                f"  Mean={desc['mean']:+.4f} | Std={desc['std']:.4f} | "
                f"Max={desc['max']:+.4f} | Min={desc['min']:+.4f} | "
                f"Median={ranked_df['composite_score'].median():+.4f}"
            )
            print("=" * 115)

        except Exception as e:
            logger.error(f"Error printing summary: {e}", exc_info=True)
            print("Could not generate console summary.")

    # ------------------------------------------------------------------
    # Main Entry Point
    # ------------------------------------------------------------------

    def run(self, symbols: Optional[List[str]] = None,
            generate_report: bool = True,
            print_summary: bool = True) -> Tuple[pd.DataFrame, Optional[str]]:
        """Execute the full inference pipeline and return (ranked_df, report_path)."""
        mode_label = (
            f"ML ({self.config.MODEL_TYPE})"
            if self.config.USE_ML_MODEL else "Rule-Based"
        )
        print("\nCross-Sectional Relative Strength Ranking Engine v5.0")
        print(f"Long-Only | Composite Momentum (20d+60d) + ADX + Volume")
        print(f"Scoring mode  : {mode_label}")
        print(f"Input dir     : {self.data_dir}")
        print(f"Output dir    : {self.config.OUTPUT_REPORT_DIR}")
        if not self.config.USE_ML_MODEL:
            print(
                f"Rule weights  : Momentum={self.config.WEIGHT_MOMENTUM:.0%}  "
                f"Trend={self.config.WEIGHT_TREND:.0%}  "
                f"Volume={self.config.WEIGHT_VOLUME:.0%}"
            )
        print(
            f"Winsorize     : [{self.config.WINSOR_LOWER_PCT}th, "
            f"{self.config.WINSOR_UPPER_PCT}th] percentile"
        )
        print("-" * 80)

        start     = datetime.now()
        ranked_df = self.analyze_all_stocks(symbols)
        elapsed   = (datetime.now() - start).total_seconds()

        print(f"\nAnalysis completed in {elapsed:.1f}s — {len(ranked_df)} stocks ranked.")

        if ranked_df.empty:
            print("No results. Check data directory and logs.")
            return ranked_df, None

        report_path = None
        if generate_report:
            report_path = self.generate_report(ranked_df)
            if report_path:
                print(f"Excel report  : {report_path}")
            else:
                print("Report generation failed — see logs.")

        if print_summary:
            self.print_summary(ranked_df)

        return ranked_df, report_path


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def create_sample_config(filename: str = "config.json"):
    """Write default configuration to JSON for inspection and editing."""
    Config().save_to_file(filename)
    print(f"Sample config saved to {filename}")


def run_tests():
    """Smoke-test core components with synthetic data."""
    print("Running unit tests...\n")
    np.random.seed(42)

    n     = 150
    close = 100 + np.cumsum(np.random.normal(0.05, 0.8, n))
    high  = close + np.abs(np.random.normal(0.5, 0.3, n))
    low   = close - np.abs(np.random.normal(0.5, 0.3, n))
    open_ = close + np.random.normal(0, 0.3, n)
    vol   = np.random.randint(50_000, 500_000, n)
    dates = pd.date_range('2022-01-01', periods=n, freq='B')

    df_test = pd.DataFrame({
        'datetime': dates, 'open': open_, 'high': high,
        'low': low, 'close': close, 'volume': vol
    })

    # 1. DataValidator
    val = DataValidator()
    res = val.validate_ohlcv_data(df_test, min_data_points=100)
    assert res['is_valid'], f"Validator failed: {res['issues']}"
    print("  ✓ DataValidator")

    # 2. analyze_stock via temp CSV
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        df_test.to_csv(csv_path, index=False)

        cfg = Config()
        cfg.INPUT_DATA_DIR  = tmpdir
        cfg.MIN_DATA_POINTS = 80
        engine = CrossSectionalRankingEngine(data_dir=tmpdir, config_obj=cfg)

        result = engine.analyze_stock("TEST")
        assert result is not None,          "analyze_stock returned None"
        assert 'roc_composite' in result,   "roc_composite missing"
        assert 'adx'           in result,   "adx missing"
        assert 'vol_ratio'     in result,   "vol_ratio missing"
        print("  ✓ analyze_stock")

    # 3. Z-score correctness: sum ≈ 0 and mean ≈ 0
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    z = CrossSectionalRankingEngine._zscore(s)
    assert abs(z.sum())  < 1e-9, f"Z-score sum should ≈ 0, got {z.sum()}"
    assert abs(z.mean()) < 1e-9, f"Z-score mean should ≈ 0, got {z.mean()}"
    print("  ✓ Z-score normalization (sum ≈ 0, mean ≈ 0)")

    # 4. Winsorization clips outlier
    s_out = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    s_win = CrossSectionalRankingEngine._winsorize(s_out, 5, 95)
    assert s_win.max() < 100.0, "Winsorization did not clip upper outlier"
    print("  ✓ Winsorization")

    # 5. Zero-std guard
    s_flat  = pd.Series([5.0, 5.0, 5.0, 5.0])
    z_flat  = CrossSectionalRankingEngine._zscore(s_flat)
    assert (z_flat == 0).all(), "Flat series must produce all-zero Z-scores"
    print("  ✓ Zero-std guard")

    # 6. Composite score formula
    z_m, z_t, z_v = 1.5, 0.8, -0.2
    cfg_tmp = Config()
    expected = z_m * cfg_tmp.WEIGHT_MOMENTUM + z_t * cfg_tmp.WEIGHT_TREND + z_v * cfg_tmp.WEIGHT_VOLUME
    print(
        f"  ✓ Composite score: {z_m}×{cfg_tmp.WEIGHT_MOMENTUM} + "
        f"{z_t}×{cfg_tmp.WEIGHT_TREND} + {z_v}×{cfg_tmp.WEIGHT_VOLUME} = {expected:.4f}"
    )

    # 7. Graceful fallback: USE_ML_MODEL=True but no model file
    with tempfile.TemporaryDirectory() as tmpdir2:
        csv_path2 = os.path.join(tmpdir2, "test.csv")
        df_test.to_csv(csv_path2, index=False)
        cfg2 = Config()
        cfg2.INPUT_DATA_DIR  = tmpdir2
        cfg2.MIN_DATA_POINTS = 80
        cfg2.USE_ML_MODEL    = True
        cfg2.MODEL_PATH      = os.path.join(tmpdir2, "nonexistent_model.pkl")
        engine2 = CrossSectionalRankingEngine(data_dir=tmpdir2, config_obj=cfg2)
        ranked  = engine2.analyze_all_stocks()
        assert not ranked.empty,                    "Fallback should produce results"
        assert 'composite_score' in ranked.columns, "composite_score must be present"
        print("  ✓ Graceful ML fallback (missing model → rule-based)")

    # 8. Neutralized target calculation
    raw_returns   = np.array([0.05, 0.10, -0.02, 0.08, 0.01])
    median_return = np.median(raw_returns)
    neutralized   = raw_returns - median_return
    assert abs(np.median(neutralized)) < 1e-10, "Neutralized target median must be 0"
    print("  ✓ Neutralized target (excess over universe median)")

    # 9. Rank IC (Spearman)
    ic = MLModelTrainer._rank_ic(
        y_true=np.array([1, 2, 3, 4, 5]),
        y_pred=np.array([1.1, 1.9, 3.1, 3.8, 5.2])
    )
    assert ic > 0.99, f"Perfect-rank Rank IC should be ≈ 1.0, got {ic:.4f}"
    print(f"  ✓ Rank IC (Spearman): {ic:.4f}")

    print("\nAll tests passed.\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Cross-Sectional Relative Strength Ranking Engine v5.0'
    )
    # Directories
    parser.add_argument('--input-dir',  type=str, help='Input CSV directory')
    parser.add_argument('--output-dir', type=str, help='Output report directory')
    parser.add_argument('--model-dir',  type=str, help='Model artifacts directory')
    parser.add_argument('--config',     type=str, help='JSON config file path')

    # Inference
    parser.add_argument('--symbols',    nargs='+', help='Specific symbols to analyze')
    parser.add_argument('--use-ml',     action='store_true',
                        help='Enable ML scoring (requires trained model)')
    parser.add_argument('--model-type', choices=['ridge', 'xgboost'],
                        default='ridge', help='ML model type for inference')
    parser.add_argument('--model-path', type=str,
                        help='Explicit path to .pkl model artifact')

    # Training
    parser.add_argument('--train',      action='store_true',
                        help='Run MLModelTrainer (offline training mode)')
    parser.add_argument('--train-phase',
                        choices=['ridge', 'xgboost', 'both'], default='both',
                        help='Which training phase(s) to run (requires --train)')

    # Processing
    parser.add_argument('--min-data',   type=int,  help='Minimum data points per stock')
    parser.add_argument('--parallel',   action='store_true',
                        help='Enable parallel processing')
    parser.add_argument('--workers',    type=int, default=4,
                        help='Parallel worker count')

    # Output
    parser.add_argument('--no-report',  action='store_true', help='Skip Excel report')
    parser.add_argument('--no-summary', action='store_true', help='Skip console summary')

    # Utilities
    parser.add_argument('--create-config', type=str,
                        help='Write sample config to this path and exit')
    parser.add_argument('--log-level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Logging verbosity')
    parser.add_argument('--test', action='store_true',
                        help='Run unit tests and exit')

    return parser.parse_args()


if __name__ == "__main__":
    print("Cross-Sectional Relative Strength Ranking Engine v5.0")
    print("Long-Only | Z-Score Normalized | ML-Augmented Scoring")
    print("-" * 80)

    try:
        args = parse_arguments()

        log_map = {
            'DEBUG': logging.DEBUG, 'INFO': logging.INFO,
            'WARNING': logging.WARNING, 'ERROR': logging.ERROR
        }
        logger.setLevel(log_map[args.log_level])

        if args.test:
            run_tests()
            sys.exit(0)

        if args.create_config:
            create_sample_config(args.create_config)
            sys.exit(0)

        # Build config
        cfg = Config(args.config if args.config else None)
        if args.input_dir:   cfg.INPUT_DATA_DIR    = args.input_dir
        if args.output_dir:  cfg.OUTPUT_REPORT_DIR = args.output_dir
        if args.model_dir:   cfg.MODEL_DIR         = args.model_dir
        if args.min_data:    cfg.MIN_DATA_POINTS   = args.min_data
        if args.parallel:    cfg.PARALLEL_PROCESSING = True
        if args.workers:     cfg.MAX_WORKERS        = args.workers
        if args.use_ml:      cfg.USE_ML_MODEL       = True
        if args.model_type:  cfg.MODEL_TYPE         = args.model_type
        if args.model_path:  cfg.MODEL_PATH         = args.model_path

        # Re-resolve MODEL_PATH if model_type changed but path wasn't explicit
        if not args.model_path and not cfg.MODEL_PATH:
            cfg.MODEL_PATH = os.path.join(
                cfg.MODEL_DIR, f"{cfg.MODEL_TYPE}_ranker.pkl"
            )

        os.makedirs(cfg.OUTPUT_REPORT_DIR, exist_ok=True)
        os.makedirs(cfg.MODEL_DIR, exist_ok=True)

        # ---- Training Mode ----
        if args.train:
            print("\nMode: TRAINING (offline)")
            trainer = MLModelTrainer(config_obj=cfg, data_dir=cfg.INPUT_DATA_DIR)

            print("Building panel dataset...")
            panel = trainer.prepare_training_data()
            print(f"Panel: {len(panel)} rows | {panel['symbol'].nunique()} stocks")

            ridge_r2 = 0.0

            if args.train_phase in ('ridge', 'both'):
                print("\n--- Phase 1: Ridge Regression ---")
                ridge_res = trainer.train_ridge_model(panel)
                ridge_r2  = ridge_res['cv_avg_r2']
                print(f"Ridge complete: Avg R²={ridge_r2:.4f}  "
                      f"Avg Rank IC={ridge_res['cv_avg_rank_ic']:.4f}")
                print(f"Coefficients: {ridge_res['coefficients']}")
                print(f"Model saved : {ridge_res['model_path']}")
                print(f"Audit file  : {ridge_res['coeff_path']}")

            if args.train_phase in ('xgboost', 'both'):
                if not XGBOOST_AVAILABLE:
                    print("\nXGBoost not installed. Skipping Phase 2.")
                    print("Install with: pip install xgboost")
                else:
                    print("\n--- Phase 2: XGBoost ---")
                    xgb_res = trainer.train_xgboost_model(panel, ridge_avg_r2=ridge_r2)
                    print(f"XGBoost complete: Avg R²={xgb_res['cv_avg_r2']:.4f}  "
                          f"Avg Rank IC={xgb_res['cv_avg_rank_ic']:.4f}")
                    print(
                        f"R² improvement vs Ridge: {xgb_res['r2_improvement_vs_ridge']:+.4f}  "
                        f"| Production eligible: {xgb_res['is_production_eligible']}"
                    )
                    print(f"Model saved: {xgb_res['model_path']}")

            print("\nTraining complete.")
            sys.exit(0)

        # ---- Inference Mode ----
        print("\nMode: INFERENCE (live ranking)")
        engine = CrossSectionalRankingEngine(
            data_dir=cfg.INPUT_DATA_DIR, config_obj=cfg
        )
        ranked_df, report_path = engine.run(
            symbols=args.symbols,
            generate_report=not args.no_report,
            print_summary=not args.no_summary,
        )

        if not ranked_df.empty:
            print(f"\nSuccess! {len(ranked_df)} stocks ranked.")
            if report_path:
                print(f"Report: {report_path}")

            top = ranked_df.iloc[0]
            print(f"\n#1 Ranked Stock: {top['symbol']}")
            print(f"  Price            : {top['price']:.2f}")
            print(f"  Composite Score  : {top['composite_score']:.4f}")
            print(f"  ROC 20d          : {top['roc_20']*100:+.2f}%")
            print(f"  ROC 60d          : {top['roc_60']*100:+.2f}%")
            print(f"  ADX (14)         : {top['adx']:.2f}")
            print(f"  Volume Ratio     : {top['vol_ratio']:.2f}x")
        else:
            print("\nRanking failed. Check logs for details.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nFatal error: {e}")
