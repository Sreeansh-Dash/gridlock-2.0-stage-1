"""
Traffic Demand Prediction - Complete ML Solution
=================================================
Predicts traffic demand using an ensemble of LightGBM, XGBoost, and CatBoost.
Evaluation metric: R² score (maximize).
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pygeohash as pgh
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import os
import json
import time

# ============================================================
# 1. LOAD DATA
# ============================================================
def load_data(data_dir='./dataset'):
    """Load train, test and sample submission CSVs."""
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test  = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    sample_sub = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    return train, test, sample_sub


# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================
def run_eda(train, test):
    """Print shapes, dtypes, null counts, basic stats, and value counts."""
    print("=" * 70)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # --- Shape ---
    print(f"\nTrain shape: {train.shape}")
    print(f"Test  shape: {test.shape}")

    # --- Dtypes ---
    print(f"\n--- Train dtypes ---\n{train.dtypes}")
    print(f"\n--- Test  dtypes ---\n{test.dtypes}")

    # --- Null counts ---
    print(f"\n--- Train null counts ---\n{train.isnull().sum()}")
    print(f"\n--- Test  null counts ---\n{test.isnull().sum()}")

    # --- Basic stats (numeric) ---
    print(f"\n--- Train describe ---\n{train.describe()}")

    # --- Demand distribution ---
    print(f"\n--- Demand stats ---")
    print(f"  min   = {train['demand'].min():.6f}")
    print(f"  max   = {train['demand'].max():.6f}")
    print(f"  mean  = {train['demand'].mean():.6f}")
    print(f"  median= {train['demand'].median():.6f}")
    print(f"  std   = {train['demand'].std():.6f}")
    q = train['demand'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(f"  quantiles:\n{q}")

    # --- Categorical value counts ---
    cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
    for col in cat_cols:
        print(f"\n--- {col} value counts (train) ---\n{train[col].value_counts(dropna=False)}")
        print(f"--- {col} value counts (test) ---\n{test[col].value_counts(dropna=False)}")

    # --- Duplicates ---
    dup_train = train.duplicated().sum()
    dup_test  = test.duplicated().sum()
    print(f"\nTrain duplicates: {dup_train}")
    print(f"Test  duplicates: {dup_test}")

    # --- Unique geohashes ---
    print(f"\nUnique geohashes in train: {train['geohash'].nunique()}")
    print(f"Unique geohashes in test:  {test['geohash'].nunique()}")

    # --- Day and timestamp ---
    print(f"\nTrain day range: {train['day'].min()} - {train['day'].max()}")
    print(f"Test  day range: {test['day'].min()} - {test['day'].max()}")
    print(f"\nTrain timestamp samples: {train['timestamp'].unique()[:20]}")
    print(f"Test  timestamp samples: {test['timestamp'].unique()[:20]}")


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
def engineer_features(train, test):
    """
    Apply all feature engineering steps to both train and test.
    Returns X_train, y_train, X_test, feature_names.
    """
    # Save target and Index
    y_train = train['demand'].values
    train_idx = train['Index'].values
    test_idx = test['Index'].values

    # Combine for consistent encoding
    df = pd.concat([train.drop('demand', axis=1), test], axis=0, ignore_index=True)
    n_train = len(train)

    # -----------------------------------------------------------
    # 3a. Geohash decoding → lat, lon
    # -----------------------------------------------------------
    print("\n[Feature Engineering] Decoding geohashes...")
    df['lat'] = df['geohash'].apply(lambda g: pgh.decode(g)[0])
    df['lon'] = df['geohash'].apply(lambda g: pgh.decode(g)[1])

    # Label-encode geohash
    le_geo = LabelEncoder()
    df['geohash_enc'] = le_geo.fit_transform(df['geohash'].astype(str))

    # -----------------------------------------------------------
    # 3b. Timestamp features
    # -----------------------------------------------------------
    print("[Feature Engineering] Parsing timestamps...")
    # timestamp is in H:M format (e.g. "0:0", "2:15")
    ts_split = df['timestamp'].astype(str).str.split(':', expand=True)
    df['hour']   = ts_split[0].astype(int)
    df['minute'] = ts_split[1].astype(int)

    # Time-of-day as total minutes
    df['time_minutes'] = df['hour'] * 60 + df['minute']

    # Rush hour (7-9am, 5-8pm)
    df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 20))).astype(int)

    # Time-of-day bins
    bins_tod = [0, 6, 12, 17, 21, 24]
    labels_tod = [0, 1, 2, 3, 4]  # night, morning, afternoon, evening, late_night
    df['time_of_day_bin'] = pd.cut(df['hour'], bins=bins_tod, labels=labels_tod, right=False, include_lowest=True).astype(float).fillna(0).astype(int)

    # Cyclical encoding: hour and minute
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

    # -----------------------------------------------------------
    # 3c. Day features
    # -----------------------------------------------------------
    print("[Feature Engineering] Engineering day features...")
    # Day is a numeric integer
    df['day'] = df['day'].astype(int)
    df['day_mod7'] = df['day'] % 7  # pseudo day_of_week
    df['is_weekend'] = (df['day_mod7'].isin([5, 6])).astype(int)

    # Cyclical encoding of day_mod7
    df['day_sin'] = np.sin(2 * np.pi * df['day_mod7'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_mod7'] / 7)

    # -----------------------------------------------------------
    # 3d. Encode categoricals
    # -----------------------------------------------------------
    print("[Feature Engineering] Encoding categoricals...")
    cat_cols = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']

    label_encoders = {}
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # NumberofLanes as numeric
    df['NumberofLanes'] = pd.to_numeric(df['NumberofLanes'], errors='coerce')

    # -----------------------------------------------------------
    # 3e. Handle missing values
    # -----------------------------------------------------------
    print("[Feature Engineering] Handling missing values...")
    df['Temperature'] = df['Temperature'].fillna(df['Temperature'].median())
    df['NumberofLanes'] = df['NumberofLanes'].fillna(df['NumberofLanes'].median())

    # -----------------------------------------------------------
    # 3f. Interaction features
    # -----------------------------------------------------------
    print("[Feature Engineering] Creating interaction features...")

    # hour × RoadType
    df['hour_road'] = df['hour'] * df['RoadType_enc']

    # Temperature bins
    temp_bins = [-50, 5, 15, 25, 50]
    temp_labels = [0, 1, 2, 3]  # cold, cool, mild, hot
    df['temp_bin'] = pd.cut(df['Temperature'], bins=temp_bins, labels=temp_labels, include_lowest=True).astype(float).fillna(1).astype(int)

    # Weather × LargeVehicles
    df['weather_largeveh'] = df['Weather_enc'] * df['LargeVehicles_enc']

    # Temperature × RoadType
    df['temp_road'] = df['Temperature'] * df['RoadType_enc']

    # Geohash prefix features (spatial hierarchy)
    df['geohash_prefix4'] = df['geohash'].str[:4]
    le_geo4 = LabelEncoder()
    df['geohash_prefix4_enc'] = le_geo4.fit_transform(df['geohash_prefix4'].astype(str))

    df['geohash_prefix5'] = df['geohash'].str[:5]
    le_geo5 = LabelEncoder()
    df['geohash_prefix5_enc'] = le_geo5.fit_transform(df['geohash_prefix5'].astype(str))

    # NumberofLanes × RoadType
    df['lanes_road'] = df['NumberofLanes'] * df['RoadType_enc']

    # Landmarks × RoadType
    df['landmarks_road'] = df['Landmarks_enc'] * df['RoadType_enc']

    # -----------------------------------------------------------
    # 3g. Aggregate features (geohash-level statistics from train)
    # -----------------------------------------------------------
    print("[Feature Engineering] Creating aggregate features...")

    # We compute aggregates on train only, then merge back
    train_tmp = df.iloc[:n_train].copy()
    train_tmp['demand'] = y_train

    # Geohash-level demand stats
    geo_stats = train_tmp.groupby('geohash')['demand'].agg(['mean', 'std', 'median', 'count']).reset_index()
    geo_stats.columns = ['geohash', 'geo_demand_mean', 'geo_demand_std', 'geo_demand_median', 'geo_count']
    geo_stats['geo_demand_std'] = geo_stats['geo_demand_std'].fillna(0)
    df = df.merge(geo_stats, on='geohash', how='left')
    df['geo_demand_mean'] = df['geo_demand_mean'].fillna(y_train.mean())
    df['geo_demand_std'] = df['geo_demand_std'].fillna(0)
    df['geo_demand_median'] = df['geo_demand_median'].fillna(y_train.mean())
    df['geo_count'] = df['geo_count'].fillna(0)

    # Hour-level demand stats
    hour_stats = train_tmp.groupby('hour')['demand'].agg(['mean', 'std']).reset_index()
    hour_stats.columns = ['hour', 'hour_demand_mean', 'hour_demand_std']
    df = df.merge(hour_stats, on='hour', how='left')
    df['hour_demand_mean'] = df['hour_demand_mean'].fillna(y_train.mean())
    df['hour_demand_std'] = df['hour_demand_std'].fillna(0)

    # RoadType-level demand stats
    road_stats = train_tmp.groupby('RoadType_enc')['demand'].agg(['mean', 'std']).reset_index()
    road_stats.columns = ['RoadType_enc', 'road_demand_mean', 'road_demand_std']
    df = df.merge(road_stats, on='RoadType_enc', how='left')
    df['road_demand_mean'] = df['road_demand_mean'].fillna(y_train.mean())
    df['road_demand_std'] = df['road_demand_std'].fillna(0)

    # Geohash × hour demand stats
    geo_hour_stats = train_tmp.groupby(['geohash', 'hour'])['demand'].agg(['mean']).reset_index()
    geo_hour_stats.columns = ['geohash', 'hour', 'geo_hour_demand_mean']
    df = df.merge(geo_hour_stats, on=['geohash', 'hour'], how='left')
    df['geo_hour_demand_mean'] = df['geo_hour_demand_mean'].fillna(df['geo_demand_mean'])

    # -----------------------------------------------------------
    # Select features
    # -----------------------------------------------------------
    feature_cols = [
        'lat', 'lon', 'geohash_enc',
        'hour', 'minute', 'time_minutes', 'is_rush_hour', 'time_of_day_bin',
        'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
        'day', 'day_mod7', 'is_weekend', 'day_sin', 'day_cos',
        'RoadType_enc', 'Weather_enc', 'LargeVehicles_enc', 'Landmarks_enc',
        'NumberofLanes', 'Temperature', 'temp_bin',
        'hour_road', 'weather_largeveh', 'temp_road',
        'geohash_prefix4_enc', 'geohash_prefix5_enc',
        'lanes_road', 'landmarks_road',
        'geo_demand_mean', 'geo_demand_std', 'geo_demand_median', 'geo_count',
        'hour_demand_mean', 'hour_demand_std',
        'road_demand_mean', 'road_demand_std',
        'geo_hour_demand_mean',
    ]

    X_train = df.iloc[:n_train][feature_cols].values
    X_test  = df.iloc[n_train:][feature_cols].values

    print(f"\nFeature set: {len(feature_cols)} features")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test  shape: {X_test.shape}")

    return X_train, y_train, X_test, feature_cols, test_idx


# ============================================================
# 4. MODEL TRAINING with OOF predictions
# ============================================================
N_FOLDS = 5
SEED = 42


def train_lightgbm(X, y, X_test, feature_names, n_folds=N_FOLDS, params=None):
    """Train LightGBM with K-Fold, return OOF and test predictions."""
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'n_estimators': 2000,
            'learning_rate': 0.03,
            'num_leaves': 127,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': SEED,
            'n_jobs': -1,
            'verbosity': -1,
        }

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    print(f"\n{'='*60}")
    print(f"Training LightGBM ({n_folds}-fold)")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n  Fold {fold+1}/{n_folds}")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / n_folds

        fold_r2 = r2_score(y_val, oof_preds[val_idx])
        print(f"    Fold {fold+1} R² = {fold_r2:.6f}  |  best_iter = {model.best_iteration_}")

    oof_r2 = r2_score(y, oof_preds)
    print(f"\n  *** LightGBM OOF R² = {oof_r2:.6f} ***")

    # Feature importance
    imp = model.feature_importances_
    fi = pd.DataFrame({'feature': feature_names, 'importance': imp}).sort_values('importance', ascending=False)
    print(f"\n  Top-15 features (last fold):")
    print(fi.head(15).to_string(index=False))

    return oof_preds, test_preds, oof_r2


def train_xgboost(X, y, X_test, feature_names, n_folds=N_FOLDS):
    """Train XGBoost with K-Fold, return OOF and test predictions."""
    params = {
        'n_estimators': 2000,
        'learning_rate': 0.03,
        'max_depth': 7,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': SEED,
        'n_jobs': -1,
        'tree_method': 'hist',
        'verbosity': 0,
        'early_stopping_rounds': 100,
    }

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    print(f"\n{'='*60}")
    print(f"Training XGBoost ({n_folds}-fold)")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n  Fold {fold+1}/{n_folds}")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / n_folds

        fold_r2 = r2_score(y_val, oof_preds[val_idx])
        best_iter = getattr(model, 'best_iteration', params['n_estimators'])
        print(f"    Fold {fold+1} R² = {fold_r2:.6f}  |  best_iter = {best_iter}")

    oof_r2 = r2_score(y, oof_preds)
    print(f"\n  *** XGBoost OOF R² = {oof_r2:.6f} ***")

    return oof_preds, test_preds, oof_r2


def train_catboost(X, y, X_test, feature_names, n_folds=N_FOLDS):
    """Train CatBoost with K-Fold, return OOF and test predictions."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    print(f"\n{'='*60}")
    print(f"Training CatBoost ({n_folds}-fold)")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n  Fold {fold+1}/{n_folds}")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3,
            random_seed=SEED,
            verbose=0,
            early_stopping_rounds=100,
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / n_folds

        fold_r2 = r2_score(y_val, oof_preds[val_idx])
        print(f"    Fold {fold+1} R² = {fold_r2:.6f}  |  best_iter = {model.best_iteration_}")

    oof_r2 = r2_score(y, oof_preds)
    print(f"\n  *** CatBoost OOF R² = {oof_r2:.6f} ***")

    return oof_preds, test_preds, oof_r2


# ============================================================
# 4d. Optuna tuning for LightGBM
# ============================================================
def optuna_tune_lgbm(X, y, n_trials=50, n_folds=5):
    """Run Optuna to find best LightGBM hyperparameters."""
    print(f"\n{'='*60}")
    print(f"Optuna Tuning LightGBM ({n_trials} trials)")
    print(f"{'='*60}")

    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'n_estimators': 2000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'random_state': SEED,
            'n_jobs': -1,
            'verbosity': -1,
        }

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        oof_preds = np.zeros(len(X))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            oof_preds[val_idx] = model.predict(X_val)

        return r2_score(y, oof_preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Best trial R²: {study.best_trial.value:.6f}")
    print(f"  Best params: {json.dumps(study.best_trial.params, indent=2)}")

    # Build the best params dict
    best = study.best_trial.params
    best_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 2000,
        'random_state': SEED,
        'n_jobs': -1,
        'verbosity': -1,
    }
    best_params.update(best)
    return best_params


# ============================================================
# 5. ENSEMBLE + POST-PROCESSING
# ============================================================
def ensemble_predictions(oof_lgb, oof_xgb, oof_cat, test_lgb, test_xgb, test_cat, y):
    """Blend OOF and test predictions with optimized weights."""
    # Default weights
    w_lgb, w_xgb, w_cat = 0.5, 0.3, 0.2

    # Try to find better weights via grid search
    print(f"\n{'='*60}")
    print("ENSEMBLE WEIGHT OPTIMIZATION")
    print(f"{'='*60}")

    best_r2 = -999
    best_w = (w_lgb, w_xgb, w_cat)

    for w1 in np.arange(0.3, 0.8, 0.05):
        for w2 in np.arange(0.1, 0.5, 0.05):
            w3 = 1.0 - w1 - w2
            if w3 < 0.05:
                continue
            blend = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
            r2 = r2_score(y, blend)
            if r2 > best_r2:
                best_r2 = r2
                best_w = (round(w1, 2), round(w2, 2), round(w3, 2))

    w_lgb, w_xgb, w_cat = best_w
    print(f"  Best weights: LGB={w_lgb}, XGB={w_xgb}, CAT={w_cat}")
    print(f"  Ensemble OOF R² = {best_r2:.6f}")

    # Blend test preds
    final_test = w_lgb * test_lgb + w_xgb * test_xgb + w_cat * test_cat

    return final_test, best_r2


def postprocess(predictions, y_train):
    """Clip predictions to training target range."""
    lo = y_train.min()
    hi = y_train.max()
    print(f"\n[Post-processing] Clipping to [{lo:.6f}, {hi:.6f}]")
    return np.clip(predictions, lo, hi)


# ============================================================
# 6. GENERATE SUBMISSION
# ============================================================
def generate_submission(test_idx, predictions, out_path='submission.csv'):
    """Save submission CSV."""
    sub = pd.DataFrame({'Index': test_idx, 'demand': predictions})
    sub.to_csv(out_path, index=False)
    print(f"\nSubmission saved to {out_path}")
    print(f"  Shape: {sub.shape}")
    print(f"  Head:\n{sub.head(10)}")
    return sub


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    start_time = time.time()

    # 1. Load
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)
    train, test, sample_sub = load_data()

    # 2. EDA
    run_eda(train, test)

    # 3. Feature Engineering
    print("\n" + "=" * 70)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 70)
    X_train, y_train, X_test, feature_names, test_idx = engineer_features(train, test)

    # 4a. Train LightGBM (default)
    print("\n" + "=" * 70)
    print("STEP 4: MODEL TRAINING")
    print("=" * 70)
    oof_lgb, test_lgb, r2_lgb = train_lightgbm(X_train, y_train, X_test, feature_names)

    # 4b. Train XGBoost
    oof_xgb, test_xgb, r2_xgb = train_xgboost(X_train, y_train, X_test, feature_names)

    # 4c. Train CatBoost
    oof_cat, test_cat, r2_cat = train_catboost(X_train, y_train, X_test, feature_names)

    # 4d. Optuna tuning for LightGBM
    best_lgb_params = optuna_tune_lgbm(X_train, y_train, n_trials=50, n_folds=5)

    # Re-train LightGBM with tuned params
    oof_lgb_tuned, test_lgb_tuned, r2_lgb_tuned = train_lightgbm(
        X_train, y_train, X_test, feature_names, params=best_lgb_params
    )

    # Use tuned LightGBM if it's better
    if r2_lgb_tuned > r2_lgb:
        print(f"\n  Tuned LightGBM is better: {r2_lgb_tuned:.6f} > {r2_lgb:.6f}")
        oof_lgb, test_lgb, r2_lgb = oof_lgb_tuned, test_lgb_tuned, r2_lgb_tuned
    else:
        print(f"\n  Default LightGBM is better: {r2_lgb:.6f} >= {r2_lgb_tuned:.6f}")

    # 5. Ensemble
    print("\n" + "=" * 70)
    print("STEP 5: ENSEMBLE")
    print("=" * 70)
    final_test, ens_r2 = ensemble_predictions(
        oof_lgb, oof_xgb, oof_cat,
        test_lgb, test_xgb, test_cat,
        y_train
    )

    # 6. Post-processing
    final_test = postprocess(final_test, y_train)

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  LightGBM OOF R²  = {r2_lgb:.6f}  ->  Score = {max(0, 100*r2_lgb):.2f}")
    print(f"  XGBoost  OOF R²  = {r2_xgb:.6f}  ->  Score = {max(0, 100*r2_xgb):.2f}")
    print(f"  CatBoost OOF R²  = {r2_cat:.6f}  ->  Score = {max(0, 100*r2_cat):.2f}")
    print(f"  Ensemble OOF R²  = {ens_r2:.6f}  ->  Score = {max(0, 100*ens_r2):.2f}")

    # 7. Generate submission
    print("\n" + "=" * 70)
    print("STEP 7: SUBMISSION")
    print("=" * 70)
    generate_submission(test_idx, final_test, out_path='submission.csv')

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    print("Done!")


if __name__ == '__main__':
    main()
