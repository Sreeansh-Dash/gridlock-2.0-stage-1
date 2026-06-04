import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("""# Traffic Demand Prediction - Complete ML Solution

This notebook presents a complete, optimized solution for predicting traffic demand.
The objective is to maximize the R² score using an ensemble of models (LightGBM, XGBoost, CatBoost).

We will follow these steps:
1. **Setup & Data Loading**
2. **Exploratory Data Analysis (EDA)**
3. **Feature Engineering**
4. **Model Training & Tuning (Optuna)**
5. **Ensemble & Blending**
6. **Submission Generation**"""))

# Setup
cells.append(nbf.v4.new_markdown_cell("""## 1. Setup & Data Loading
Import necessary libraries and load the datasets."""))

cells.append(nbf.v4.new_code_cell("""import warnings
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

def load_data(data_dir='./dataset'):
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test  = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    sample_sub = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    return train, test, sample_sub

train, test, sample_sub = load_data()
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")"""))

# EDA
cells.append(nbf.v4.new_markdown_cell("""## 2. Exploratory Data Analysis (EDA)
Let's inspect the data shapes, missing values, variable distributions, and the target variable `demand`."""))

cells.append(nbf.v4.new_code_cell("""def run_eda(train, test):
    print("--- Train dtypes ---")
    print(train.dtypes)
    print("\\n--- Null counts (Train) ---")
    print(train.isnull().sum())
    
    print("\\n--- Demand Target Stats ---")
    print(train['demand'].describe())
    
    cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
    for col in cat_cols:
        print(f"\\n--- {col} Value Counts ---")
        print(train[col].value_counts(dropna=False))

run_eda(train, test)"""))

# Feature Engineering
cells.append(nbf.v4.new_markdown_cell("""## 3. Feature Engineering
We will engineer several types of features:
- **Geohash Decoding**: Extract latitude and longitude from the geohash strings.
- **Time Features**: Parse hour/minute from the `timestamp`, compute rush hour, and cyclical encoding (sin/cos).
- **Day Features**: Extract cyclical day information.
- **Interactions**: Create interaction terms like `Temperature * RoadType` and `Weather * LargeVehicles`.
- **Target Encoding / Aggregates**: Compute geohash-level and hour-level statistics from the train set."""))

cells.append(nbf.v4.new_code_cell("""def engineer_features(train, test):
    y_train = train['demand'].values
    train_idx = train['Index'].values
    test_idx = test['Index'].values

    df = pd.concat([train.drop('demand', axis=1), test], axis=0, ignore_index=True)
    n_train = len(train)

    # 3a. Geohash decoding
    df['lat'] = df['geohash'].apply(lambda g: pgh.decode(g)[0])
    df['lon'] = df['geohash'].apply(lambda g: pgh.decode(g)[1])
    le_geo = LabelEncoder()
    df['geohash_enc'] = le_geo.fit_transform(df['geohash'].astype(str))

    # 3b. Timestamp features
    ts_split = df['timestamp'].astype(str).str.split(':', expand=True)
    df['hour']   = ts_split[0].astype(int)
    df['minute'] = ts_split[1].astype(int)
    df['time_minutes'] = df['hour'] * 60 + df['minute']
    df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 20))).astype(int)
    
    bins_tod = [0, 6, 12, 17, 21, 24]
    labels_tod = [0, 1, 2, 3, 4]
    df['time_of_day_bin'] = pd.cut(df['hour'], bins=bins_tod, labels=labels_tod, right=False, include_lowest=True).astype(float).fillna(0).astype(int)
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

    # 3c. Day features
    df['day'] = df['day'].astype(int)
    df['day_mod7'] = df['day'] % 7
    df['is_weekend'] = (df['day_mod7'].isin([5, 6])).astype(int)
    df['day_sin'] = np.sin(2 * np.pi * df['day_mod7'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_mod7'] / 7)

    # 3d. Encode categoricals & Missing values
    cat_cols = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
        df[col + '_enc'] = LabelEncoder().fit_transform(df[col].astype(str))
        
    df['NumberofLanes'] = pd.to_numeric(df['NumberofLanes'], errors='coerce')
    df['Temperature'] = df['Temperature'].fillna(df['Temperature'].median())
    df['NumberofLanes'] = df['NumberofLanes'].fillna(df['NumberofLanes'].median())

    # 3f. Interaction features
    df['hour_road'] = df['hour'] * df['RoadType_enc']
    temp_bins = [-50, 5, 15, 25, 50]
    df['temp_bin'] = pd.cut(df['Temperature'], bins=temp_bins, labels=[0, 1, 2, 3], include_lowest=True).astype(float).fillna(1).astype(int)
    df['weather_largeveh'] = df['Weather_enc'] * df['LargeVehicles_enc']
    df['temp_road'] = df['Temperature'] * df['RoadType_enc']
    
    df['geohash_prefix4_enc'] = LabelEncoder().fit_transform(df['geohash'].str[:4].astype(str))
    df['geohash_prefix5_enc'] = LabelEncoder().fit_transform(df['geohash'].str[:5].astype(str))
    df['lanes_road'] = df['NumberofLanes'] * df['RoadType_enc']
    df['landmarks_road'] = df['Landmarks_enc'] * df['RoadType_enc']

    # 3g. Aggregate features (train only)
    train_tmp = df.iloc[:n_train].copy()
    train_tmp['demand'] = y_train

    geo_stats = train_tmp.groupby('geohash')['demand'].agg(['mean', 'std', 'median', 'count']).reset_index()
    geo_stats.columns = ['geohash', 'geo_demand_mean', 'geo_demand_std', 'geo_demand_median', 'geo_count']
    df = df.merge(geo_stats, on='geohash', how='left')
    
    hour_stats = train_tmp.groupby('hour')['demand'].agg(['mean', 'std']).reset_index()
    hour_stats.columns = ['hour', 'hour_demand_mean', 'hour_demand_std']
    df = df.merge(hour_stats, on='hour', how='left')
    
    road_stats = train_tmp.groupby('RoadType_enc')['demand'].agg(['mean', 'std']).reset_index()
    road_stats.columns = ['RoadType_enc', 'road_demand_mean', 'road_demand_std']
    df = df.merge(road_stats, on='RoadType_enc', how='left')
    
    geo_hour_stats = train_tmp.groupby(['geohash', 'hour'])['demand'].agg(['mean']).reset_index()
    geo_hour_stats.columns = ['geohash', 'hour', 'geo_hour_demand_mean']
    df = df.merge(geo_hour_stats, on=['geohash', 'hour'], how='left')
    
    # Fill NAs in aggregates
    df = df.fillna(0)

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

    return X_train, y_train, X_test, feature_cols, test_idx

X_train, y_train, X_test, feature_names, test_idx = engineer_features(train, test)
print(f"X_train shape: {X_train.shape}")"""))

# LightGBM Tuning
cells.append(nbf.v4.new_markdown_cell("""## 4. Model Training & Tuning
We use a 5-fold Cross-Validation strategy. We will:
1. Tune LightGBM using Optuna for 50 trials.
2. Train XGBoost.
3. Train CatBoost.
4. Keep the Out-of-Fold (OOF) predictions for the ensemble."""))

cells.append(nbf.v4.new_code_cell("""N_FOLDS = 5
SEED = 42

def optuna_tune_lgbm(X, y, n_trials=50, n_folds=N_FOLDS):
    print(f"Running Optuna for {n_trials} trials...")
    def objective(trial):
        params = {
            'objective': 'regression', 'metric': 'rmse',
            'n_estimators': 2000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'random_state': SEED, 'n_jobs': -1, 'verbosity': -1,
        }
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        oof_preds = np.zeros(len(X))
        for train_idx, val_idx in kf.split(X):
            model = lgb.LGBMRegressor(**params)
            model.fit(X[train_idx], y[train_idx], eval_set=[(X[val_idx], y[val_idx])], callbacks=[lgb.early_stopping(50, verbose=False)])
            oof_preds[val_idx] = model.predict(X[val_idx])
        return r2_score(y, oof_preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    print(f"Best Trial R²: {study.best_trial.value:.6f}")
    best_params = {'objective': 'regression', 'metric': 'rmse', 'n_estimators': 2000, 'random_state': SEED, 'n_jobs': -1, 'verbosity': -1}
    best_params.update(study.best_trial.params)
    return best_params

# We load the best params that were already discovered in our tuning script
best_lgb_params = {
  "objective": "regression", "metric": "rmse", "n_estimators": 2000, "random_state": 42, "n_jobs": -1, "verbosity": -1,
  "learning_rate": 0.03541, "num_leaves": 42, "max_depth": 11, "min_child_samples": 10,
  "subsample": 0.882, "colsample_bytree": 0.983, "reg_alpha": 0.0136, "reg_lambda": 3.35
}
print(f"Using Tuned LightGBM Params: {best_lgb_params}")"""))

# Training the Ensemble
cells.append(nbf.v4.new_markdown_cell("""### Train Base Models (LGBM, XGBoost, CatBoost)"""))

cells.append(nbf.v4.new_code_cell("""def train_model_cv(X, y, X_test, model_fn, n_folds=N_FOLDS):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        model = model_fn()
        if isinstance(model, xgb.XGBRegressor):
            model.fit(X[train_idx], y[train_idx], eval_set=[(X[val_idx], y[val_idx])], verbose=False)
        elif isinstance(model, lgb.LGBMRegressor):
            model.fit(X[train_idx], y[train_idx], eval_set=[(X[val_idx], y[val_idx])], callbacks=[lgb.early_stopping(100, verbose=False)])
        else:
            model.fit(X[train_idx], y[train_idx], eval_set=[(X[val_idx], y[val_idx])])
            
        oof_preds[val_idx] = model.predict(X[val_idx])
        test_preds += model.predict(X_test) / n_folds
    
    oof_r2 = r2_score(y, oof_preds)
    print(f"OOF R²: {oof_r2:.6f}")
    return oof_preds, test_preds, oof_r2

# 1. LightGBM
print("Training LightGBM...")
oof_lgb, test_lgb, r2_lgb = train_model_cv(X_train, y_train, X_test, lambda: lgb.LGBMRegressor(**best_lgb_params))

# 2. XGBoost
print("\\nTraining XGBoost...")
xgb_params = {'n_estimators': 2000, 'learning_rate': 0.03, 'max_depth': 7, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': SEED, 'n_jobs': -1, 'tree_method': 'hist', 'early_stopping_rounds': 100, 'verbosity': 0}
oof_xgb, test_xgb, r2_xgb = train_model_cv(X_train, y_train, X_test, lambda: xgb.XGBRegressor(**xgb_params))

# 3. CatBoost
print("\\nTraining CatBoost...")
oof_cat, test_cat, r2_cat = train_model_cv(X_train, y_train, X_test, lambda: CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=8, l2_leaf_reg=3, random_seed=SEED, verbose=0, early_stopping_rounds=100))
"""))

# Ensemble Blending
cells.append(nbf.v4.new_markdown_cell("""## 5. Ensemble Blending
We will blend the Out-of-Fold predictions using an optimized set of weights to maximize the final R² Score."""))

cells.append(nbf.v4.new_code_cell("""# Found weights from earlier grid search
w_lgb, w_xgb, w_cat = 0.45, 0.10, 0.45

blend_oof = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cat * oof_cat
final_r2 = r2_score(y_train, blend_oof)
print(f"Ensemble OOF R² = {final_r2:.6f} -> Score: {max(0, 100*final_r2):.2f}")

# Blend test predictions
final_test_preds = w_lgb * test_lgb + w_xgb * test_xgb + w_cat * test_cat

# Post-processing: Clip predictions to target range [0, 1]
final_test_preds = np.clip(final_test_preds, y_train.min(), y_train.max())
"""))

# Submission
cells.append(nbf.v4.new_markdown_cell("""## 6. Generate Submission
Create the final `submission.csv` containing the `Index` and predicted `demand`."""))

cells.append(nbf.v4.new_code_cell("""submission = pd.DataFrame({
    'Index': test_idx,
    'demand': final_test_preds
})

submission.to_csv('submission.csv', index=False)
print(f"Submission saved with shape: {submission.shape}")
submission.head()"""))

nb.cells.extend(cells)

with open('traffic_demand_solution.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook saved successfully!")
