import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Traffic Demand Prediction - V3 Pipeline (100% Retraining + OOF Target Encoding)

This notebook contains the fully optimized ML pipeline:
- **Leakage-Free OOF Features**: Group-level demand aggregations are computed using 5-Fold Out-Of-Fold for the training set, eliminating target leakage.
- **Validation**: Strict Time-Based Holdout (last 20%) is used only for accurate local evaluation.
- **Categoricals**: `geohash_enc` and `geohash_prefix` are restored to allow trees to memorize location-specific baselines.
- **Complexity Tuning**: Models use deeper trees and tuned learning rates closer to the Optuna baseline.
- **100% Retraining**: After validation, models are fully retrained on the entire dataset to maximize leaderboard score!
"""))

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
import os
import time

SEED = 42

def load_data(data_dir='./dataset'):
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test  = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    return train, test

train, test = load_data()
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
"""))

cells.append(nbf.v4.new_markdown_cell("""## OOF Aggregations Helper"""))

cells.append(nbf.v4.new_code_cell("""def get_oof_aggregations(train_df, test_df, groupby_cols, target_col='demand', stat_funcs=['mean', 'std', 'median', 'count'], n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    prefix = "_".join(groupby_cols) if isinstance(groupby_cols, list) else groupby_cols
    out_cols = [f"{prefix}_{target_col}_{func}" for func in stat_funcs]
    for col in out_cols:
        train_df[col] = np.nan
        
    for tr_idx, val_idx in kf.split(train_df):
        tr_fold = train_df.iloc[tr_idx]
        val_fold = train_df.iloc[val_idx]
        agg = tr_fold.groupby(groupby_cols)[target_col].agg(stat_funcs)
        agg.columns = out_cols
        val_fold_joined = val_fold.drop(columns=out_cols).join(agg, on=groupby_cols)
        for col in out_cols:
            train_df.loc[val_idx, col] = val_fold_joined[col].values
            
    agg_full = train_df.groupby(groupby_cols)[target_col].agg(stat_funcs)
    agg_full.columns = out_cols
    test_df = test_df.join(agg_full, on=groupby_cols)
    
    global_stats = train_df[target_col].agg(stat_funcs)
    for col, func in zip(out_cols, stat_funcs):
        fill_val = global_stats[func] if func in ['mean', 'median'] else 0
        train_df[col] = train_df[col].fillna(fill_val)
        test_df[col] = test_df[col].fillna(fill_val)
        
    return train_df, test_df
"""))

cells.append(nbf.v4.new_markdown_cell("""## Feature Engineering"""))

cells.append(nbf.v4.new_code_cell("""def engineer_features(train, test):
    print("Engineering features...")
    train['time_minutes_temp'] = train['timestamp'].astype(str).str.split(':', expand=True)[0].astype(int) * 60 + train['timestamp'].astype(str).str.split(':', expand=True)[1].astype(int)
    train = train.sort_values(['day', 'time_minutes_temp']).reset_index(drop=True)
    train.drop('time_minutes_temp', axis=1, inplace=True)
    
    y_train = train['demand'].values
    test_idx = test['Index'].values

    n_train = len(train)
    df = pd.concat([train.drop('demand', axis=1), test], axis=0, ignore_index=True)

    df['lat'] = df['geohash'].apply(lambda g: pgh.decode(g)[0])
    df['lon'] = df['geohash'].apply(lambda g: pgh.decode(g)[1])
    df['geohash_enc'] = LabelEncoder().fit_transform(df['geohash'].astype(str))
    
    ts_split = df['timestamp'].astype(str).str.split(':', expand=True)
    df['hour']   = ts_split[0].astype(int)
    df['minute'] = ts_split[1].astype(int)
    df['time_minutes'] = df['hour'] * 60 + df['minute']
    df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 20))).astype(int)
    
    bins_tod = [0, 6, 12, 17, 21, 24]
    df['time_of_day_bin'] = pd.cut(df['hour'], bins=bins_tod, labels=[0, 1, 2, 3, 4], right=False, include_lowest=True).astype(float).fillna(0).astype(int)
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

    df['day'] = df['day'].astype(int)
    df['day_mod7'] = df['day'] % 7
    df['is_weekend'] = (df['day_mod7'].isin([5, 6])).astype(int)
    df['day_sin'] = np.sin(2 * np.pi * df['day_mod7'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_mod7'] / 7)

    cat_cols = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
        df[col + '_enc'] = LabelEncoder().fit_transform(df[col].astype(str))

    df['NumberofLanes'] = pd.to_numeric(df['NumberofLanes'], errors='coerce')
    df['Temperature'] = df['Temperature'].fillna(df['Temperature'].median())
    df['NumberofLanes'] = df['NumberofLanes'].fillna(df['NumberofLanes'].median())

    df['hour_road'] = df['hour'] * df['RoadType_enc']
    df['temp_bin'] = pd.cut(df['Temperature'], bins=[-50, 5, 15, 25, 50], labels=[0, 1, 2, 3], include_lowest=True).astype(float).fillna(1).astype(int)
    df['weather_largeveh'] = df['Weather_enc'] * df['LargeVehicles_enc']
    df['temp_road'] = df['Temperature'] * df['RoadType_enc']
    
    df['geohash_prefix4_enc'] = LabelEncoder().fit_transform(df['geohash'].str[:4].astype(str))
    df['geohash_prefix5_enc'] = LabelEncoder().fit_transform(df['geohash'].str[:5].astype(str))
    
    df['lanes_road'] = df['NumberofLanes'] * df['RoadType_enc']
    df['landmarks_road'] = df['Landmarks_enc'] * df['RoadType_enc']

    train_feat = df.iloc[:n_train].copy()
    test_feat = df.iloc[n_train:].copy()
    train_feat['demand'] = y_train

    print("Computing OOF Aggregations...")
    train_feat, test_feat = get_oof_aggregations(train_feat, test_feat, 'geohash_enc', 'demand', ['mean', 'std', 'median', 'count'])
    train_feat, test_feat = get_oof_aggregations(train_feat, test_feat, 'hour', 'demand', ['mean', 'std'])
    train_feat, test_feat = get_oof_aggregations(train_feat, test_feat, 'RoadType_enc', 'demand', ['mean', 'std'])
    train_feat, test_feat = get_oof_aggregations(train_feat, test_feat, ['geohash_enc', 'hour'], 'demand', ['mean'])

    drop_cols = ['Index', 'geohash', 'timestamp', 'demand', 'RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
    feature_cols = [col for col in train_feat.columns if col not in drop_cols]

    X_train_full = train_feat[feature_cols].values
    X_test  = test_feat[feature_cols].values

    split_idx = int(len(train_feat) * 0.8)
    X_train_val = X_train_full[:split_idx]
    y_train_val = y_train[:split_idx]
    X_val = X_train_full[split_idx:]
    y_val = y_train[split_idx:]

    return X_train_val, y_train_val, X_val, y_val, X_test, test_idx, feature_cols, X_train_full, y_train

X_train_val, y_train_val, X_val, y_val, X_test, test_idx, feature_cols, X_full, y_full = engineer_features(train, test)
"""))

cells.append(nbf.v4.new_markdown_cell("""## Validation (80/20 Time Split)"""))

cells.append(nbf.v4.new_code_cell("""lgb_params = {
    'objective': 'regression', 'metric': 'rmse', 'num_leaves': 42, 'max_depth': 11, 'min_child_samples': 20,
    'learning_rate': 0.03, 'n_estimators': 2000, 'reg_lambda': 3.0, 'reg_alpha': 0.1,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': SEED, 'n_jobs': -1, 'verbosity': -1,
}
model_lgb = lgb.LGBMRegressor(**lgb_params)
model_lgb.fit(X_train_val, y_train_val, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
preds_lgb_val = model_lgb.predict(X_val)
r2_lgb = r2_score(y_val, preds_lgb_val)
print(f"Validation R² (LightGBM) = {r2_lgb:.4f}")

xgb_params = {
    'n_estimators': 2000, 'learning_rate': 0.03, 'max_depth': 7, 'min_child_weight': 20,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 3.0,
    'random_state': SEED, 'n_jobs': -1, 'tree_method': 'hist', 'early_stopping_rounds': 100, 'verbosity': 0
}
model_xgb = xgb.XGBRegressor(**xgb_params)
model_xgb.fit(X_train_val, y_train_val, eval_set=[(X_val, y_val)], verbose=False)
preds_xgb_val = model_xgb.predict(X_val)
r2_xgb = r2_score(y_val, preds_xgb_val)
print(f"Validation R² (XGBoost) = {r2_xgb:.4f}")

cat_params = {
    'iterations': 2000, 'learning_rate': 0.03, 'depth': 8, 'l2_leaf_reg': 3.0, 'min_data_in_leaf': 20,
    'random_seed': SEED, 'verbose': 0, 'early_stopping_rounds': 100
}
model_cat = CatBoostRegressor(**cat_params)
model_cat.fit(X_train_val, y_train_val, eval_set=[(X_val, y_val)])
preds_cat_val = model_cat.predict(X_val)
r2_cat = r2_score(y_val, preds_cat_val)
print(f"Validation R² (CatBoost) = {r2_cat:.4f}")

w_lgb, w_xgb, w_cat = 0.45, 0.10, 0.45
preds_ens_val = w_lgb * preds_lgb_val + w_xgb * preds_xgb_val + w_cat * preds_cat_val
r2_ens = r2_score(y_val, preds_ens_val)
print(f"Validation R² (Ensemble) = {r2_ens:.4f}")
"""))

cells.append(nbf.v4.new_markdown_cell("""## 100% Retraining & Submission"""))

cells.append(nbf.v4.new_code_cell("""print("\\nRetraining models on FULL 100% dataset...")
lgb_params['n_estimators'] = 800
model_lgb = lgb.LGBMRegressor(**lgb_params)
model_lgb.fit(X_full, y_full)
preds_lgb = model_lgb.predict(X_test)

xgb_params['n_estimators'] = 800
del xgb_params['early_stopping_rounds']
model_xgb = xgb.XGBRegressor(**xgb_params)
model_xgb.fit(X_full, y_full, verbose=False)
preds_xgb = model_xgb.predict(X_test)

cat_params['iterations'] = 800
del cat_params['early_stopping_rounds']
model_cat = CatBoostRegressor(**cat_params)
model_cat.fit(X_full, y_full)
preds_cat = model_cat.predict(X_test)

preds_ens = w_lgb * preds_lgb + w_xgb * preds_xgb + w_cat * preds_cat
preds_ens = np.clip(preds_ens, y_full.min(), y_full.max())

submission = pd.DataFrame({'Index': test_idx, 'demand': preds_ens})
submission.to_csv('submission.csv', index=False)
print(f"Final Submission shape: {submission.shape}")
"""))

nb.cells.extend(cells)

with open('traffic_demand_v3.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("traffic_demand_v3.ipynb successfully generated!")
