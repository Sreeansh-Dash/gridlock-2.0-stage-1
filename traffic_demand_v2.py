import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pygeohash as pgh
from sklearn.preprocessing import LabelEncoder
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

def engineer_features(train, test):
    # Sort train by day and time for time-based split
    train['time_minutes_temp'] = train['timestamp'].astype(str).str.split(':', expand=True)[0].astype(int) * 60 + train['timestamp'].astype(str).str.split(':', expand=True)[1].astype(int)
    train = train.sort_values(['day', 'time_minutes_temp']).reset_index(drop=True)
    train.drop('time_minutes_temp', axis=1, inplace=True)
    
    split_idx = int(len(train) * 0.8)
    
    y_train = train['demand'].values
    test_idx = test['Index'].values

    df = pd.concat([train.drop('demand', axis=1), test], axis=0, ignore_index=True)
    n_train = len(train)

    # 3a. Geohash decoding -> lat, lon
    df['lat'] = df['geohash'].apply(lambda g: pgh.decode(g)[0])
    df['lon'] = df['geohash'].apply(lambda g: pgh.decode(g)[1])
    # DROP geohash_enc to prevent overfitting
    
    # 3b. Timestamp features
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

    # 3c. Day features
    df['day'] = df['day'].astype(int)
    df['day_mod7'] = df['day'] % 7
    df['is_weekend'] = (df['day_mod7'].isin([5, 6])).astype(int)
    df['day_sin'] = np.sin(2 * np.pi * df['day_mod7'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_mod7'] / 7)

    # 3d. Encode categoricals
    cat_cols = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
        df[col + '_enc'] = LabelEncoder().fit_transform(df[col].astype(str))

    df['NumberofLanes'] = pd.to_numeric(df['NumberofLanes'], errors='coerce')
    df['Temperature'] = df['Temperature'].fillna(df['Temperature'].median())
    df['NumberofLanes'] = df['NumberofLanes'].fillna(df['NumberofLanes'].median())

    # 3f. Interaction features
    df['hour_road'] = df['hour'] * df['RoadType_enc']
    df['temp_bin'] = pd.cut(df['Temperature'], bins=[-50, 5, 15, 25, 50], labels=[0, 1, 2, 3], include_lowest=True).astype(float).fillna(1).astype(int)
    df['weather_largeveh'] = df['Weather_enc'] * df['LargeVehicles_enc']
    df['temp_road'] = df['Temperature'] * df['RoadType_enc']
    
    df['geohash_prefix4_enc'] = LabelEncoder().fit_transform(df['geohash'].str[:4].astype(str))
    # DROP geohash_prefix5_enc
    
    df['lanes_road'] = df['NumberofLanes'] * df['RoadType_enc']
    df['landmarks_road'] = df['Landmarks_enc'] * df['RoadType_enc']

    # 3g. Aggregate features (train only on split_idx)
    train_tmp = df.iloc[:split_idx].copy()
    train_tmp['demand'] = y_train[:split_idx]

    geo_stats = train_tmp.groupby('geohash')['demand'].agg(['mean', 'std', 'median', 'count']).reset_index()
    geo_stats.columns = ['geohash', 'geo_demand_mean', 'geo_demand_std', 'geo_demand_median', 'geo_count']
    geo_stats['geo_demand_std'] = geo_stats['geo_demand_std'].fillna(0)
    df = df.merge(geo_stats, on='geohash', how='left')
    df['geo_demand_mean'] = df['geo_demand_mean'].fillna(y_train[:split_idx].mean())
    df['geo_demand_std'] = df['geo_demand_std'].fillna(0)
    df['geo_demand_median'] = df['geo_demand_median'].fillna(y_train[:split_idx].mean())
    df['geo_count'] = df['geo_count'].fillna(0)

    hour_stats = train_tmp.groupby('hour')['demand'].agg(['mean', 'std']).reset_index()
    hour_stats.columns = ['hour', 'hour_demand_mean', 'hour_demand_std']
    df = df.merge(hour_stats, on='hour', how='left')
    df['hour_demand_mean'] = df['hour_demand_mean'].fillna(y_train[:split_idx].mean())
    df['hour_demand_std'] = df['hour_demand_std'].fillna(0)

    road_stats = train_tmp.groupby('RoadType_enc')['demand'].agg(['mean', 'std']).reset_index()
    road_stats.columns = ['RoadType_enc', 'road_demand_mean', 'road_demand_std']
    df = df.merge(road_stats, on='RoadType_enc', how='left')
    df['road_demand_mean'] = df['road_demand_mean'].fillna(y_train[:split_idx].mean())
    df['road_demand_std'] = df['road_demand_std'].fillna(0)

    geo_hour_stats = train_tmp.groupby(['geohash', 'hour'])['demand'].agg(['mean']).reset_index()
    geo_hour_stats.columns = ['geohash', 'hour', 'geo_hour_demand_mean']
    df = df.merge(geo_hour_stats, on=['geohash', 'hour'], how='left')
    df['geo_hour_demand_mean'] = df['geo_hour_demand_mean'].fillna(df['geo_demand_mean'])

    # Select features
    feature_cols = [
        'lat', 'lon',
        'hour', 'minute', 'time_minutes', 'is_rush_hour', 'time_of_day_bin',
        'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
        'day', 'day_mod7', 'is_weekend', 'day_sin', 'day_cos',
        'RoadType_enc', 'Weather_enc', 'LargeVehicles_enc', 'Landmarks_enc',
        'NumberofLanes', 'Temperature', 'temp_bin',
        'hour_road', 'weather_largeveh', 'temp_road',
        'geohash_prefix4_enc',
        'lanes_road', 'landmarks_road',
        'geo_demand_mean', 'geo_demand_std', 'geo_demand_median', 'geo_count',
        'hour_demand_mean', 'hour_demand_std',
        'road_demand_mean', 'road_demand_std',
        'geo_hour_demand_mean',
    ]

    X_train_full = df.iloc[:n_train][feature_cols].values
    X_test  = df.iloc[n_train:][feature_cols].values

    # Train/Val Split (80/20 Time-Based)
    X_train = X_train_full[:split_idx]
    y_train_split = y_train[:split_idx]
    X_val = X_train_full[split_idx:]
    y_val = y_train[split_idx:]

    return X_train, y_train_split, X_val, y_val, X_test, test_idx, feature_cols, X_train_full, y_train

def train_models(X_train, y_train, X_val, y_val, X_test):
    print("Training LightGBM...")
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'min_child_samples': 50,
        'learning_rate': 0.01,
        'n_estimators': 3000,
        'reg_lambda': 5.0,
        'reg_alpha': 1.0,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'random_state': SEED,
        'n_jobs': -1,
        'verbosity': -1,
    }
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(150, verbose=False)])
    preds_lgb_val = model_lgb.predict(X_val)
    preds_lgb_test = model_lgb.predict(X_test)
    r2_lgb = r2_score(y_val, preds_lgb_val)
    print(f"New local validation R² (LightGBM) = {r2_lgb:.4f}")

    print("Training XGBoost...")
    xgb_params = {
        'n_estimators': 3000,
        'learning_rate': 0.01,
        'max_depth': 6,
        'min_child_weight': 50,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 1.0,
        'reg_lambda': 5.0,
        'random_state': SEED,
        'n_jobs': -1,
        'tree_method': 'hist',
        'early_stopping_rounds': 150,
        'verbosity': 0
    }
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds_xgb_val = model_xgb.predict(X_val)
    preds_xgb_test = model_xgb.predict(X_test)
    r2_xgb = r2_score(y_val, preds_xgb_val)
    print(f"New local validation R² (XGBoost) = {r2_xgb:.4f}")

    print("Training CatBoost...")
    cat_params = {
        'iterations': 3000,
        'learning_rate': 0.01,
        'depth': 6,
        'l2_leaf_reg': 5.0,
        'min_data_in_leaf': 50,
        'random_seed': SEED,
        'verbose': 0,
        'early_stopping_rounds': 150
    }
    model_cat = CatBoostRegressor(**cat_params)
    model_cat.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    preds_cat_val = model_cat.predict(X_val)
    preds_cat_test = model_cat.predict(X_test)
    r2_cat = r2_score(y_val, preds_cat_val)
    print(f"New local validation R² (CatBoost) = {r2_cat:.4f}")

    # Ensemble
    w_lgb, w_xgb, w_cat = 0.45, 0.10, 0.45
    preds_ens_val = w_lgb * preds_lgb_val + w_xgb * preds_xgb_val + w_cat * preds_cat_val
    r2_ens = r2_score(y_val, preds_ens_val)
    print(f"New local validation R² (Ensemble) = {r2_ens:.4f}")
    
    preds_ens_test = w_lgb * preds_lgb_test + w_xgb * preds_xgb_test + w_cat * preds_cat_test
    return preds_ens_test, r2_lgb, r2_xgb, r2_cat, r2_ens

def main():
    train, test = load_data()
    X_train, y_train_split, X_val, y_val, X_test, test_idx, feature_cols, X_full, y_full = engineer_features(train, test)
    preds_test, r2_lgb, r2_xgb, r2_cat, r2_ens = train_models(X_train, y_train_split, X_val, y_val, X_test)
    
    # Clip to train range
    preds_test = np.clip(preds_test, y_full.min(), y_full.max())
    
    submission = pd.DataFrame({
        'Index': test_idx,
        'demand': preds_test
    })
    submission.to_csv('submission.csv', index=False)
    print(f"Submission shape: {submission.shape}")
    
    with open("results.txt", "w") as f:
        f.write(f"{r2_lgb:.4f},{r2_xgb:.4f},{r2_cat:.4f},{r2_ens:.4f}")

if __name__ == '__main__':
    main()
