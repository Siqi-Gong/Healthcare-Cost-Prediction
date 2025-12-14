# Healthcare Cost Prediction using LightGBM and Conformal Prediction
# Author: Siqi Gong
# Description: This pipeline implements feature engineering, time-series forecasting, 
# and uncertainty quantification for hospital cost data.

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ========== CONFIGURATION & COLUMN MAPPING ==========
COLS = {
    "cost": "mean_cost",
    "year": "year",
    "severity": "apr_severity_of_illness_code",
    "hospital": "facility_name",
    "diagnosis_text": "apr_drg_description",
    "drg_code": "apr_drg_code",
    "discharges": "discharges", 
    "med_surg": "apr_medical_surgical_description"
}

def parse_args():
    parser = argparse.ArgumentParser(description="Healthcare Cost Prediction Pipeline")
    # Defaults to looking for data.csv in a 'data' folder one level up, or current directory
    default_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'data.csv')
    if not os.path.exists(default_path):
        default_path = 'data.csv' # Fallback to current dir
        
    parser.add_argument("--csv", type=str, default=default_path, help="Path to input dataset")
    parser.add_argument("--svd_components", type=int, default=100, help="Dimensions for text embeddings")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

# ========== FEATURE ENGINEERING ==========

class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target Encoder with leakage prevention (fit on train, transform on test).
    """
    def __init__(self, col_name, target_name):
        self.col_name = col_name
        self.target_name = target_name
        self.map_ = {}
        self.global_mean_ = 0

    def fit(self, X, y=None):
        self.global_mean_ = X[self.target_name].mean()
        self.map_ = X.groupby(self.col_name)[self.target_name].mean().to_dict()
        return self

    def transform(self, X):
        return X[self.col_name].map(self.map_).fillna(self.global_mean_).values.reshape(-1, 1)

def build_lag_features(df):
    """
    Constructs temporal features (Lag-1 Year Cost and YoY Growth Rate).
    """
    print("[Preprocessing] Engineering temporal features...")
    df = df.sort_values(by=[COLS['hospital'], COLS['drg_code'], COLS['year']])
    
    # 1-Year Lag
    df['lag_cost_1y'] = df.groupby([COLS['hospital'], COLS['drg_code']])[COLS['cost']].shift(1)
    
    # Growth Rate
    prev_cost = df.groupby([COLS['hospital'], COLS['drg_code']])[COLS['cost']].shift(2)
    df['cost_growth'] = df['lag_cost_1y'] / prev_cost - 1
    
    # Impute initial NaNs
    mean_cost = df[COLS['cost']].mean()
    df['lag_cost_1y'] = df['lag_cost_1y'].fillna(mean_cost)
    df['cost_growth'] = df['cost_growth'].fillna(0)
    
    return df

def get_feature_names(column_transformer, num_features, cat_features, svd_n_components):
    """
    Helper to reconstruct feature names after pipeline transformation for SHAP.
    """
    feature_names = []
    # Text SVD
    feature_names.extend([f"text_svd_{i}" for i in range(svd_n_components)])
    # Categorical
    try:
        ohe = column_transformer.named_transformers_['cat'].named_steps['onehot']
        cat_names = ohe.get_feature_names_out(cat_features)
        feature_names.extend(cat_names)
    except:
        feature_names.extend([f"cat_{i}" for i in range(len(cat_features) * 4)])
    # Numerical
    feature_names.extend(num_features)
    # Target Encoded
    feature_names.append("hospital_target_enc")
    return feature_names

# ========== UNCERTAINTY QUANTIFICATION ==========

def conformal_prediction_interval(model, X_val, y_val, X_test, alpha=0.1):
    """
    Calibrates prediction intervals using Conformal Prediction (Split Conformal).
    Target Coverage = 1 - alpha.
    """
    print("[Uncertainty] Calibrating Conformal Prediction intervals...")
    preds_val = model.predict(X_val)
    # Non-conformity scores (Absolute Residuals)
    scores_val = np.abs(y_val - preds_val)
    
    # Calculate q-hat (1-alpha quantile of non-conformity scores)
    n = len(y_val)
    q_val = np.quantile(scores_val, np.ceil((n + 1) * (1 - alpha)) / n)
    
    # Construct intervals
    preds_test = model.predict(X_test)
    lower = preds_test - q_val
    upper = preds_test + q_val
    
    return lower, upper, q_val

# ========== VISUALIZATION & INTERPRETATION ==========

def run_shap_analysis(model, X_test, feature_names, sample_size=2000, output_dir="results"):
    """
    Generates SHAP summary and importance plots.
    """
    print("\n[Interpretation] Running SHAP analysis...")
    os.makedirs(output_dir, exist_ok=True)
    
    if len(X_test) > sample_size:
        np.random.seed(42)
        indices = np.random.choice(X_test.shape[0], sample_size, replace=False)
        X_shap = X_test[indices]
    else:
        X_shap = X_test

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    # Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False)
    plt.title("SHAP Summary: Drivers of Hospital Costs", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"))
    plt.close()

    # Importance Bar Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("Feature Importance (Mean |SHAP|)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_importance.png"))
    plt.close()
    print(f"[Done] SHAP plots saved to {output_dir}/")

# ========== MAIN PIPELINE ==========

def main():
    args = parse_args()
    
    # 1. Data Loading
    print(f"[Info] Loading data from {args.csv}...")
    if not os.path.exists(args.csv):
        print(f"Error: Dataset not found at {args.csv}")
        return
        
    df = pd.read_csv(args.csv)
    
    # Basic Cleaning
    df = df.dropna(subset=[COLS['cost'], COLS['severity'], COLS['diagnosis_text']])
    df = df[df[COLS['cost']] > 0].copy()
    
    # Log Transformation
    df['log_cost'] = np.log1p(df[COLS['cost']])
    
    # Feature Engineering
    df = build_lag_features(df)
    
    # 2. Temporal Splitting Strategy
    # Train: 2009-2015, Validation: 2016, Test: 2017 (Future generalization)
    print("[Info] Splitting data by time (Train < 2016, Val=2016, Test=2017)...")
    train_mask = df[COLS['year']] <= 2015
    val_mask = df[COLS['year']] == 2016
    test_mask = df[COLS['year']] == 2017
    
    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()
    df_test = df[test_mask].copy()
    
    print(f"[Split Stats] Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    # 3. Pipeline Definition
    # Text Processing (TF-IDF + SVD)
    text_pipeline = Pipeline([
        ('selector', FunctionTransformer(lambda x: x[COLS['diagnosis_text']], validate=False)),
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9)),
        ('svd', TruncatedSVD(n_components=args.svd_components, random_state=args.seed))
    ])
    
    # Categorical Processing
    cat_features = [COLS['severity'], COLS['med_surg']]
    cat_pipeline = Pipeline([
        ('selector', FunctionTransformer(lambda x: x[cat_features].astype(str), validate=False)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Numerical Processing
    num_features = ['lag_cost_1y', 'cost_growth']
    num_pipeline = Pipeline([
        ('selector', FunctionTransformer(lambda x: x[num_features], validate=False)),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer([
        ('text', text_pipeline, [COLS['diagnosis_text']]),
        ('cat', cat_pipeline, cat_features),
        ('num', num_pipeline, num_features)
    ])
    
    # Feature Transformation
    print("[Info] Transforming features...")
    # Target Encoding for Facility Name (fit only on train to prevent leakage)
    te = TargetEncoder(COLS['hospital'], 'log_cost')
    te.fit(df_train)
    
    def get_matrix(dataset):
        X_base = preprocessor.fit_transform(dataset) if dataset is df_train else preprocessor.transform(dataset)
        X_hosp = te.transform(dataset)
        X = np.hstack([X_base, X_hosp])
        y = dataset['log_cost'].values
        w = dataset[COLS['discharges']].fillna(1.0).values 
        return X, y, w

    X_train, y_train, w_train = get_matrix(df_train)
    X_val, y_val, w_val = get_matrix(df_val)
    X_test, y_test, w_test = get_matrix(df_test)
    
    # 4. Modeling
    
    # Baseline: ElasticNet
    print("\n[Model] Training Baseline (ElasticNet)...")
    enet = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=args.seed)
    enet.fit(X_train, y_train, sample_weight=w_train)
    y_pred_enet = np.expm1(enet.predict(X_test))
    y_true = np.expm1(y_test)
    print(f"ElasticNet MAE: {mean_absolute_error(y_true, y_pred_enet):.2f}")
    
    # Advanced: LightGBM
    print("\n[Model] Training LightGBM (Gradient Boosting)...")
    lgb_params = {
        'objective': 'regression',
        'metric': 'mae',
        'n_estimators': 3000,
        'learning_rate': 0.05,
        'num_leaves': 63,
        'random_state': args.seed,
        'verbose': -1,
        'n_jobs': -1
    }
    
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    y_pred_log = model_lgb.predict(X_test)
    y_pred_lgb = np.expm1(y_pred_log)
    
    # Performance Metrics
    mae_lgb = mean_absolute_error(y_true, y_pred_lgb)
    r2_lgb = r2_score(y_true, y_pred_lgb)
    
    print(f"LightGBM MAE: {mae_lgb:.2f}")
    print(f"LightGBM R2 Score: {r2_lgb:.4f}")
    
    # 5. Uncertainty Quantification
    print("\n[Uncertainty] Calculating Prediction Intervals...")
    
    # A. Quantile Regression (for comparison)
    lgb_low = lgb.LGBMRegressor(objective='quantile', alpha=0.1, n_estimators=500, verbose=-1)
    lgb_high = lgb.LGBMRegressor(objective='quantile', alpha=0.9, n_estimators=500, verbose=-1)
    lgb_low.fit(X_train, y_train, sample_weight=w_train)
    lgb_high.fit(X_train, y_train, sample_weight=w_train)
    
    # Clip to prevent overflow during expm1
    q_low = np.expm1(np.clip(lgb_low.predict(X_test), 0, 20))
    q_high = np.expm1(np.clip(lgb_high.predict(X_test), 0, 20))
    
    # B. Conformal Prediction
    lower_cp, upper_cp, _ = conformal_prediction_interval(model_lgb, X_val, y_val, X_test, alpha=0.1)
    lower_cp_real = np.expm1(np.clip(lower_cp, 0, 20))
    upper_cp_real = np.expm1(np.clip(upper_cp, 0, 20))
    lower_cp_real = np.maximum(lower_cp_real, 0) # Cost cannot be negative
    
    print(f"Quantile Reg Coverage (10-90): {np.mean((y_true >= q_low) & (y_true <= q_high)):.2%}")
    print(f"Conformal Coverage (Target 90%): {np.mean((y_true >= lower_cp_real) & (y_true <= upper_cp_real)):.2%}")

    # 6. Visualization & Artifacts
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Forecast Plot
    plt.figure(figsize=(12, 6))
    subset_idx = np.random.choice(len(y_true), 100, replace=False)
    subset_idx.sort()
    
    plt.plot(y_true[subset_idx], label='Actual Cost', marker='o', linestyle='', markersize=5, alpha=0.7)
    plt.plot(y_pred_lgb[subset_idx], label='Predicted Cost', alpha=0.8)
    plt.fill_between(range(100), lower_cp_real[subset_idx], upper_cp_real[subset_idx], 
                     color='gray', alpha=0.2, label='Conformal Interval (90%)')
    
    plt.title("Hospital Cost Forecasting (Test Set 2017)")
    plt.ylabel("Cost ($)")
    plt.xlabel("Sample Index (Subset)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "forecast_result.png"))
    print(f"[Done] Forecast plot saved to {output_dir}/")
    
    # 7. Model Interpretation (SHAP)
    # Reconstruct feature names
    cat_features_list = [COLS['severity'], COLS['med_surg']]
    num_features_list = ['lag_cost_1y', 'cost_growth']
    
    feat_names = get_feature_names(
        preprocessor, 
        num_features_list, 
        cat_features_list, 
        args.svd_components
    )
    
    try:
        run_shap_analysis(model_lgb, X_test, feat_names, output_dir=output_dir)
    except Exception as e:
        print(f"[Warning] Could not run SHAP analysis: {e}")

if __name__ == "__main__":
    main()
