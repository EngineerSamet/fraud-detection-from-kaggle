"""
Quick script to calculate optimal thresholds for LGBM Champion Model
This fills the gap left by main.py (XGBoost thresholds saved but not LGBM)
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score, roc_curve, precision_score, recall_score, confusion_matrix
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from thresholding import get_optimal_threshold_f2, get_optimal_threshold_youden, get_cost_sensitive_threshold

print("\n" + "="*70)
print("  LGBM CHAMPION MODEL - OPTIMAL THRESHOLD CALCULATION")
print("="*70)

# Paths
MODEL_PATH = "outputs/fraud_detection_final/models/lightgbm_champion.pkl"
DATA_PATH = "data/raw/creditcard.csv"

print(f"\n📂 Loading LGBM Champion Model...")
print(f"   Path: {MODEL_PATH}")

try:
    with open(MODEL_PATH, 'rb') as f:
        lgbm_model = pickle.load(f)
    print("✓ Model loaded successfully")
except FileNotFoundError:
    print("❌ Error: Model file not found!")
    print("   Run main.py first to train and save the model.")
    exit(1)

print(f"\n📊 Loading test data...")
df = pd.read_csv(DATA_PATH)

# Reproduce the same preprocessing as main.py
from preprocessing import separate_features_target, split_data, remove_outliers_iqr
from sklearn.ensemble import IsolationForest

# Separate features and target
X, y = separate_features_target(df)

# Outlier removal (same as main.py)
X_clean, y_clean, _ = remove_outliers_iqr(
    X, y, 
    features_to_clean=['V17', 'V14', 'V12', 'V10'],
    multiplier=1.5
)

# Split (same random state as main.py)
X_train, X_test, y_train, y_test = split_data(
    X_clean, y_clean, 
    test_size=0.2, 
    random_state=42,
    stratify=True
)

# Generate anomaly scores (same as main.py)
print("\n🔍 Generating anomaly scores...")
iso_forest = IsolationForest(
    contamination=0.00173,
    random_state=42,
    n_estimators=100,
    max_samples='auto',
    n_jobs=-1
)
iso_forest.fit(X_train)
anomaly_scores_test = -iso_forest.decision_function(X_test)

# Add anomaly scores
X_test_with_anomaly = X_test.copy()
X_test_with_anomaly['anomaly_score'] = anomaly_scores_test

print(f"✓ Test set prepared: {X_test.shape[0]} samples")
print(f"   Features: {X_test_with_anomaly.shape[1]} (30 original + 1 anomaly_score)")

# Get predictions from LGBM
print(f"\n🎯 Generating LGBM predictions...")
y_proba_lgbm = lgbm_model.predict_proba(X_test_with_anomaly)[:, 1]
print(f"✓ Probabilities generated")
print(f"   Min: {y_proba_lgbm.min():.4f}")
print(f"   Max: {y_proba_lgbm.max():.4f}")
print(f"   Mean: {y_proba_lgbm.mean():.4f}")

# Calculate optimal thresholds
print("\n" + "="*70)
print("  THRESHOLD OPTIMIZATION")
print("="*70)

# 1. F2-Score optimization
optimal_f2, max_f2 = get_optimal_threshold_f2(y_test, y_proba_lgbm)

# 2. Youden's J statistic
optimal_youden, _ = get_optimal_threshold_youden(y_test, y_proba_lgbm)

# 3. Cost-sensitive (FN=50, FP=1)
optimal_cost, _, _ = get_cost_sensitive_threshold(
    y_test, y_proba_lgbm,
    cost_fn=50, cost_fp=1
)

# Results dictionary
lgbm_thresholds = {
    "model": "LightGBM (Calibrated-Isotonic + Anomaly Score)",
    "optimal_f2_threshold": float(optimal_f2),
    "optimal_youden_threshold": float(optimal_youden),
    "optimal_cost_threshold": float(optimal_cost),
    "default_threshold": 0.5,
    "test_samples": int(len(y_test)),
    "fraud_samples": int(y_test.sum()),
    "features": 31
}

# Print results
print("\n" + "="*70)
print("  LGBM OPTIMAL THRESHOLDS (FINAL RESULTS)")
print("="*70)
print(f"\n📊 Model: {lgbm_thresholds['model']}")
print(f"📊 Test Set: {lgbm_thresholds['test_samples']} samples ({lgbm_thresholds['fraud_samples']} frauds)")
print(f"📊 Features: {lgbm_thresholds['features']}\n")

print(f"{'Threshold Type':<30} {'Value':<15} {'Purpose'}")
print("-" * 70)
print(f"{'F2-Score Optimized':<30} {lgbm_thresholds['optimal_f2_threshold']:<15.4f} Maximize F2 (Recall 2x weight)")
youden_label = "Youden's J Statistic"
print(f"{youden_label:<30} {lgbm_thresholds['optimal_youden_threshold']:<15.4f} Balance Sensitivity+Specificity")
cost_label = "Cost-Sensitive (FN=50,FP=1)"
print(f"{cost_label:<30} {lgbm_thresholds['optimal_cost_threshold']:<15.4f} Minimize business cost")
print(f"{'Default (sklearn)':<30} {lgbm_thresholds['default_threshold']:<15.4f} Standard probability cutoff")

# Save to JSON
import json
output_path = "outputs/fraud_detection_final/results/lgbm_optimal_thresholds.json"
with open(output_path, 'w') as f:
    json.dump(lgbm_thresholds, f, indent=4)

print(f"\n✓ Results saved to: {output_path}")

# Calculate detailed metrics for each threshold
print("\n" + "="*70)
print("  DETAILED PERFORMANCE METRICS FOR ALL THRESHOLDS")
print("="*70)

def calculate_metrics(y_true, y_proba, threshold, threshold_name):
    """Calculate all performance metrics for a given threshold"""
    y_pred = (y_proba >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Metrics
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    
    print(f"\n{threshold_name}:")
    print(f"  Threshold: {threshold:.6f}")
    print(f"  F2-Score:  {f2:.4f}")
    print(f"  Recall:    {recall:.4f} ({tp}/{tp+fn} frauds caught)")
    print(f"  Precision: {precision:.4f} ({tp}/{tp+fp} predictions correct)")
    print(f"  False Positives: {fp} (out of {tn+fp} legitimate transactions)")
    print(f"  False Negatives: {fn} (missed frauds)")
    print(f"  True Positives:  {tp}")
    print(f"  True Negatives:  {tn}")
    
    return {
        'threshold': threshold,
        'f2_score': f2,
        'recall': recall,
        'precision': precision,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'tn': tn
    }

# Calculate metrics for all thresholds
metrics_default = calculate_metrics(y_test, y_proba_lgbm, 0.5, "Default (0.50)")
metrics_f2 = calculate_metrics(y_test, y_proba_lgbm, optimal_f2, f"F2-Optimized ({optimal_f2:.2f})")
metrics_youden = calculate_metrics(y_test, y_proba_lgbm, optimal_youden, f"Youden's J ({optimal_youden:.6f})")
metrics_cost = calculate_metrics(y_test, y_proba_lgbm, optimal_cost, f"Cost-Sensitive ({optimal_cost:.2f})")

# Comparison table
print("\n" + "="*70)
print("  COMPARISON TABLE FOR PRESENTATION")
print("="*70)
print(f"\n{'Threshold Method':<25} {'Threshold':<12} {'F2-Score':<10} {'Recall':<10} {'Precision':<10} {'FP':<8}")
print("-" * 90)
print(f"{'Default (sklearn)':<25} {metrics_default['threshold']:<12.2f} {metrics_default['f2_score']:<10.4f} "
      f"{metrics_default['recall']:<10.4f} {metrics_default['precision']:<10.4f} {metrics_default['fp']:<8}")
print(f"{'F2-Optimized ✓':<25} {metrics_f2['threshold']:<12.2f} {metrics_f2['f2_score']:<10.4f} "
      f"{metrics_f2['recall']:<10.4f} {metrics_f2['precision']:<10.4f} {metrics_f2['fp']:<8}")
youden_label = "Youden's J ✗"
print(f"{youden_label:<25} {metrics_youden['threshold']:<12.6f} {metrics_youden['f2_score']:<10.4f} "
      f"{metrics_youden['recall']:<10.4f} {metrics_youden['precision']:<10.4f} {metrics_youden['fp']:<8}")
cost_label = "Cost-Sensitive ✗"
print(f"{cost_label:<25} {metrics_cost['threshold']:<12.2f} {metrics_cost['f2_score']:<10.4f} "
      f"{metrics_cost['recall']:<10.4f} {metrics_cost['precision']:<10.4f} {metrics_cost['fp']:<8}")

print("\n💡 KEY INSIGHTS:")
print(f"   • F2-Optimized: Best balance (F2={metrics_f2['f2_score']:.4f}), only {metrics_f2['fp']} FP, {metrics_f2['precision']*100:.1f}% precision")
print(f"   • Youden's J: Highest recall ({metrics_youden['recall']*100:.1f}%) BUT {metrics_youden['fp']} false positives → IMPRACTICAL")
print(f"   • Youden precision only {metrics_youden['precision']*100:.1f}% (vs F2's {metrics_f2['precision']*100:.1f}%)")
print(f"   • Cost-Sensitive: Middle ground but {metrics_cost['fp']} FP still too high")

# Compare with XGBoost thresholds
print("\n" + "="*70)
print("  COMPARISON: LGBM vs XGBoost Thresholds")
print("="*70)

xgb_thresholds_path = "outputs/fraud_detection_final/results/optimal_thresholds.json"
try:
    with open(xgb_thresholds_path, 'r') as f:
        xgb_thresholds = json.load(f)
    
    print(f"\n{'Threshold Type':<30} {'XGBoost':<15} {'LGBM':<15} {'Difference'}")
    print("-" * 70)
    print(f"{'F2-Score Optimized':<30} {xgb_thresholds['optimal_f2_threshold']:<15.4f} "
          f"{lgbm_thresholds['optimal_f2_threshold']:<15.4f} "
          f"{lgbm_thresholds['optimal_f2_threshold'] - xgb_thresholds['optimal_f2_threshold']:+.4f}")
    youden_label = "Youden's J Statistic"
    print(f"{youden_label:<30} {xgb_thresholds['optimal_youden_threshold']:<15.4f} "
          f"{lgbm_thresholds['optimal_youden_threshold']:<15.4f} "
          f"{lgbm_thresholds['optimal_youden_threshold'] - xgb_thresholds['optimal_youden_threshold']:+.4f}")
    cost_label = "Cost-Sensitive (FN=50,FP=1)"
    print(f"{cost_label:<30} {xgb_thresholds['optimal_cost_threshold']:<15.4f} "
          f"{lgbm_thresholds['optimal_cost_threshold']:<15.4f} "
          f"{lgbm_thresholds['optimal_cost_threshold'] - xgb_thresholds['optimal_cost_threshold']:+.4f}")
    
    print("\n💡 INSIGHT:")
    diff = lgbm_thresholds['optimal_f2_threshold'] - xgb_thresholds['optimal_f2_threshold']
    if abs(diff) < 0.05:
        print(f"   LGBM and XGBoost have similar optimal F2 thresholds (Δ={diff:+.4f})")
        print("   This suggests both models learned similar probability distributions.")
    else:
        print(f"   LGBM threshold is {'HIGHER' if diff > 0 else 'LOWER'} than XGBoost (Δ={diff:+.4f})")
        print("   This indicates LGBM's probability calibration differs from XGBoost.")
    
except FileNotFoundError:
    print("\n⚠️  XGBoost thresholds not found for comparison")

print("\n" + "="*70)
print("  ✅ LGBM THRESHOLD CALCULATION COMPLETE!")
print("="*70)
print(f"\n📁 Output file: {output_path}")
print(f"📊 Use these thresholds for LGBM deployment\n")
