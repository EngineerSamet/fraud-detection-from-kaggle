# ⚡ Credit Card Fraud Detection - Quick Start Guide

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-green?logo=microsoft)
![Runtime](https://img.shields.io/badge/Runtime-~12--15%20min-yellow)

**🎯 Get the champion fraud detection model running in 5 minutes!**

---

## 📋 Prerequisites

- **Python 3.11+** (recommended) or Python 3.8+
- **8GB RAM** minimum (16GB recommended)
- **500MB** disk space for dataset + outputs
- **Internet connection** for dataset download

---

## 🚀 Installation (3 Steps)

### Step 1: Clone/Download Project

```bash
git clone https://github.com/EngineerSamet/fraud-detection.git
cd fraud-detection
```

Or download ZIP and extract.

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies installed:**
- `numpy`, `pandas` - Data manipulation
- `scikit-learn` - Base ML framework
- `xgboost`, `lightgbm` - Gradient boosting models
- `imbalanced-learn` - Imbalance handling (EasyEnsemble, BalancedBagging)
- `matplotlib`, `seaborn` - Visualization
- `shap` - Model interpretability

**Installation time:** ~2-3 minutes

### Step 3: Download Dataset

1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Click **Download** (143 MB)
3. Extract `creditcard.csv`
4. Place in: `data/raw/creditcard.csv`

**Folder structure:**
```
ML Project/
├── data/
│   └── raw/
│       └── creditcard.csv  ← PUT FILE HERE
├── main.py
├── predict_fraud.py
└── requirements.txt
```

---

## 🏃 Run Training Pipeline

### Execute Full Pipeline

```bash
python main.py
```

**What happens:**

```
[Phase 1] Data Exploration & Preprocessing      (1-2 min)
  ✅ Load 284,807 transactions
  ✅ Detect class imbalance (0.17% fraud)
  ✅ Outlier removal with IQR method
  ✅ Anomaly detection with IsolationForest

[Phase 2] Imbalance Strategy Comparison         (2-3 min)
  ✅ Test class weights, EasyEnsemble, BalancedBagging
  ✅ Train baseline models

[Phase 3] Model Training (17 Configurations)    (4-6 min)
  ✅ Logistic Regression (baseline + class weights)
  ✅ Random Forest (baseline, class weights, optimized)
  ✅ XGBoost (baseline, scale_pos_weight, optimized)
  ✅ LightGBM (class weights, optimized F2) ⭐ CHAMPION
  ✅ Voting Ensemble (LGBM + XGB + RF)
  ✅ EasyEnsemble Classifier
  ✅ BalancedBagging Classifier

[Phase 4] Model Calibration                      (1 min)
  ✅ Isotonic calibration for LGBM, XGB, RF
  ✅ Brier score analysis

[Phase 5] Threshold Optimization                 (2 min)
  ✅ F2-Score optimization (LGBM)
  ✅ Cost-sensitive analysis (4 ratios for XGB & LGBM)
  ✅ Youden's J statistic

[Phase 6] SHAP Analysis                          (2-3 min)
  ✅ Global feature importance (XGBoost)
  ✅ Fraud-only SHAP analysis
  ✅ Feature interaction detection

[Phase 7] Cross-Validation                       (2 min)
  ✅ 5-fold stratified CV (4 models)
  ✅ Stability analysis

[Phase 8] Results Saving                         (10 sec)
  ✅ Save models to outputs/fraud_detection_final/models/
  ✅ Save results to outputs/fraud_detection_final/results/
  ✅ Generate 38 visualizations
```

**Total Runtime:** ~12-15 minutes on modern CPU (Intel i5/i7, AMD Ryzen 5/7)

---

## 🎁 What You'll Get

### 📊 Trained Models (9 files)

**Location:** `outputs/fraud_detection_final/models/`

1. **`lightgbm_champion.pkl`** - Champion model (88.07% PR-AUC)
2. **`xgboost.pkl`** - XGBoost baseline model
3. **`xgboost_calibrated_isotonic.pkl`** - XGBoost calibrated (87.88% PR-AUC)
4. **`random_forest.pkl`** - Random Forest baseline
5. **`random_forest_calibrated.pkl`** - Random Forest calibrated (84.95% PR-AUC)
6. **`logistic_regression.pkl`** - Logistic Regression baseline (72.13% PR-AUC)
7. **`voting_ensemble.pkl`** - Voting ensemble LGBM+XGB+RF (87.24% PR-AUC)
8. **`balanced_bagging_hybrid.pkl`** - BalancedBagging (58.46% PR-AUC)
9. **`isolation_forest_feature.pkl`** - Anomaly detection model

---

### 📈 Results Files (8 files)

**Location:** `outputs/fraud_detection_final/results/`

1. **`model_comparison.csv`** - 17 models ranked by PR-AUC
2. **`lgbm_cost_sensitivity.json`** - LGBM cost-sensitive thresholds (4 ratios)
3. **`cost_sensitivity_analysis.json`** - XGBoost cost analysis (4 ratios)
4. **`optimal_thresholds.json`** - F2/Youden's J thresholds
5. **`cross_validation_results.json`** - 5-fold CV stability (4 models)
6. **`fraud_only_shap_features.csv`** - Fraud-specific feature importance
7. **`top_shap_features.csv`** - Global SHAP rankings
8. **`all_results.json`** - Complete metrics for all models

---

### 🖼️ Visualizations (38+ PNG files)

**Location:** `outputs/fraud_detection_final/figures/`

#### Class Imbalance & Data Exploration
- `class_distribution.png` - 99.83% normal vs 0.17% fraud
- `fraud_vs_normal_distributions.png` - Feature separation analysis
- `correlation_heatmap.png` - Feature correlations
- `pca_explained_variance.png` - PCA component importance
- `tsne_fraud_vs_normal.png` - 2D clustering visualization

#### Model Performance (17 models × 3 plots each)
- Confusion matrices for all configurations
- PR Curves (precision-recall trade-off)
- ROC Curves (TPR vs FPR)
- `model_comparison_bar.png` - PR-AUC comparison chart

#### Threshold Optimization
- `threshold_analysis_LGBM.png` - Cost-sensitive threshold optimization
- `LGBM_optimal_f2_confusion_matrix.png` - F2-optimized results
- `calibration_comparison_LGBM.png` - Calibration curve analysis

#### SHAP Interpretability (7 files in `shap/` folder)
- `fraud_only_shap_summary.png` ⭐ **Fraud-specific feature importance**
- `fraud_only_shap_bar.png` - Mean |SHAP| for fraud cases
- `XGBoost (with Anomaly Score)_shap_summary.png` - Global SHAP
- `XGBoost (with Anomaly Score)_shap_bar.png` - Feature rankings
- `shap_interaction.png` - Feature interactions

#### Anomaly Detection
- `isolationforest_anomaly_scores.png` - Fraud vs Normal anomaly scores

---

## 🧪 Test Deployment Pipeline

After training completes, test the production inference pipeline:

```bash
python predict_fraud.py
```

**Output Example:**

```
=============================================================================
               🔒 FRAUD DETECTION PREDICTION SYSTEM 🔒
=============================================================================

📦 Loading models...
   ✅ LightGBM model loaded
   ✅ IsolationForest loaded

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                           🔍 PREDICTION RESULTS 🔍

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚨 Fraud Probability: 99.99%
🎯 Risk Level: CRITICAL
🛡️  Recommended Action: BLOCK IMMEDIATELY

┌─────────────────────────────────────────────────────────────────────────┐
│                         Threshold Analysis                              │
├────────────────────┬─────────────┬────────────┬───────────────────────┤
│ Strategy           │ Threshold   │ Prediction │ Confidence            │
├────────────────────┼─────────────┼────────────┼───────────────────────┤
│ default            │ 0.50        │ FRAUD      │ 🔴 Very High         │
│ f2_optimized       │ 0.18        │ FRAUD      │ 🔴 Very High         │
│ cost_50            │ 0.30        │ FRAUD      │ 🔴 Very High         │
│ cost_100           │ 0.23        │ FRAUD      │ 🔴 Very High         │
│ cost_200           │ 0.15        │ FRAUD      │ 🔴 Very High         │
│ cost_500           │ 0.07        │ FRAUD      │ 🔴 Very High         │
└────────────────────┴─────────────┴────────────┴───────────────────────┘
```

---

## 📊 Performance Summary

### Champion Model: LightGBM + Isotonic Calibration

| Metric          | Value   | Interpretation                                    |
|-----------------|---------|---------------------------------------------------|
| **PR-AUC**      | 88.07%  | Excellent fraud detection (gold standard metric)  |
| **Recall**      | 83.87%  | Catches 84 out of 100 frauds                      |
| **Precision**   | 90.70%  | 9 out of 10 alerts are real frauds                |
| **F2-Score**    | 85.15%  | Balanced metric (recall-weighted)                 |
| **ROC-AUC**     | 98.88%  | High true positive rate                           |
| **MCC**         | 0.872   | Excellent correlation (robust metric)             |

### Top 5 Models by PR-AUC

| Rank | Model Configuration      | PR-AUC | Recall | Precision |
|------|--------------------------|--------|--------|-----------|
| 1    | LGBM_Calibrated_Isotonic | 88.07% | 83.87% | 90.70%    |
| 2    | XGB_Calibrated_Sigmoid   | 87.91% | 82.80% | 81.91%    |
| 3    | XGB_Calibrated_Isotonic  | 87.88% | 79.57% | 90.24%    |
| 4    | LGBM_ClassWeights        | 87.96% | 84.95% | 84.04%    |
| 5    | Voting_LGBM_XGB_RF       | 87.24% | 84.95% | 85.87%    |

### Cost-Sensitive Analysis

**LGBM - Business Scenario (FN costs 100x more than FP):**

| FN/FP Ratio | Threshold | Recall | Precision | Cost Reduction | Use Case               |
|-------------|-----------|--------|-----------|----------------|------------------------|
| 50:1        | 0.23      | 86.02% | 85.11%    | 12.4%          | Moderate risk          |
| 100:1       | 0.23      | 86.02% | 85.11%    | **12.9%**      | **Banking standard**   |
| 200:1       | 0.23      | 86.02% | 85.11%    | 13.1%          | High-risk industry     |
| 500:1       | 0.23      | 86.02% | 85.11%    | 13.2%          | Critical infrastructure|

**XGBoost - More Aggressive Strategy:**

| FN/FP Ratio | Threshold | Recall | Precision | Cost Reduction |
|-------------|-----------|--------|-----------|----------------|
| 100:1       | 0.07      | 92.47% | 20.72%    | **22.8%**      |
| 200:1       | 0.07      | 92.47% | 20.72%    | **34.3%**      |

**Recommendation:** 
- **Conservative:** LGBM threshold 0.23 (balanced recall/precision)
- **Aggressive:** XGBoost threshold 0.07 (maximum fraud detection)

---

## 🔧 Troubleshooting

### Issue: "FileNotFoundError: creditcard.csv not found"

**Solution:**
```bash
# Check if file exists
Test-Path "data\raw\creditcard.csv"

# If False, download from:
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

### Issue: "ImportError: No module named 'lightgbm'"

**Solution:**
```bash
pip install lightgbm>=4.0.0
```

### Issue: "MemoryError during training"

**Solution:** Reduce dataset size for testing:
```python
# In main.py, after loading data (line ~120):
df = df.sample(n=50000, random_state=42)  # Use 50k transactions
```

### Issue: "SHAP analysis takes too long"

**Solution:** Reduce SHAP sample size:
```python
# In src/shap_analysis.py, line ~53:
X_sample = X_data.sample(n=500, random_state=42)  # Use 500 samples
```

---

## 🎯 Next Steps

### 1. Explore Results (5 min)
- Open `outputs/fraud_detection_final/results/model_comparison.csv`
- View top 5 models and their metrics
- Check cost-sensitive analysis results

### 2. View Visualizations (10 min)
- Confusion matrices show prediction accuracy
- PR curves compare model performance
- SHAP plots explain feature importance

### 3. Test Deployment Pipeline (5 min)
```bash
python predict_fraud.py
```
- See real-time fraud prediction
- Understand different threshold strategies
- View risk level classification

### 4. Read Full Documentation (30 min)
- See `README.md` for complete methodology
- Understand why class weights beat SMOTE
- Learn about model selection rationale

---

## 📚 Key Learnings

### Why LightGBM Won?

**Compared to XGBoost:**
- ✅ Better PR-AUC (88.07% vs 87.91%)
- ✅ More stable threshold (0.23 works across all cost ratios)
- ✅ Faster training (gradient-based tree growth)

**Compared to Random Forest:**
- ✅ Higher PR-AUC (88.07% vs 84.95%)
- ✅ Better recall (83.87% vs 78.49%)
- ✅ More efficient (boosting vs bagging)

### Why Class Weights > SMOTE?

**SMOTE was tested and REJECTED:**
- ❌ Creates synthetic samples in PCA space
- ❌ Risk of unrealistic feature combinations
- ❌ Lower performance in our tests
- ✅ Class weights adjust loss function naturally
- ✅ No fake data generation
- ✅ Better results (88% vs estimated 78% with SMOTE)

**Evidence:** Check `src/imbalance.py` header for detailed explanation

### What is PR-AUC?

**PR-AUC (Precision-Recall Area Under Curve)** is the best metric for imbalanced data.

- **Why not Accuracy?** → 99.8% by predicting all "normal" (useless!)
- **Why not ROC-AUC?** → Can be misleading with 0.17% fraud rate
- **PR-AUC focuses on minority class** → Perfect for fraud detection

**Interpretation:**
- 50% = Random guessing
- 70% = Good
- 80% = Very good
- **88% = Excellent** ⭐

---

## 💬 Support

**Questions?** Contact:
- **Email:** sametsanlikan@gmail.com
- **GitHub:** [@EngineerSamet](https://github.com/EngineerSamet)

---

## ⭐ Project Highlights

✅ **Production-Ready:** Deployment pipeline included (`predict_fraud.py`)  
✅ **Academic Rigor:** 17 model configurations tested  
✅ **Interpretable:** SHAP analysis explains predictions  
✅ **Business-Aware:** Cost-sensitive thresholds for real scenarios  
✅ **Well-Documented:** Comprehensive README + Quick Start  
✅ **Reproducible:** Fixed random seeds, exact dependency versions  

---

**🎉 Happy Fraud Detection!**

If this guide helped you, please ⭐ star the project!
