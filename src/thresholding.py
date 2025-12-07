"""
Threshold Tuning Module for Credit Card Fraud Detection

CRITICAL CONCEPT: Default threshold (0.5) is NOT optimal for fraud detection!

In fraud detection:
- Catching frauds (Recall) is more important than avoiding false alarms
- F2-Score weights Recall 2x more than Precision
- Cost-based thresholding considers business impact

This module provides:
- Optimal threshold for F2-Score
- Optimal threshold for Youden's J statistic
- Threshold analysis via grid search
- Cost-sensitive threshold optimization
"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, fbeta_score,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt


def get_optimal_threshold_f2(y_true, y_proba):
    """
    Find optimal threshold maximizing F2-Score.
    
    F2-Score gives 2x weight to Recall compared to Precision.
    This is appropriate for fraud detection where missing frauds
    is more costly than false alarms.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Prediction probabilities
        
    Returns:
    --------
    optimal_threshold : float
        Threshold that maximizes F2-Score
    best_f2 : float
        Maximum F2-Score achieved
    """
    print("\n" + "="*60)
    print("Finding Optimal Threshold (F2-Score)")
    print("="*60)
    
    # Try different thresholds
    thresholds = np.arange(0.1, 0.9, 0.01)
    f2_scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f2 = fbeta_score(y_true, y_pred, beta=2)
        f2_scores.append(f2)
    
    best_idx = np.argmax(f2_scores)
    optimal_threshold = thresholds[best_idx]
    best_f2 = f2_scores[best_idx]
    
    # Also get metrics at default threshold for comparison
    y_pred_default = (y_proba >= 0.5).astype(int)
    f2_default = fbeta_score(y_true, y_pred_default, beta=2)
    
    print(f"\nDefault threshold (0.5):")
    print(f"  F2-Score: {f2_default:.4f}")
    
    print(f"\nOptimal threshold ({optimal_threshold:.2f}):")
    print(f"  F2-Score: {best_f2:.4f}")
    print(f"  Improvement: {((best_f2 - f2_default) / f2_default * 100):.2f}%")
    
    return optimal_threshold, best_f2


def get_optimal_threshold_youden(y_true, y_proba):
    """
    Find optimal threshold using Youden's J statistic.
    
    Youden's J = Sensitivity + Specificity - 1
    This balances True Positive Rate and True Negative Rate.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Prediction probabilities
        
    Returns:
    --------
    optimal_threshold : float
        Threshold that maximizes Youden's J
    best_j : float
        Maximum J statistic
    """
    print("\n" + "="*60)
    print("Finding Optimal Threshold (Youden's J)")
    print("="*60)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    # Youden's J = TPR - FPR = Sensitivity + Specificity - 1
    j_scores = tpr - fpr
    
    best_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[best_idx]
    best_j = j_scores[best_idx]
    
    print(f"\nOptimal threshold (Youden's J): {optimal_threshold:.4f}")
    print(f"  J statistic: {best_j:.4f}")
    print(f"  Sensitivity (TPR): {tpr[best_idx]:.4f}")
    print(f"  Specificity (1-FPR): {1 - fpr[best_idx]:.4f}")
    
    return optimal_threshold, best_j


def get_optimal_threshold_precision_recall(y_true, y_proba, target_recall=0.9):
    """
    Find threshold that achieves target recall with maximum precision.
    
    In fraud detection, we often want to guarantee high recall
    (e.g., catch at least 90% of frauds) while maximizing precision.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Prediction probabilities
    target_recall : float
        Minimum desired recall (default: 0.9)
        
    Returns:
    --------
    optimal_threshold : float
        Threshold achieving target recall with best precision
    achieved_metrics : dict
        Precision and recall at optimal threshold
    """
    print("\n" + "="*60)
    print(f"Finding Threshold for Target Recall ≥ {target_recall}")
    print("="*60)
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Find thresholds that meet target recall
    valid_indices = np.where(recall >= target_recall)[0]
    
    if len(valid_indices) == 0:
        print(f"⚠️  Warning: Cannot achieve recall ≥ {target_recall}")
        print(f"   Maximum achievable recall: {recall.max():.4f}")
        return None, None
    
    # Among valid thresholds, choose one with highest precision
    best_idx = valid_indices[np.argmax(precision[valid_indices])]
    optimal_threshold = thresholds[best_idx]
    
    achieved_metrics = {
        'precision': precision[best_idx],
        'recall': recall[best_idx],
        'threshold': optimal_threshold
    }
    
    print(f"\nOptimal threshold: {optimal_threshold:.4f}")
    print(f"  Precision: {achieved_metrics['precision']:.4f}")
    print(f"  Recall: {achieved_metrics['recall']:.4f}")
    
    return optimal_threshold, achieved_metrics


def threshold_grid_search(y_true, y_proba, thresholds=None):
    """
    Comprehensive threshold analysis with multiple metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Prediction probabilities
    thresholds : array-like or None
        Thresholds to evaluate. If None, uses np.arange(0.1, 0.9, 0.05)
        
    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame with metrics for each threshold
    """
    import pandas as pd
    
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        f2 = fbeta_score(y_true, y_pred, beta=2)
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'f2_score': f2
        })
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("Threshold Grid Search Results")
    print("="*60)
    print(results_df.to_string(index=False))
    
    return results_df


def plot_threshold_analysis(y_true, y_proba, save_path=None):
    """
    Plot how Precision, Recall, F1, and F2 change with threshold.
    
    This visualization helps understand the trade-off between
    catching frauds and avoiding false alarms.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Prediction probabilities
    save_path : str or None
        Path to save the figure
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    
    precisions = []
    recalls = []
    f1_scores = []
    f2_scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        precisions.append(precision_score(y_true, y_pred))
        recalls.append(recall_score(y_true, y_pred))
        f1_scores.append(f1_score(y_true, y_pred))
        f2_scores.append(fbeta_score(y_true, y_pred, beta=2))
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(thresholds, precisions, label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, label='Recall', linewidth=2)
    plt.plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
    plt.plot(thresholds, f2_scores, label='F2-Score', linewidth=2, linestyle='--')
    
    # Mark default threshold
    plt.axvline(x=0.5, color='red', linestyle=':', alpha=0.7, label='Default (0.5)')
    
    # Mark optimal F2 threshold
    optimal_f2_idx = np.argmax(f2_scores)
    optimal_f2_threshold = thresholds[optimal_f2_idx]
    plt.axvline(x=optimal_f2_threshold, color='green', linestyle=':', alpha=0.7,
                label=f'Optimal F2 ({optimal_f2_threshold:.2f})')
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Threshold Analysis: Impact on Metrics', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Threshold analysis plot saved to {save_path}")
    
    plt.close()


def get_cost_sensitive_threshold(y_true, y_proba, cost_fn=10, cost_fp=1):
    """
    Find optimal threshold based on cost of errors.
    
    In fraud detection:
    - False Negative (missing fraud): High cost (money lost)
    - False Positive (false alarm): Low cost (customer inconvenience)
    
    Typical ratio: cost_fn / cost_fp = 10 to 100
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Prediction probabilities
    cost_fn : float
        Cost of False Negative (missing fraud)
    cost_fp : float
        Cost of False Positive (false alarm)
        
    Returns:
    --------
    optimal_threshold : float
        Threshold minimizing total cost
    min_cost : float
        Minimum total cost
    default_cost : float
        Total cost at default threshold (0.5)
    """
    print("\n" + "="*60)
    print("Cost-Sensitive Threshold Optimization")
    print("="*60)
    print(f"Cost of False Negative (missing fraud): {cost_fn}")
    print(f"Cost of False Positive (false alarm): {cost_fp}")
    print(f"Cost ratio (FN/FP): {cost_fn/cost_fp:.1f}")
    
    thresholds = np.arange(0.05, 0.95, 0.01)
    costs = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # Count errors
        fn = np.sum((y_true == 1) & (y_pred == 0))  # Missed frauds
        fp = np.sum((y_true == 0) & (y_pred == 1))  # False alarms
        
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        costs.append(total_cost)
    
    best_idx = np.argmin(costs)
    optimal_threshold = thresholds[best_idx]
    min_cost = costs[best_idx]
    
    # Cost at default threshold
    y_pred_default = (y_proba >= 0.5).astype(int)
    fn_default = np.sum((y_true == 1) & (y_pred_default == 0))
    fp_default = np.sum((y_true == 0) & (y_pred_default == 1))
    cost_default = (fn_default * cost_fn) + (fp_default * cost_fp)
    
    print(f"\nDefault threshold (0.5):")
    print(f"  Total cost: {cost_default:.2f}")
    
    print(f"\nOptimal threshold ({optimal_threshold:.2f}):")
    print(f"  Total cost: {min_cost:.2f}")
    print(f"  Cost reduction: {((cost_default - min_cost) / cost_default * 100):.2f}%")
    
    return optimal_threshold, min_cost, cost_default


if __name__ == "__main__":
    print("Threshold tuning module loaded successfully.")
    print("\nAvailable methods:")
    print("1. F2-Score optimization (favors recall)")
    print("2. Youden's J statistic (balances sensitivity/specificity)")
    print("3. Target recall with maximum precision")
    print("4. Cost-sensitive optimization")
    print("5. Comprehensive grid search")
