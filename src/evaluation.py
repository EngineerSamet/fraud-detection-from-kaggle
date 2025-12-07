"""
Evaluation Module for Credit Card Fraud Detection

This module provides comprehensive evaluation metrics suitable for 
imbalanced classification tasks. 

CRITICAL: Accuracy is NOT a good metric for fraud detection!
With 0.17% fraud rate, a model that always predicts "normal" 
would have 99.83% accuracy but catch zero frauds.

We use:
- Precision: Of all predicted frauds, how many are real?
- Recall: Of all real frauds, how many did we catch?
- F1-Score: Harmonic mean of precision and recall
- F2-Score: Weighted F-score favoring recall (critical for fraud)
- ROC-AUC: Area under ROC curve
- PR-AUC: Area under Precision-Recall curve (BETTER for imbalanced data)
- MCC: Matthews Correlation Coefficient (robust for imbalanced data)
- Cohen's Kappa: Agreement measure
- Balanced Accuracy: Average of recall for each class
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(y_true, y_pred, y_proba=None, model_name="Model"):
    """
    Comprehensive evaluation of binary classification model.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_proba : array-like or None
        Prediction probabilities (for AUC metrics)
    model_name : str
        Name of the model for display
        
    Returns:
    --------
    metrics_dict : dict
        Dictionary containing all metrics
    """
    print("\n" + "="*60)
    print(f"Evaluation: {model_name}")
    print("="*60)
    
    # Basic metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    f2 = fbeta_score(y_true, y_pred, beta=2)  # Favors recall
    
    # Robust metrics for imbalanced data
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    metrics_dict = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'f2_score': f2,
        'mcc': mcc,
        'cohen_kappa': kappa,
        'balanced_accuracy': balanced_acc
    }
    
    # AUC metrics (require probabilities)
    if y_proba is not None:
        roc_auc = roc_auc_score(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        metrics_dict['roc_auc'] = roc_auc
        metrics_dict['pr_auc'] = pr_auc
    
    # Print metrics
    print(f"\n{'Metric':<25} {'Value':<10}")
    print("-" * 35)
    print(f"{'Precision':<25} {precision:.4f}")
    print(f"{'Recall (Sensitivity)':<25} {recall:.4f}")
    print(f"{'F1-Score':<25} {f1:.4f}")
    print(f"{'F2-Score':<25} {f2:.4f}")
    print(f"{'Balanced Accuracy':<25} {balanced_acc:.4f}")
    print(f"{'MCC':<25} {mcc:.4f}")
    kappa_label = "Cohen's Kappa"
    print(f"{kappa_label:<25} {kappa:.4f}")
    
    if y_proba is not None:
        print(f"{'ROC-AUC':<25} {roc_auc:.4f}")
        print(f"{'PR-AUC':<25} {pr_auc:.4f}")
    
    return metrics_dict


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model
    save_path : str or None
        Path to save the figure
        
    Returns:
    --------
    cm : array
        Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add counts
    tn, fp, fn, tp = cm.ravel()
    plt.text(0.5, -0.15, f'TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}',
             ha='center', transform=plt.gca().transAxes, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    
    plt.close()
    
    return cm


def plot_roc_curve(y_true, y_proba, model_name="Model", save_path=None):
    """
    Plot ROC curve.
    
    NOTE: ROC curve can be misleading for highly imbalanced datasets.
    PR curve is more informative for fraud detection.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Prediction probabilities
    model_name : str
        Name of the model
    save_path : str or None
        Path to save the figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to {save_path}")
    
    plt.close()


def plot_precision_recall_curve(y_true, y_proba, model_name="Model", save_path=None):
    """
    Plot Precision-Recall curve.
    
    CRITICAL: PR curve is MORE INFORMATIVE than ROC for imbalanced data.
    It focuses on the minority class (fraud) performance.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Prediction probabilities
    model_name : str
        Name of the model
    save_path : str or None
        Path to save the figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkgreen', lw=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')
    
    # Baseline (random classifier)
    no_skill = len(y_true[y_true==1]) / len(y_true)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='navy',
             label=f'No Skill (baseline = {no_skill:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ PR curve saved to {save_path}")
    
    plt.close()


def plot_all_curves(y_true, y_proba, model_name="Model", save_dir=None):
    """
    Plot both ROC and PR curves in a single figure.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Prediction probabilities
    model_name : str
        Name of the model
    save_dir : str or None
        Directory to save figures
    """
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        roc_path = os.path.join(save_dir, f"{model_name}_roc_curve.png")
        pr_path = os.path.join(save_dir, f"{model_name}_pr_curve.png")
        cm_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    else:
        roc_path = pr_path = cm_path = None
    
    plot_roc_curve(y_true, y_proba, model_name, roc_path)
    plot_precision_recall_curve(y_true, y_proba, model_name, pr_path)


def print_classification_report(y_true, y_pred, target_names=None):
    """
    Print detailed classification report.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    target_names : list or None
        Names for classes
    """
    if target_names is None:
        target_names = ['Normal', 'Fraud']
    
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=target_names))


def compare_models(results_dict, metric='pr_auc'):
    """
    Compare multiple models based on a specific metric.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary of {model_name: metrics_dict}
    metric : str
        Metric to compare (default: 'pr_auc')
        
    Returns:
    --------
    comparison_df : pd.DataFrame
        Comparison table
    """
    comparison_data = []
    
    for model_name, metrics in results_dict.items():
        row = {'model': model_name}
        row.update(metrics)
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values(metric, ascending=False)
    
    print("\n" + "="*60)
    print(f"Model Comparison (sorted by {metric})")
    print("="*60)
    print(comparison_df.to_string(index=False))
    
    return comparison_df


if __name__ == "__main__":
    print("Evaluation module loaded successfully.")
    print("\nAvailable metrics:")
    print("- Precision, Recall, F1-Score, F2-Score")
    print("- ROC-AUC, PR-AUC (Precision-Recall AUC)")
    print("- MCC (Matthews Correlation Coefficient)")
    print("- Cohen's Kappa")
    print("- Balanced Accuracy")
    print("\n⚠️  For fraud detection, prioritize: PR-AUC, F2-Score, Recall")
