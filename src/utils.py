"""
Utility Functions for Credit Card Fraud Detection

This module provides helper functions for:
- Saving and loading models
- Logging results
- Plotting utilities
- Configuration management
"""

import os
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def save_model(model, filepath, model_name="model"):
    """
    Save trained model to disk.
    
    Parameters:
    -----------
    model : trained model
        Scikit-learn or XGBoost model
    filepath : str
        Path to save the model
    model_name : str
        Name of the model for logging
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✓ {model_name} saved to {filepath}")


def load_model(filepath):
    """
    Load trained model from disk.
    
    Parameters:
    -----------
    filepath : str
        Path to the saved model
        
    Returns:
    --------
    model : loaded model
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✓ Model loaded from {filepath}")
    return model


def save_results(results_dict, filepath):
    """
    Save results dictionary to JSON file.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing results
    filepath : str
        Path to save the JSON file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    results_dict = convert_types(results_dict)
    
    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    print(f"✓ Results saved to {filepath}")


def load_results(filepath):
    """
    Load results from JSON file.
    
    Parameters:
    -----------
    filepath : str
        Path to the JSON file
        
    Returns:
    --------
    results_dict : dict
        Loaded results
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        results_dict = json.load(f)
    
    print(f"✓ Results loaded from {filepath}")
    return results_dict


def create_experiment_folder(base_dir="outputs", experiment_name=None):
    """
    Create timestamped folder for experiment outputs.
    
    Parameters:
    -----------
    base_dir : str
        Base directory for experiments
    experiment_name : str or None
        Name of experiment. If None, uses timestamp.
        
    Returns:
    --------
    experiment_dir : str
        Path to created experiment directory
    """
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    experiment_dir = os.path.join(base_dir, experiment_name)
    
    # Create subdirectories
    os.makedirs(os.path.join(experiment_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "results"), exist_ok=True)
    
    print(f"✓ Experiment folder created: {experiment_dir}")
    
    return experiment_dir


def plot_class_distribution(y, title="Class Distribution", save_path=None):
    """
    Plot class distribution bar chart.
    
    Parameters:
    -----------
    y : array-like
        Target labels
    title : str
        Plot title
    save_path : str or None
        Path to save the figure
    """
    import numpy as np
    
    unique, counts = np.unique(y, return_counts=True)
    percentages = counts / len(y) * 100
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Normal (0)', 'Fraud (1)'], counts, color=['#2ecc71', '#e74c3c'])
    
    # Add count labels on bars
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({pct:.2f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.ylabel('Count', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Class distribution plot saved to {save_path}")
    
    plt.close()


def plot_feature_distributions(df, features, save_path=None, fraud_only=False):
    """
    Plot distribution of multiple features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing features and 'Class' column
    features : list
        List of feature names to plot
    save_path : str or None
        Path to save the figure
    fraud_only : bool
        If True, plot fraud and normal separately
    """
    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        if fraud_only and 'Class' in df.columns:
            df[df['Class'] == 0][feature].hist(ax=ax, bins=50, alpha=0.7, 
                                                 label='Normal', color='blue')
            df[df['Class'] == 1][feature].hist(ax=ax, bins=50, alpha=0.7, 
                                                 label='Fraud', color='red')
            ax.legend()
        else:
            df[feature].hist(ax=ax, bins=50, color='steelblue')
        
        ax.set_title(f'Distribution of {feature}', fontweight='bold')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Feature distributions plot saved to {save_path}")
    
    plt.close()


def plot_fraud_vs_normal_distributions(df, top_features=10, save_path=None):
    """
    Plot fraud vs normal transaction distributions for top features.
    
    This is THE MOST CRITICAL VISUALIZATION in fraud detection:
    - Shows which features separate fraud from normal
    - Reveals pattern differences between classes
    - Essential for data understanding (not just black-box modeling)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with 'Class' column (0=normal, 1=fraud)
    top_features : int
        Number of top correlated features to plot (default: 10)
    save_path : str or None
        Path to save the figure
    """
    # Get top features by correlation with fraud
    if 'Class' in df.columns:
        corr_with_fraud = df.corr()['Class'].abs().sort_values(ascending=False)
        # Exclude 'Class' itself
        top_feature_names = corr_with_fraud.index[1:top_features+1].tolist()
    else:
        top_feature_names = df.columns[:top_features].tolist()
    
    n_features = len(top_feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    fraud_data = df[df['Class'] == 1]
    normal_data = df[df['Class'] == 0]
    
    for idx, feature in enumerate(top_feature_names):
        ax = axes[idx]
        
        # Plot overlapping histograms
        ax.hist(normal_data[feature], bins=50, alpha=0.6, 
                label=f'Normal ({len(normal_data)})', color='blue', density=True)
        ax.hist(fraud_data[feature], bins=50, alpha=0.7, 
                label=f'Fraud ({len(fraud_data)})', color='red', density=True)
        
        ax.set_title(f'{feature} Distribution', fontweight='bold', fontsize=11)
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Fraud vs Normal: Feature Distribution Comparison\n(Top {} Most Correlated Features)'.format(top_features),
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Fraud vs Normal distribution overlay saved to {save_path}")
    
    plt.close()


def plot_rf_feature_importance(model, feature_names, top_n=15, save_path=None):
    """
    Plot Random Forest feature importance bar chart.
    
    Parameters:
    -----------
    model : RandomForestClassifier
        Trained Random Forest model
    feature_names : list
        Names of features
    top_n : int
        Number of top features to display
    save_path : str or None
        Path to save the figure
    """
    import numpy as np
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': [feature_names[i] for i in indices],
        'importance': importances[indices]
    })
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importance_df)), importance_df['importance'], color='steelblue')
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Feature Importance (Gini)', fontsize=12)
    plt.title(f'Random Forest Feature Importance (Top {top_n})', 
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ RF feature importance plot saved to {save_path}")
    
    plt.close()
    
    return importance_df


def plot_pca_explained_variance(X, n_components=30, save_path=None):
    """
    Plot PCA explained variance ratio (cumulative).
    
    Critical for PCA datasets to understand:
    - How much information each component captures
    - How many components are needed for 95% variance
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Feature data (V1-V28 are already PCA components)
    n_components : int
        Number of components (dataset has V1-V28 = 28 PCA features)
    save_path : str or None
        Path to save the figure
    """
    from sklearn.decomposition import PCA
    import numpy as np
    
    # Note: Dataset is already PCA-transformed
    # We're analyzing the existing PCA structure
    pca_features = [col for col in X.columns if col.startswith('V')]
    
    if len(pca_features) == 0:
        print("⚠️  No PCA features (V1-V28) found in dataset")
        return
    
    # Re-run PCA on PCA features to get explained variance
    pca = PCA(n_components=min(len(pca_features), n_components))
    pca.fit(X[pca_features])
    
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Individual variance
    ax1.bar(range(1, len(explained_var)+1), explained_var, color='steelblue', alpha=0.7)
    ax1.set_xlabel('PCA Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title('Individual PCA Component Variance', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Cumulative variance
    ax2.plot(range(1, len(cumulative_var)+1), cumulative_var, marker='o', 
             linewidth=2, color='darkblue')
    ax2.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% Threshold')
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax2.set_title('Cumulative PCA Variance', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.suptitle('PCA Explained Variance Analysis\n(Dataset uses V1-V28 PCA-transformed features)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ PCA explained variance plot saved to {save_path}")
    
    plt.close()
    
    # Print summary
    n_95 = np.argmax(cumulative_var >= 0.95) + 1
    print(f"\n📊 PCA Summary:")
    print(f"   Components for 95% variance: {n_95}/{len(explained_var)}")
    print(f"   Total explained variance: {cumulative_var[-1]:.4f}")


def plot_calibration_curve(y_true, y_proba, model_name="Model", n_bins=10, save_path=None):
    """
    Plot probability calibration curve.
    
    CRITICAL for fraud detection:
    - Banks require calibrated probabilities for risk assessment
    - Uncalibrated models give misleading confidence scores
    - This plot shows if predicted probabilities match reality
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Predicted probabilities for positive class
    model_name : str
        Name of the model
    n_bins : int
        Number of bins for calibration curve
    save_path : str or None
        Path to save the figure
    """
    from sklearn.calibration import calibration_curve
    
    # Compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy='uniform'
    )
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Calibration curve
    ax1.plot(mean_predicted_value, fraction_of_positives, marker='o', 
             linewidth=2, label=model_name, color='darkblue')
    ax1.plot([0, 1], [0, 1], linestyle='--', color='red', linewidth=2, 
             label='Perfectly Calibrated')
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives (True Fraud Rate)', fontsize=12)
    ax1.set_title('Calibration Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)
    
    # Probability distribution
    ax2.hist(y_proba[y_true == 0], bins=50, alpha=0.6, label='Normal', 
             color='blue', density=True)
    ax2.hist(y_proba[y_true == 1], bins=50, alpha=0.7, label='Fraud', 
             color='red', density=True)
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Predicted Probability Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.suptitle(f'{model_name}: Probability Calibration Analysis\n(Required for banking risk assessment)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Calibration curve saved to {save_path}")
    
    plt.close()


def calculate_brier_score(y_true, y_proba):
    """
    Calculate Brier score (mean squared error of probability predictions).
    
    Lower is better (0 = perfect, 1 = worst).
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_proba : array-like
        Predicted probabilities for positive class
        
    Returns:
    --------
    float
        Brier score
    """
    from sklearn.metrics import brier_score_loss
    return brier_score_loss(y_true, y_proba)


def plot_calibration_comparison(y_true, y_probas_list, model_names, brier_scores=None, 
                                  n_bins=10, save_path=None):
    """
    Compare calibration curves for multiple models (e.g., before/after calibration).
    
    CRITICAL for showing calibration improvement.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_probas_list : list of array-like
        List of predicted probabilities for each model
    model_names : list of str
        List of model names (e.g., ['Original', 'Isotonic', 'Sigmoid'])
    brier_scores : list of float or None
        List of Brier scores for each model
    n_bins : int
        Number of bins for calibration curve
    save_path : str or None
        Path to save the figure
    """
    from sklearn.calibration import calibration_curve
    import numpy as np
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['darkblue', 'green', 'orange', 'purple', 'brown']
    
    # Plot calibration curves
    for idx, (y_proba, name) in enumerate(zip(y_probas_list, model_names)):
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=n_bins, strategy='uniform'
        )
        
        label = name
        if brier_scores and len(brier_scores) > idx:
            label += f" (Brier: {brier_scores[idx]:.4f})"
        
        ax1.plot(mean_predicted_value, fraction_of_positives, marker='o',
                linewidth=2, label=label, color=colors[idx % len(colors)])
    
    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], linestyle='--', color='red', linewidth=2,
            label='Perfectly Calibrated')
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title('Calibration Curve Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Probability distributions comparison
    for idx, (y_proba, name) in enumerate(zip(y_probas_list, model_names)):
        ax2.hist(y_proba, bins=50, alpha=0.5, label=name,
                color=colors[idx % len(colors)], density=True)
    
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Probability Distribution Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.suptitle('Calibration Analysis: Before vs After',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Calibration comparison saved to {save_path}")
    
    plt.close()


def plot_correlation_matrix(df, save_path=None, top_n=None):
    """
    Plot correlation matrix heatmap.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with features
    save_path : str or None
        Path to save the figure
    top_n : int or None
        If specified, only plot top N most correlated features
    """
    corr = df.corr()
    
    if top_n is not None:
        # Get top N features correlated with target
        if 'Class' in corr.columns:
            top_features = corr['Class'].abs().sort_values(ascending=False).head(top_n).index
            corr = corr.loc[top_features, top_features]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Correlation matrix saved to {save_path}")
    
    plt.close()


def plot_correlation_comparison(df_imbalanced, df_balanced, save_path=None, top_n=15):
    """
    Plot side-by-side correlation heatmaps for imbalanced vs balanced data.
    
    This visualization demonstrates why balancing is necessary:
    - Imbalanced data correlations are misleading (dominated by majority class)
    - Balanced data reveals true fraud patterns
    
    Parameters:
    -----------
    df_imbalanced : pd.DataFrame
        Original imbalanced dataset (with 'Class' column)
    df_balanced : pd.DataFrame
        Balanced dataset after class weight application (with 'Class' column)
    save_path : str or None
        Path to save the figure
    top_n : int
        Number of top correlated features to display (default: 15)
    """
    # Calculate correlations
    corr_imbalanced = df_imbalanced.corr()
    corr_balanced = df_balanced.corr()
    
    # Get top N features correlated with Class in balanced data
    if 'Class' in corr_balanced.columns:
        top_features = corr_balanced['Class'].abs().sort_values(ascending=False).head(top_n).index.tolist()
        corr_imbalanced = corr_imbalanced.loc[top_features, top_features]
        corr_balanced = corr_balanced.loc[top_features, top_features]
    
    # Create side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Imbalanced heatmap
    sns.heatmap(corr_imbalanced, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=axes[0])
    axes[0].set_title('Correlation Matrix - Imbalanced Data\n(Misleading: 99.83% Normal, 0.17% Fraud)',
                      fontsize=13, fontweight='bold')
    
    # Balanced heatmap
    sns.heatmap(corr_balanced, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=axes[1])
    axes[1].set_title('Correlation Matrix - Balanced Data (Class Weights)\n(True fraud patterns revealed)',
                      fontsize=13, fontweight='bold', color='green')
    
    plt.suptitle('Why Data Balancing Matters: Correlation Analysis', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Correlation comparison plot saved to {save_path}")
    
    plt.close()


def plot_tsne_clustering(X, y, save_path=None, sample_size=5000, random_state=42):
    """
    Create t-SNE clustering visualization to show fraud vs normal separation.
    
    t-SNE reduces high-dimensional data to 2D while preserving local structure.
    This visualization demonstrates whether fraud cases form distinct clusters.
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Features (high-dimensional)
    y : pd.Series or np.ndarray
        Target labels (0=normal, 1=fraud)
    save_path : str or None
        Path to save the figure
    sample_size : int
        Number of samples to use (t-SNE is slow, default: 5000)
    random_state : int
        Random seed for reproducibility
    """
    from sklearn.manifold import TSNE
    import numpy as np
    
    print(f"\nGenerating t-SNE clustering visualization...")
    print(f"  Original data size: {len(X)} samples")
    
    # Sample data if too large (t-SNE is computationally expensive)
    if len(X) > sample_size:
        np.random.seed(random_state)
        
        # Stratified sampling to maintain fraud ratio
        fraud_indices = np.where(y == 1)[0]
        normal_indices = np.where(y == 0)[0]
        
        # Sample fraud cases (take all if less than 50% of sample_size)
        n_fraud = min(len(fraud_indices), sample_size // 2)
        n_normal = sample_size - n_fraud
        
        sampled_fraud = np.random.choice(fraud_indices, size=n_fraud, replace=False)
        sampled_normal = np.random.choice(normal_indices, size=n_normal, replace=False)
        
        sampled_indices = np.concatenate([sampled_fraud, sampled_normal])
        np.random.shuffle(sampled_indices)
        
        X_sample = X.iloc[sampled_indices] if hasattr(X, 'iloc') else X[sampled_indices]
        y_sample = y.iloc[sampled_indices] if hasattr(y, 'iloc') else y[sampled_indices]
        
        print(f"  Sampled: {len(X_sample)} samples ({n_fraud} fraud, {n_normal} normal)")
    else:
        X_sample = X
        y_sample = y
        print(f"  Using all {len(X)} samples")
    
    # Apply t-SNE
    print("  Running t-SNE (this may take 1-2 minutes)...")
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X_sample)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot normal transactions
    normal_mask = y_sample == 0
    plt.scatter(X_tsne[normal_mask, 0], X_tsne[normal_mask, 1], 
                c='blue', alpha=0.3, s=10, label=f'Normal ({normal_mask.sum()})')
    
    # Plot fraud transactions
    fraud_mask = y_sample == 1
    plt.scatter(X_tsne[fraud_mask, 0], X_tsne[fraud_mask, 1], 
                c='red', alpha=0.8, s=30, label=f'Fraud ({fraud_mask.sum()})', 
                edgecolors='black', linewidths=0.5)
    
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.title('t-SNE Clustering: Fraud vs Normal Transactions\n(2D projection of 30-dimensional feature space)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ t-SNE clustering plot saved to {save_path}")
    
    plt.close()
    print("✓ t-SNE visualization completed")


def create_comparison_table(results_dict, metrics=None):
    """
    Create comparison table for multiple models.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary of {model_name: metrics_dict}
    metrics : list or None
        List of metrics to include. If None, includes all.
        
    Returns:
    --------
    comparison_df : pd.DataFrame
        Comparison table
    """
    comparison_data = []
    
    for model_name, model_metrics in results_dict.items():
        row = {'Model': model_name}
        
        if metrics is None:
            row.update(model_metrics)
        else:
            for metric in metrics:
                row[metric] = model_metrics.get(metric, None)
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    return comparison_df


def print_section_header(title):
    """
    Print formatted section header.
    
    Parameters:
    -----------
    title : str
        Section title
    """
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def setup_plot_style():
    """
    Set up consistent plot style for all visualizations.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


if __name__ == "__main__":
    print("Utility module loaded successfully.")
    print("\nAvailable utilities:")
    print("- Model save/load")
    print("- Results save/load")
    print("- Experiment folder creation")
    print("- Plotting utilities")
    print("- Comparison tables")
