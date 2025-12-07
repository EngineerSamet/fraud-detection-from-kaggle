"""
SHAP Analysis Module for Credit Card Fraud Detection

SHAP (SHapley Additive exPlanations) provides model interpretability
by showing:
- Which features are most important globally
- How each feature contributes to individual predictions
- Feature interactions and dependencies

CRITICAL NOTE: 
Since this dataset uses PCA-transformed features (V1-V28),
SHAP can only explain importance in the PCA space, not in the
original transaction feature space. This is a fundamental limitation
of anonymized datasets.

This module works with tree-based models (Random Forest, XGBoost).
"""

import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def explain_model_shap(model, X_data, model_type='tree', max_display=20):
    """
    Create SHAP explainer for the model.
    
    Parameters:
    -----------
    model : trained model
        Random Forest or XGBoost model
    X_data : pd.DataFrame or np.array
        Data to explain (typically test set or sample)
    model_type : str
        'tree' for tree-based models, 'linear' for linear models
    max_display : int
        Maximum number of samples to use for explanation
        
    Returns:
    --------
    explainer : shap.Explainer
        SHAP explainer object
    shap_values : np.array
        SHAP values for each feature
    """
    print("\n" + "="*60)
    print("Creating SHAP Explainer")
    print("="*60)
    
    # Use subset if dataset is large (SHAP can be slow)
    if len(X_data) > max_display:
        print(f"Using {max_display} samples for SHAP analysis (dataset is large)")
        indices = np.random.choice(len(X_data), max_display, replace=False)
        X_sample = X_data.iloc[indices] if hasattr(X_data, 'iloc') else X_data[indices]
    else:
        X_sample = X_data
    
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model)
    
    print("Computing SHAP values...")
    shap_values = explainer.shap_values(X_sample)
    
    # For binary classification, XGBoost returns shap_values for class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Focus on fraud class
    
    print(f"✓ SHAP values computed for {len(X_sample)} samples")
    
    return explainer, shap_values, X_sample


def plot_shap_summary(shap_values, X_data, feature_names=None, save_path=None):
    """
    Create SHAP summary plot showing feature importance.
    
    This plot shows:
    - Which features are most important
    - Distribution of impact for each feature
    - Whether high/low feature values increase or decrease predictions
    
    Parameters:
    -----------
    shap_values : np.array
        SHAP values from explainer
    X_data : pd.DataFrame or np.array
        Feature data
    feature_names : list or None
        Names of features
    save_path : str or None
        Path to save the figure
    """
    print("\n" + "="*60)
    print("Creating SHAP Summary Plot")
    print("="*60)
    
    plt.figure(figsize=(10, 8))
    
    if feature_names is not None and hasattr(X_data, 'columns'):
        shap.summary_plot(shap_values, X_data, feature_names=feature_names, 
                         show=False, plot_type='dot')
    else:
        shap.summary_plot(shap_values, X_data, show=False, plot_type='dot')
    
    plt.title('SHAP Summary Plot - Feature Importance', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ SHAP summary plot saved to {save_path}")
    
    plt.close()


def plot_shap_bar(shap_values, X_data, feature_names=None, max_display=20, save_path=None):
    """
    Create SHAP bar plot showing mean absolute SHAP values.
    
    This is a simpler view of feature importance compared to summary plot.
    
    Parameters:
    -----------
    shap_values : np.array
        SHAP values from explainer
    X_data : pd.DataFrame or np.array
        Feature data
    feature_names : list or None
        Names of features
    max_display : int
        Number of top features to display
    save_path : str or None
        Path to save the figure
    """
    print("\n" + "="*60)
    print(f"Creating SHAP Bar Plot (Top {max_display} Features)")
    print("="*60)
    
    plt.figure(figsize=(10, 8))
    
    if feature_names is not None and hasattr(X_data, 'columns'):
        shap.summary_plot(shap_values, X_data, feature_names=feature_names,
                         plot_type='bar', max_display=max_display, show=False)
    else:
        shap.summary_plot(shap_values, X_data, plot_type='bar', 
                         max_display=max_display, show=False)
    
    plt.title('SHAP Bar Plot - Average Feature Impact', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ SHAP bar plot saved to {save_path}")
    
    plt.close()


def plot_shap_waterfall(explainer, shap_values, X_data, sample_index=0, save_path=None):
    """
    Create SHAP waterfall plot for a single prediction.
    
    This shows how each feature contributed to moving the prediction
    from the base value to the final prediction for one specific sample.
    
    Parameters:
    -----------
    explainer : shap.Explainer
        SHAP explainer object
    shap_values : np.array
        SHAP values from explainer
    X_data : pd.DataFrame or np.array
        Feature data
    sample_index : int
        Index of sample to explain
    save_path : str or None
        Path to save the figure
    """
    print("\n" + "="*60)
    print(f"Creating SHAP Waterfall Plot for Sample {sample_index}")
    print("="*60)
    
    plt.figure(figsize=(10, 8))
    
    # Create explanation object for single sample
    if hasattr(explainer, 'expected_value'):
        expected_value = explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[1]  # For binary classification
    else:
        expected_value = 0
    
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[sample_index],
            base_values=expected_value,
            data=X_data.iloc[sample_index] if hasattr(X_data, 'iloc') else X_data[sample_index]
        ),
        show=False
    )
    
    plt.title(f'SHAP Waterfall - Sample {sample_index}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ SHAP waterfall plot saved to {save_path}")
    
    plt.close()


def plot_shap_dependence(shap_values, X_data, feature_name, interaction_feature=None, save_path=None):
    """
    Create SHAP dependence plot for a specific feature.
    
    This shows how SHAP values for one feature vary with the feature's value,
    and optionally colored by another feature to show interactions.
    
    Parameters:
    -----------
    shap_values : np.array
        SHAP values from explainer
    X_data : pd.DataFrame or np.array
        Feature data
    feature_name : str or int
        Feature to analyze
    interaction_feature : str, int, or None
        Feature to use for coloring (auto-selected if None)
    save_path : str or None
        Path to save the figure
    """
    print("\n" + "="*60)
    print(f"Creating SHAP Dependence Plot for {feature_name}")
    print("="*60)
    
    plt.figure(figsize=(10, 6))
    
    shap.dependence_plot(
        feature_name,
        shap_values,
        X_data,
        interaction_index=interaction_feature,
        show=False
    )
    
    plt.title(f'SHAP Dependence Plot - {feature_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ SHAP dependence plot saved to {save_path}")
    
    plt.close()


def get_top_shap_features(shap_values, feature_names=None, top_n=10):
    """
    Get top N most important features by mean absolute SHAP value.
    
    Parameters:
    -----------
    shap_values : np.array
        SHAP values from explainer
    feature_names : list or None
        Names of features
    top_n : int
        Number of top features to return
        
    Returns:
    --------
    top_features_df : pd.DataFrame
        DataFrame with top features and their importance
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(mean_abs_shap))]
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False).head(top_n)
    
    print("\n" + "="*60)
    print(f"Top {top_n} Features by Mean Absolute SHAP Value")
    print("="*60)
    print(importance_df.to_string(index=False))
    
    return importance_df


def plot_shap_force(explainer, shap_values, X_data, sample_index=0, save_path=None):
    """
    Create SHAP force plot for a single prediction.
    
    Shows additive forces pushing prediction from base value to final output.
    More intuitive than waterfall for presentations.
    
    Parameters:
    -----------
    explainer : shap.Explainer
        SHAP explainer object
    shap_values : np.array
        SHAP values
    X_data : pd.DataFrame
        Feature data
    sample_index : int
        Sample to explain
    save_path : str or None
        Path to save (as HTML)
    """
    print(f"\nCreating SHAP force plot for sample {sample_index}...")
    
    # Create force plot
    shap.force_plot(
        explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        shap_values[sample_index],
        X_data.iloc[sample_index] if hasattr(X_data, 'iloc') else X_data[sample_index],
        matplotlib=True,
        show=False
    )
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ SHAP force plot saved to {save_path}")
    
    plt.close()


def plot_shap_interaction(shap_values, X_data, feature1, feature2=None, save_path=None):
    """
    Plot SHAP interaction (dependence plot with interactions).
    
    Shows how two features interact to influence predictions.
    Critical for understanding feature relationships in fraud patterns.
    
    Parameters:
    -----------
    shap_values : np.array
        SHAP values
    X_data : pd.DataFrame
        Feature data
    feature1 : str or int
        Primary feature
    feature2 : str or int or None
        Interaction feature (auto-detected if None)
    save_path : str or None
        Path to save
    """
    print(f"\nCreating SHAP interaction plot for {feature1}...")
    
    plt.figure(figsize=(10, 6))
    
    shap.dependence_plot(
        feature1,
        shap_values,
        X_data,
        interaction_index=feature2,
        show=False
    )
    
    plt.title(f'SHAP Interaction: {feature1}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ SHAP interaction plot saved to {save_path}")
    
    plt.close()


def create_all_shap_plots(model, X_test, model_name="Model", save_dir=None, max_samples=100):
    """
    Create comprehensive SHAP analysis with all plots.
    
    Parameters:
    -----------
    model : trained model
        Tree-based model (Random Forest or XGBoost)
    X_test : pd.DataFrame
        Test data
    model_name : str
        Name of the model
    save_dir : str or None
        Directory to save plots
    max_samples : int
        Maximum samples for SHAP computation
        
    Returns:
    --------
    shap_values : np.array
        Computed SHAP values
    """
    import os
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Create explainer and compute SHAP values
    explainer, shap_values, X_sample = explain_model_shap(
        model, X_test, model_type='tree', max_display=max_samples
    )
    
    feature_names = X_test.columns.tolist() if hasattr(X_test, 'columns') else None
    
    # Summary plot
    summary_path = os.path.join(save_dir, f"{model_name}_shap_summary.png") if save_dir else None
    plot_shap_summary(shap_values, X_sample, feature_names, summary_path)
    
    # Bar plot
    bar_path = os.path.join(save_dir, f"{model_name}_shap_bar.png") if save_dir else None
    plot_shap_bar(shap_values, X_sample, feature_names, max_display=15, save_path=bar_path)
    
    # Get top features
    top_features = get_top_shap_features(shap_values, feature_names, top_n=10)
    
    print("\n✓ SHAP analysis completed successfully")
    print(f"  Plots saved to: {save_dir if save_dir else 'Not saved'}")
    
    return explainer, shap_values, top_features, X_sample


if __name__ == "__main__":
    print("SHAP analysis module loaded successfully.")
    print("\nSHAP provides model interpretability through:")
    print("- Summary plots (global feature importance)")
    print("- Bar plots (mean absolute impact)")
    print("- Waterfall plots (individual prediction explanation)")
    print("- Dependence plots (feature interactions)")
    print("\n⚠️  Note: PCA-transformed features limit real-world interpretability")
