"""
Modeling Module for Credit Card Fraud Detection

This module provides training functions for:
- Logistic Regression (baseline linear model)
- Random Forest (non-linear ensemble)
- XGBoost (state-of-the-art gradient boosting)

Each model supports class weighting for imbalanced data.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np
import time


def train_logistic_regression(X_train, y_train, class_weight=None, max_iter=1000, random_state=42):
    """
    Train Logistic Regression model.
    
    Logistic Regression is the baseline linear model. It's:
    - Fast to train
    - Interpretable
    - Requires feature scaling
    - Good for linearly separable problems
    
    For fraud detection, it may underperform compared to non-linear models.
    
    Parameters:
    -----------
    X_train : array-like
        Training features (should be scaled)
    y_train : array-like
        Training labels
    class_weight : dict or 'balanced' or None
        Class weights to handle imbalance
    max_iter : int
        Maximum iterations for convergence
    random_state : int
        Random seed
        
    Returns:
    --------
    model : trained LogisticRegression model
    training_time : float
    """
    print("\n" + "="*60)
    print("Training Logistic Regression")
    print("="*60)
    
    start_time = time.time()
    
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        solver='lbfgs',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    print(f"✓ Training completed in {training_time:.2f} seconds")
    print(f"  Class weight: {class_weight}")
    print(f"  Number of iterations: {model.n_iter_[0]}")
    
    return model, training_time


def train_random_forest(X_train, y_train, class_weight=None, n_estimators=100, 
                        max_depth=None, random_state=42):
    """
    Train Random Forest Classifier.
    
    Random Forest is a strong non-linear ensemble model. It's:
    - Robust to outliers
    - Handles non-linear relationships well
    - Provides feature importance
    - Does NOT require feature scaling
    - Less prone to overfitting than single decision trees
    
    Parameters:
    -----------
    X_train : array-like
        Training features (scaling not required)
    y_train : array-like
        Training labels
    class_weight : dict or 'balanced' or None
        Class weights to handle imbalance
    n_estimators : int
        Number of trees in the forest
    max_depth : int or None
        Maximum depth of trees
    random_state : int
        Random seed
        
    Returns:
    --------
    model : trained RandomForestClassifier model
    training_time : float
    """
    print("\n" + "="*60)
    print("Training Random Forest")
    print("="*60)
    
    start_time = time.time()
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    print(f"✓ Training completed in {training_time:.2f} seconds")
    print(f"  Number of trees: {n_estimators}")
    print(f"  Class weight: {class_weight}")
    print(f"  Max depth: {max_depth}")
    
    return model, training_time


def train_xgboost(X_train, y_train, scale_pos_weight=None, n_estimators=100,
                  max_depth=6, learning_rate=0.1, random_state=42,
                  min_child_weight=1, gamma=0, subsample=1.0, colsample_bytree=1.0,
                  reg_alpha=0, reg_lambda=1):
    """
    Train XGBoost Classifier with extended hyperparameter support.
    
    XGBoost is the state-of-the-art model for tabular data. It's:
    - Extremely powerful for structured data
    - Handles missing values automatically
    - Provides feature importance
    - Does NOT require feature scaling
    - Widely used in fraud detection
    - Supports GPU acceleration
    
    For imbalanced data, use scale_pos_weight parameter.
    
    Precision-focused hyperparameters:
    - Lower max_depth (4) → less overfitting, higher precision
    - Higher min_child_weight (5) → conservative splits
    - Higher gamma (1) → minimum loss reduction for split
    - Higher reg_lambda (5) → stronger L2 regularization
    
    Parameters:
    -----------
    X_train : array-like
        Training features (scaling not required)
    y_train : array-like
        Training labels
    scale_pos_weight : float or None
        Weight of positive class (fraud)
        Formula: (# negative samples) / (# positive samples)
    n_estimators : int
        Number of boosting rounds
    max_depth : int
        Maximum tree depth (lower = less overfitting)
    learning_rate : float
        Boosting learning rate
    min_child_weight : int
        Minimum sum of instance weight in child (higher = more conservative)
    gamma : float
        Minimum loss reduction for split (higher = regularization)
    subsample : float
        Subsample ratio of training instances
    colsample_bytree : float
        Subsample ratio of features
    reg_alpha : float
        L1 regularization
    reg_lambda : float
        L2 regularization (higher = stronger)
    random_state : int
        Random seed
        
    Returns:
    --------
    model : trained XGBClassifier model
    training_time : float
    """
    print("\n" + "="*60)
    print("Training XGBoost")
    print("="*60)
    
    start_time = time.time()
    
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        min_child_weight=min_child_weight,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1,
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    print(f"✓ Training completed in {training_time:.2f} seconds")
    print(f"  Number of estimators: {n_estimators}")
    print(f"  Max depth: {max_depth}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  scale_pos_weight: {scale_pos_weight}")
    if min_child_weight != 1 or gamma != 0 or reg_lambda != 1:
        print(f"  Regularization: min_child_weight={min_child_weight}, gamma={gamma}, lambda={reg_lambda}")
    
    return model, training_time
    print(f"  Learning rate: {learning_rate}")
    print(f"  scale_pos_weight: {scale_pos_weight}")
    
    return model, training_time


def get_feature_importance(model, feature_names=None, top_n=10):
    """
    Extract feature importance from tree-based models.
    
    Works for Random Forest and XGBoost.
    
    Parameters:
    -----------
    model : trained model
        Random Forest or XGBoost model
    feature_names : list or None
        Names of features
    top_n : int
        Number of top features to return
        
    Returns:
    --------
    importance_df : pd.DataFrame
        Feature importance scores
    """
    import pandas as pd
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        print(f"\n=== Top {top_n} Important Features ===")
        print(importance_df.to_string(index=False))
        
        return importance_df
    else:
        print("Model does not support feature_importances_")
        return None


def predict_proba_with_model(model, X_test):
    """
    Get prediction probabilities from trained model.
    
    Parameters:
    -----------
    model : trained model
        Any scikit-learn compatible model
    X_test : array-like
        Test features
        
    Returns:
    --------
    y_proba : array
        Probability predictions for positive class (fraud)
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_proba


def predict_with_threshold(y_proba, threshold=0.5):
    """
    Convert probabilities to binary predictions using custom threshold.
    
    Default threshold is 0.5, but for fraud detection we may want to:
    - Lower threshold → catch more frauds (higher recall, lower precision)
    - Raise threshold → reduce false alarms (higher precision, lower recall)
    
    Parameters:
    -----------
    y_proba : array
        Prediction probabilities
    threshold : float
        Decision threshold (default: 0.5)
        
    Returns:
    --------
    y_pred : array
        Binary predictions
    """
    y_pred = (y_proba >= threshold).astype(int)
    return y_pred


def calibrate_model(model, X_train, y_train, method='isotonic', cv=5):
    """
    Calibrate model probabilities using CalibratedClassifierCV.
    
    CRITICAL for fraud detection:
    - Raw model probabilities are often poorly calibrated
    - Banks require reliable probability estimates for risk assessment
    - Calibration fixes the gap between predicted probability and true frequency
    
    Methods:
    - 'isotonic': Non-parametric, more flexible, needs more data
    - 'sigmoid': Parametric (Platt scaling), works with less data
    
    Parameters:
    -----------
    model : trained model
        Base model to calibrate (must have predict_proba)
    X_train : array-like
        Training features (used for calibration)
    y_train : array-like
        Training labels
    method : str
        'isotonic' or 'sigmoid'
    cv : int
        Number of cross-validation folds
        
    Returns:
    --------
    calibrated_model : CalibratedClassifierCV
        Calibrated model wrapper
    calibration_time : float
        Time taken for calibration
    """
    from sklearn.calibration import CalibratedClassifierCV
    
    print(f"\n🔧 Calibrating model using {method.upper()} method...")
    print(f"   (CV folds: {cv})")
    
    start_time = time.time()
    
    calibrated_model = CalibratedClassifierCV(
        model,
        method=method,
        cv=cv,
        n_jobs=-1
    )
    
    calibrated_model.fit(X_train, y_train)
    
    calibration_time = time.time() - start_time
    
    print(f"✓ Calibration complete in {calibration_time:.2f}s")
    
    return calibrated_model, calibration_time


def train_voting_classifier(models, X_train, y_train, weights=None):
    """
    Train Voting Classifier (soft voting ensemble).
    
    Combines multiple models by averaging their probability predictions.
    Good for:
    - Combining models with different strengths (e.g., RF high precision + XGB high recall)
    - Reducing variance
    - Improving stability
    
    Parameters:
    -----------
    models : list of tuples
        [(name, model), ...] e.g., [('rf', rf_model), ('xgb', xgb_model)]
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    weights : list of float or None
        Weights for each model (e.g., [0.6, 0.4])
        
    Returns:
    --------
    voting_clf : trained VotingClassifier
    training_time : float
    """
    from sklearn.ensemble import VotingClassifier
    
    print("\n" + "="*60)
    print("Training Voting Classifier (Soft Voting Ensemble)")
    print("="*60)
    
    if weights:
        print(f"Model weights: {dict(zip([name for name, _ in models], weights))}")
    
    start_time = time.time()
    
    voting_clf = VotingClassifier(
        estimators=models,
        voting='soft',
        weights=weights,
        n_jobs=-1
    )
    
    voting_clf.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    print(f"✓ Voting Classifier trained in {training_time:.2f}s")
    
    return voting_clf, training_time


def train_stacking_classifier(base_models, final_estimator, X_train, y_train, cv=5):
    """
    Train Stacking Classifier (meta-learning ensemble).
    
    Two-level architecture:
    - Level 0: Base models make predictions
    - Level 1: Meta-model (final estimator) learns from base predictions
    
    Often outperforms voting when base models have diverse error patterns.
    
    Parameters:
    -----------
    base_models : list of tuples
        [(name, model), ...] for base level
    final_estimator : model
        Meta-learner (e.g., LogisticRegression)
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    cv : int
        CV folds for generating meta-features
        
    Returns:
    --------
    stacking_clf : trained StackingClassifier
    training_time : float
    """
    from sklearn.ensemble import StackingClassifier
    
    print("\n" + "="*60)
    print("Training Stacking Classifier (Meta-Learning Ensemble)")
    print("="*60)
    print(f"Base models: {[name for name, _ in base_models]}")
    print(f"Final estimator: {final_estimator.__class__.__name__}")
    
    start_time = time.time()
    
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=final_estimator,
        cv=cv,
        n_jobs=-1,
        passthrough=False  # Don't pass original features to meta-learner
    )
    
    stacking_clf.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    print(f"✓ Stacking Classifier trained in {training_time:.2f}s")
    
    return stacking_clf, training_time


if __name__ == "__main__":
    print("Modeling module loaded successfully.")
    print("\nAvailable models:")

    print("1. Logistic Regression - Linear baseline model")
    print("2. Random Forest - Non-linear ensemble")
    print("3. XGBoost - State-of-the-art gradient boosting")
    print("\nAll models support class weighting for imbalanced data handling.")
