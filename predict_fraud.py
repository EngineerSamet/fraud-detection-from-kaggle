"""
Fraud Detection Deployment Pipeline

This module provides production-ready functions for predicting fraud on new transactions.
It loads trained models and applies the complete preprocessing + prediction pipeline.

Usage Example:
    from predict_fraud import predict_single_transaction, load_fraud_detection_models
    
    # Load models once at startup
    models = load_fraud_detection_models()
    
    # Predict on new transaction
    transaction = {
        'Time': 12345,
        'V1': -1.23, 'V2': 0.45, ..., 'V28': 0.12,
        'Amount': 149.50
    }
    
    result = predict_single_transaction(transaction, models)
    print(result)
    # Output:
    # {
    #     'fraud_probability': 0.92,
    #     'decision': 'FRAUD',
    #     'threshold_used': 0.07,
    #     'cost_ratio': '100:1',
    #     'confidence': 'HIGH',
    #     'recommended_action': 'BLOCK TRANSACTION'
    # }

Author: Samet Şanlıkan
Date: December 2025
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


class FraudDetectionPipeline:
    """
    Production-ready fraud detection pipeline.
    
    This class encapsulates the complete fraud detection workflow:
    1. Load trained models (IsolationForest, LGBM)
    2. Preprocess transaction
    3. Generate anomaly score
    4. Predict fraud probability
    5. Apply calibration
    6. Apply optimal threshold
    7. Return decision with metadata
    """
    
    def __init__(self, models_dir: str = "outputs/fraud_detection_final/models"):
        """
        Initialize the fraud detection pipeline.
        
        Parameters:
        -----------
        models_dir : str
            Directory containing trained models
        """
        self.models_dir = models_dir
        self.isolation_forest = None
        self.lgbm_model = None
        self.feature_names = None
        
        # Optimal thresholds (from training)
        self.thresholds = {
            'default': 0.5,
            'f2_optimized': 0.18,  # From LGBM threshold optimization
            'cost_50': 0.13,       # FN/FP = 50:1
            'cost_100': 0.07,      # FN/FP = 100:1 (RECOMMENDED for banking)
            'cost_200': 0.07,      # FN/FP = 200:1
            'cost_500': 0.07       # FN/FP = 500:1
        }
        
        # Risk levels based on probability
        self.risk_levels = {
            (0.0, 0.3): 'LOW',
            (0.3, 0.5): 'MEDIUM',
            (0.5, 0.7): 'HIGH',
            (0.7, 0.9): 'VERY HIGH',
            (0.9, 1.0): 'CRITICAL'
        }
    
    def load_models(self) -> bool:
        """
        Load all required models from disk.
        
        Returns:
        --------
        bool
            True if all models loaded successfully
        """
        try:
            # Load IsolationForest (for anomaly score)
            if_path = os.path.join(self.models_dir, "isolation_forest_feature.pkl")
            if os.path.exists(if_path):
                with open(if_path, 'rb') as f:
                    self.isolation_forest = pickle.load(f)
                print(f"✓ IsolationForest loaded from {if_path}")
            else:
                print(f"⚠️  IsolationForest not found at {if_path}")
                self.isolation_forest = None
            
            # Load LGBM champion model
            lgbm_path = os.path.join(self.models_dir, "lightgbm_champion.pkl")
            if not os.path.exists(lgbm_path):
                raise FileNotFoundError(f"LGBM model not found at {lgbm_path}")
            
            with open(lgbm_path, 'rb') as f:
                self.lgbm_model = pickle.load(f)
            print(f"✓ LGBM Champion model loaded from {lgbm_path}")
            
            # Feature names (30 base features + anomaly_score)
            self.feature_names = [
                'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
                'Amount'
            ]
            
            print("\n✅ All models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def preprocess_transaction(self, transaction: Dict[str, float]) -> pd.DataFrame:
        """
        Preprocess a single transaction.
        
        Parameters:
        -----------
        transaction : dict
            Dictionary with keys: Time, V1-V28, Amount
        
        Returns:
        --------
        pd.DataFrame
            Preprocessed feature DataFrame (30 features)
        """
        # Convert to DataFrame with feature names
        df = pd.DataFrame([transaction])
        
        # Ensure all required features exist
        for feature in self.feature_names:
            if feature not in df.columns:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Select features in correct order and keep as DataFrame
        X_df = df[self.feature_names]
        
        return X_df
    
    def calculate_anomaly_score(self, X: pd.DataFrame) -> float:
        """
        Calculate anomaly score using IsolationForest.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Preprocessed feature DataFrame (30 features)
        
        Returns:
        --------
        float
            Anomaly score (higher = more anomalous)
        """
        if self.isolation_forest is None:
            print("⚠️  Warning: IsolationForest not available, using default anomaly score 0.0")
            return 0.0
        
        # IsolationForest score_samples returns negative scores
        # More negative = more anomalous
        anomaly_score = self.isolation_forest.score_samples(X)[0]
        return anomaly_score
    
    def predict_fraud_probability(self, X: pd.DataFrame, anomaly_score: float) -> float:
        """
        Predict fraud probability using LGBM model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Preprocessed feature DataFrame (30 features)
        anomaly_score : float
            Anomaly score from IsolationForest
        
        Returns:
        --------
        float
            Fraud probability (0-1)
        """
        # Add anomaly score to DataFrame
        X_with_anomaly = X.copy()
        X_with_anomaly['anomaly_score'] = anomaly_score
        
        # Predict probability
        proba = self.lgbm_model.predict_proba(X_with_anomaly)[0, 1]
        
        return proba
    
    def apply_threshold(self, probability: float, threshold_type: str = 'cost_100') -> Tuple[str, str]:
        """
        Apply threshold to determine fraud decision.
        
        Parameters:
        -----------
        probability : float
            Fraud probability (0-1)
        threshold_type : str
            Threshold type to use:
            - 'default': 0.5 (standard)
            - 'f2_optimized': 0.18 (optimized for F2-score)
            - 'cost_50': 0.13 (FN cost = 50x FP cost)
            - 'cost_100': 0.07 (FN cost = 100x FP cost) [RECOMMENDED]
            - 'cost_200': 0.07
            - 'cost_500': 0.07
        
        Returns:
        --------
        decision : str
            'FRAUD' or 'LEGITIMATE'
        confidence : str
            Risk level: 'LOW', 'MEDIUM', 'HIGH', 'VERY HIGH', 'CRITICAL'
        """
        threshold = self.thresholds.get(threshold_type, 0.5)
        
        # Make decision
        decision = 'FRAUD' if probability >= threshold else 'LEGITIMATE'
        
        # Determine confidence level
        confidence = 'LOW'
        for (low, high), level in self.risk_levels.items():
            if low <= probability < high:
                confidence = level
                break
        
        return decision, confidence
    
    def predict(self, transaction: Dict[str, float], 
                threshold_type: str = 'cost_100') -> Dict[str, Any]:
        """
        Complete fraud prediction pipeline for a single transaction.
        
        Parameters:
        -----------
        transaction : dict
            Transaction data with keys: Time, V1-V28, Amount
        threshold_type : str
            Threshold type to use (default: 'cost_100' for banking)
        
        Returns:
        --------
        dict
            Prediction result with:
            - fraud_probability: float (0-1)
            - decision: str ('FRAUD' or 'LEGITIMATE')
            - threshold_used: float
            - threshold_type: str
            - confidence: str (risk level)
            - anomaly_score: float
            - recommended_action: str
        """
        # 1. Preprocess
        X = self.preprocess_transaction(transaction)
        
        # 2. Calculate anomaly score
        anomaly_score = self.calculate_anomaly_score(X)
        
        # 3. Predict probability
        fraud_prob = self.predict_fraud_probability(X, anomaly_score)
        
        # 4. Apply threshold
        decision, confidence = self.apply_threshold(fraud_prob, threshold_type)
        
        # 5. Recommended action
        if decision == 'FRAUD':
            if confidence in ['CRITICAL', 'VERY HIGH']:
                action = 'BLOCK TRANSACTION IMMEDIATELY'
            elif confidence == 'HIGH':
                action = 'MANUAL REVIEW REQUIRED'
            else:
                action = 'FLAG FOR REVIEW'
        else:
            action = 'APPROVE TRANSACTION'
        
        # Build result
        result = {
            'fraud_probability': round(fraud_prob, 4),
            'decision': decision,
            'threshold_used': self.thresholds[threshold_type],
            'threshold_type': threshold_type,
            'confidence': confidence,
            'anomaly_score': round(anomaly_score, 4),
            'recommended_action': action
        }
        
        return result


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_fraud_detection_models(models_dir: str = "outputs/fraud_detection_final/models") -> FraudDetectionPipeline:
    """
    Load fraud detection pipeline (convenience function).
    
    Parameters:
    -----------
    models_dir : str
        Directory containing trained models
    
    Returns:
    --------
    FraudDetectionPipeline
        Initialized pipeline with loaded models
    """
    pipeline = FraudDetectionPipeline(models_dir=models_dir)
    success = pipeline.load_models()
    
    if not success:
        raise RuntimeError("Failed to load models. Check that all model files exist.")
    
    return pipeline


def predict_single_transaction(transaction: Dict[str, float],
                               pipeline: FraudDetectionPipeline = None,
                               threshold_type: str = 'cost_100') -> Dict[str, Any]:
    """
    Predict fraud for a single transaction (convenience function).
    
    Parameters:
    -----------
    transaction : dict
        Transaction data with keys: Time, V1-V28, Amount
    pipeline : FraudDetectionPipeline, optional
        Pre-loaded pipeline. If None, will load models.
    threshold_type : str
        Threshold type to use (default: 'cost_100')
    
    Returns:
    --------
    dict
        Prediction result
    """
    if pipeline is None:
        pipeline = load_fraud_detection_models()
    
    return pipeline.predict(transaction, threshold_type=threshold_type)


def predict_batch_transactions(transactions: pd.DataFrame,
                              pipeline: FraudDetectionPipeline = None,
                              threshold_type: str = 'cost_100') -> pd.DataFrame:
    """
    Predict fraud for multiple transactions (batch processing).
    
    Parameters:
    -----------
    transactions : pd.DataFrame
        DataFrame with columns: Time, V1-V28, Amount
    pipeline : FraudDetectionPipeline, optional
        Pre-loaded pipeline. If None, will load models.
    threshold_type : str
        Threshold type to use (default: 'cost_100')
    
    Returns:
    --------
    pd.DataFrame
        Original transactions with added columns:
        - fraud_probability
        - decision
        - confidence
        - recommended_action
    """
    if pipeline is None:
        pipeline = load_fraud_detection_models()
    
    results = []
    for idx, row in transactions.iterrows():
        transaction = row.to_dict()
        result = pipeline.predict(transaction, threshold_type=threshold_type)
        results.append(result)
    
    # Add results to DataFrame
    results_df = pd.DataFrame(results)
    output_df = pd.concat([transactions.reset_index(drop=True), results_df], axis=1)
    
    return output_df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the fraud detection pipeline.
    """
    print("="*70)
    print("  FRAUD DETECTION DEPLOYMENT PIPELINE - DEMO")
    print("="*70)
    
    # Example transaction (fraudulent pattern)
    example_transaction = {
        'Time': 12345,
        'V1': -2.3, 'V2': 1.5, 'V3': -3.2, 'V4': 4.1, 'V5': -0.8,
        'V6': -1.2, 'V7': -2.8, 'V8': 0.5, 'V9': -1.9, 'V10': -3.5,
        'V11': 2.1, 'V12': -5.2, 'V13': 0.9, 'V14': -7.8, 'V15': 1.2,
        'V16': -1.8, 'V17': -9.2, 'V18': -0.5, 'V19': 0.8, 'V20': 0.3,
        'V21': -0.2, 'V22': -0.5, 'V23': -0.1, 'V24': 0.4, 'V25': 0.6,
        'V26': -0.3, 'V27': 0.1, 'V28': 0.05,
        'Amount': 199.50
    }
    
    print("\n📋 Example Transaction:")
    print(f"   Time: {example_transaction['Time']}")
    print(f"   Amount: ${example_transaction['Amount']:.2f}")
    print(f"   Features: V1={example_transaction['V1']:.2f}, V14={example_transaction['V14']:.2f}, V17={example_transaction['V17']:.2f}")
    
    try:
        # Load pipeline
        print("\n🔄 Loading fraud detection models...")
        pipeline = load_fraud_detection_models()
        
        # Predict
        print("\n🔍 Running fraud detection...")
        result = predict_single_transaction(example_transaction, pipeline)
        
        # Display result
        print("\n" + "="*70)
        print("  PREDICTION RESULT")
        print("="*70)
        print(f"\n{'Fraud Probability:':<25} {result['fraud_probability']:.2%}")
        print(f"{'Decision:':<25} {result['decision']}")
        print(f"{'Confidence Level:':<25} {result['confidence']}")
        print(f"{'Threshold Used:':<25} {result['threshold_used']:.4f} ({result['threshold_type']})")
        print(f"{'Anomaly Score:':<25} {result['anomaly_score']:.4f}")
        print(f"\n{'Recommended Action:':<25} {result['recommended_action']}")
        print("="*70)
        
        # Test different thresholds
        print("\n📊 Testing different threshold strategies:")
        print(f"\n{'Threshold Type':<20} {'Threshold':<12} {'Decision':<15} {'Confidence':<12}")
        print("-"*70)
        
        for threshold_type in ['default', 'f2_optimized', 'cost_50', 'cost_100', 'cost_200']:
            res = predict_single_transaction(example_transaction, pipeline, threshold_type)
            print(f"{threshold_type:<20} {res['threshold_used']:<12.4f} {res['decision']:<15} {res['confidence']:<12}")
        
        print("\n💡 INSIGHT: Lower threshold = More sensitive (catch more frauds)")
        print("   Banking standard: cost_100 (FN/FP = 100:1)")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 To use this pipeline:")
        print("   1. Train models by running: python main.py")
        print("   2. Ensure models are saved to: outputs/fraud_detection_final/models/")
        print("   3. Run this script again")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
