"""
Train XGBoost model with time-series cross-validation.
Implements proper temporal splitting to prevent data leakage.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import (
    precision_recall_curve, auc, f1_score, 
    recall_score, precision_score, classification_report
)
from typing import Dict, List, Tuple
from datetime import datetime

from config import (
    MODEL_DIR,
    RANDOM_STATE,
    N_ESTIMATORS,
    LEARNING_RATE,
    MAX_DEPTH,
    SUBSAMPLE,
    COLSAMPLE_BYTREE,
    EVAL_METRIC,
    TREE_METHOD,
    EARLY_STOPPING_ROUNDS,
    N_SPLITS_CV,
)
from utils import setup_logger, get_scale_pos_weight, get_timestamp, save_json
from imbalance_handler import ImbalanceHandler

logger = setup_logger(__name__)


class FraudDetectionTrainer:
    """
    Train XGBoost classifier for fraud detection with time-series awareness.
    """
    
    def __init__(self):
        """Initialize trainer with default hyperparameters."""
        self.model = None
        self.feature_names = None
        self.training_history = {}
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'is_fraud',
        timestamp_col: str = 'timestamp',
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Prepare data with temporal split (NO stratification for time-series).
        
        Parameters
        ----------
        df : pd.DataFrame
            Full dataset with features and target
        target_col : str
            Target column name
        timestamp_col : str
            Timestamp column for sorting
        test_size : float
            Proportion of data for test set
        
        Returns
        -------
        tuple
            X_train, X_test, y_train, y_test, timestamps_train, timestamps_test
        
        Notes
        -----
        CRITICAL: Sort by timestamp before splitting to maintain temporal order!
        DO NOT use stratify=y with time-series data (breaks temporal order).
        """
        logger.info("Preparing data with temporal split...")
        
        # Sort by timestamp (CRITICAL for time-series)
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        logger.info(f"   Data sorted by {timestamp_col}")
        
        # Separate features, target, and timestamps
        feature_cols = [col for col in df.columns if col not in [target_col, timestamp_col, 'hash', 'from', 'to', 'address']]
        X = df[feature_cols]
        y = df[target_col]
        timestamps = df[timestamp_col]
        
        self.feature_names = feature_cols
        
        # Temporal split (train on older data, test on newer)
        split_idx = int(len(df) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        timestamps_train = timestamps.iloc[:split_idx]
        timestamps_test = timestamps.iloc[split_idx:]
        
        logger.info(f"✅ Data split:")
        logger.info(f"   Train: {len(X_train)} samples ({timestamps_train.min()} to {timestamps_train.max()})")
        logger.info(f"   Test: {len(X_test)} samples ({timestamps_test.min()} to {timestamps_test.max()})")
        logger.info(f"   Features: {len(feature_cols)}")
        logger.info(f"   Train fraud rate: {y_train.mean():.2%}")
        logger.info(f"   Test fraud rate: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test, timestamps_train, timestamps_test
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        apply_oversampling: bool = True
    ) -> XGBClassifier:
        """
        Train XGBoost model with best practices for imbalanced data.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels
        X_val : pd.DataFrame
            Validation features
        y_val : pd.Series
            Validation labels
        apply_oversampling : bool
            Whether to apply ADASYN oversampling
        
        Returns
        -------
        XGBClassifier
            Trained model
        
        References
        ----------
        XGBoost parameters: https://xgboost.readthedocs.io/en/stable/parameter.html
        Scale pos weight: ratio of negative to positive samples
        """
        logger.info("Training XGBoost model...")
        
        # Apply oversampling to training set only
        if apply_oversampling:
            handler = ImbalanceHandler(method='adasyn')
            X_train, y_train = handler.fit_resample(X_train, y_train)
        
        # Calculate scale_pos_weight
        scale_pos_weight = get_scale_pos_weight(y_train)
        
        # Initialize model with research-backed hyperparameters
        self.model = XGBClassifier(
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH,
            subsample=SUBSAMPLE,
            colsample_bytree=COLSAMPLE_BYTREE,
            scale_pos_weight=scale_pos_weight,
            eval_metric=EVAL_METRIC,
            tree_method=TREE_METHOD,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            verbosity=1
        )
        
        logger.info(f"Model hyperparameters:")
        logger.info(f"   n_estimators: {N_ESTIMATORS}")
        logger.info(f"   learning_rate: {LEARNING_RATE}")
        logger.info(f"   max_depth: {MAX_DEPTH}")
        logger.info(f"   scale_pos_weight: {scale_pos_weight:.2f}")
        logger.info(f"   eval_metric: {EVAL_METRIC}")
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )
        
        logger.info(f"✅ Training complete")
        logger.info(f"   Best iteration: {self.model.best_iteration}")
        logger.info(f"   Best score: {self.model.best_score:.4f}")
        
        return self.model
    
    def time_series_cv_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        timestamps: pd.Series,
        n_splits: int = N_SPLITS_CV
    ) -> Dict[str, List[float]]:
        """
        Perform time-series cross-validation.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        timestamps : pd.Series
            Timestamps for temporal ordering
        n_splits : int
            Number of CV splits
        
        Returns
        -------
        dict
            CV scores for each metric
        
        Notes
        -----
        CRITICAL: Sort data by timestamp before splitting to prevent data leakage!
        Only oversample training set in each fold, never test set.
        
        References
        ----------
        TimeSeriesSplit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
        """
        logger.info(f"Performing time-series cross-validation ({n_splits} splits)...")
        
        # Sort by timestamp
        sort_idx = np.argsort(timestamps)
        X_sorted = X.iloc[sort_idx].reset_index(drop=True)
        y_sorted = y.iloc[sort_idx].reset_index(drop=True)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = {
            'pr_auc': [],
            'f1': [],
            'recall': [],
            'precision': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_sorted)):
            logger.info(f"\n=== Fold {fold+1}/{n_splits} ===")
            
            X_train, X_test = X_sorted.iloc[train_idx], X_sorted.iloc[test_idx]
            y_train, y_test = y_sorted.iloc[train_idx], y_sorted.iloc[test_idx]
            
            # Apply ADASYN only to training set
            handler = ImbalanceHandler(method='adasyn')
            X_train_res, y_train_res = handler.fit_resample(X_train, y_train)
            
            # Train model
            scale_pos_weight = get_scale_pos_weight(y_train_res)
            
            model = XGBClassifier(
                n_estimators=N_ESTIMATORS,
                learning_rate=LEARNING_RATE,
                max_depth=MAX_DEPTH,
                subsample=SUBSAMPLE,
                colsample_bytree=COLSAMPLE_BYTREE,
                scale_pos_weight=scale_pos_weight,
                tree_method=TREE_METHOD,
                eval_metric=EVAL_METRIC,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                random_state=RANDOM_STATE,
                use_label_encoder=False,
                verbosity=0
            )
            
            model.fit(
                X_train_res, y_train_res,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            # Evaluate
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall, precision)
            
            cv_scores['pr_auc'].append(pr_auc)
            cv_scores['f1'].append(f1_score(y_test, y_pred))
            cv_scores['recall'].append(recall_score(y_test, y_pred))
            cv_scores['precision'].append(precision_score(y_test, y_pred))
            
            logger.info(f"PR-AUC: {pr_auc:.4f}, F1: {cv_scores['f1'][-1]:.4f}")
        
        # Print summary
        logger.info("\n=== Cross-Validation Summary ===")
        for metric, scores in cv_scores.items():
            logger.info(f"{metric.upper()}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        return cv_scores
    
    def save_model(self, suffix: str = "") -> str:
        """
        Save trained model with timestamp.
        
        Parameters
        ----------
        suffix : str
            Optional suffix for filename
        
        Returns
        -------
        str
            Saved model path
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        timestamp = get_timestamp()
        filename = f"xgb_fraud_{timestamp}{suffix}.json"
        filepath = MODEL_DIR / filename
        
        self.model.save_model(filepath)
        logger.info(f"✅ Model saved: {filepath}")
        
        # Save hyperparameters
        params_file = MODEL_DIR / f"params_{timestamp}{suffix}.json"
        params = {
            'timestamp': timestamp,
            'hyperparameters': self.model.get_params(),
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names if self.feature_names else [],
        }
        save_json(params, params_file)
        
        return str(filepath)


def main():
    """Main execution function for testing."""
    from config import PROCESSED_DATA_DIR
    
    # Load processed data
    data_file = PROCESSED_DATA_DIR / "features.csv"
    if not data_file.exists():
        logger.error(f"Processed data not found: {data_file}")
        logger.info("Run feature_engineering.py first!")
        return
    
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    logger.info(f"Loaded {len(df)} samples")
    
    # Initialize trainer
    trainer = FraudDetectionTrainer()
    
    # Prepare data
    X_train, X_test, y_train, y_test, ts_train, ts_test = trainer.prepare_data(df)
    
    # Train model
    trainer.train_model(X_train, y_train, X_test, y_test)
    
    # Evaluate
    if trainer.model:
        y_pred = trainer.model.predict(X_test)
        y_pred_proba = trainer.model.predict_proba(X_test)[:, 1]
    else:
        logger.error("Model training failed")
        return None
    
    print("\n=== Test Set Evaluation ===")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    
    # Calculate PR-AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"\nPR-AUC: {pr_auc:.4f}")
    
    # Save model
    trainer.save_model()
    
    return trainer


if __name__ == "__main__":
    main()
