"""
Evaluate fraud detection model with comprehensive metrics.
Focus on metrics suitable for imbalanced classification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score
)
from xgboost import XGBClassifier
from typing import Dict, Optional
from pathlib import Path

from config import MODEL_DIR
from utils import setup_logger, get_timestamp, save_json

logger = setup_logger(__name__)

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


class ModelEvaluator:
    """
    Comprehensive evaluation for fraud detection model.
    Prioritizes metrics suitable for imbalanced datasets.
    """
    
    def __init__(self, model: XGBClassifier):
        """
        Initialize evaluator with trained model.
        
        Parameters
        ----------
        model : XGBClassifier
            Trained model to evaluate
        """
        self.model = model
        self.results = {}
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """
        Perform comprehensive evaluation.
        
        Parameters
        ----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test labels
        
        Returns
        -------
        dict
            Evaluation metrics
        
        Metrics Priority for Imbalanced Data
        ------------------------------------
        1. PR-AUC (Precision-Recall AUC) - Most informative for imbalanced data
        2. F1-Score - Balance of precision and recall
        3. Recall - Minimize false negatives (missing fraud)
        4. Precision - Minimize false positives (false alarms)
        5. ROC-AUC - Less informative but included for completeness
        """
        logger.info("Evaluating model...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Classification metrics
        self.results['precision'] = precision_score(y_test, y_pred)
        self.results['recall'] = recall_score(y_test, y_pred)
        self.results['f1_score'] = f1_score(y_test, y_pred)
        
        # PR curve and AUC
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
        self.results['pr_auc'] = auc(recall, precision)
        self.results['avg_precision'] = average_precision_score(y_test, y_pred_proba)
        
        # ROC curve and AUC
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
        self.results['roc_auc'] = auc(fpr, tpr)
        
        # Confusion matrix (handle single-class edge case)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        self.results['confusion_matrix'] = cm.tolist()
        
        # Handle edge case where matrix might be 1x1 (single class)
        if cm.shape[0] >= 2 and cm.shape[1] >= 2:
            self.results['tn'] = int(cm[0, 0])
            self.results['fp'] = int(cm[0, 1])
            self.results['fn'] = int(cm[1, 0])
            self.results['tp'] = int(cm[1, 1])
        else:
            # Single class case - all negative or all positive
            logger.warning("Single class detected in predictions. Setting confusion matrix values.")
            self.results['tn'] = int(cm[0, 0]) if len(y_test[y_test == 0]) > 0 else 0
            self.results['fp'] = 0
            self.results['fn'] = 0
            self.results['tp'] = 0
        
        # Print results
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"\nPR-AUC (Primary Metric):    {self.results['pr_auc']:.4f}")
        print(f"ROC-AUC:                     {self.results['roc_auc']:.4f}")
        print(f"F1-Score:                    {self.results['f1_score']:.4f}")
        print(f"Precision:                   {self.results['precision']:.4f}")
        print(f"Recall:                      {self.results['recall']:.4f}")
        print(f"Average Precision:           {self.results['avg_precision']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  TN: {self.results['tn']}  FP: {self.results['fp']}")
        print(f"  FN: {self.results['fn']}  TP: {self.results['tp']}")
        
        print(f"\nClassification Report:")
        try:
            print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud'], labels=[0, 1], zero_division=0))
        except ValueError as e:
            logger.warning(f"Classification report failed: {e}")
            print("  [Skipped - insufficient classes in predictions]")
        
        logger.info("✅ Evaluation complete")
        
        return self.results
    
    def plot_confusion_matrix(self, save_path: Optional[Path] = None):
        """
        Plot confusion matrix heatmap.
        
        Parameters
        ----------
        save_path : Path, optional
            Path to save plot
        """
        if 'confusion_matrix' not in self.results:
            raise ValueError("Run evaluate() first")
        
        logger.info("Plotting confusion matrix...")
        
        cm = np.array(self.results['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud']
        )
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path is None:
            timestamp = get_timestamp()
            save_path = MODEL_DIR / f"confusion_matrix_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved confusion matrix: {save_path}")
        plt.close()
    
    def plot_pr_curve(self, X_test: pd.DataFrame, y_test: pd.Series, save_path: Optional[Path] = None):
        """
        Plot Precision-Recall curve.
        
        Parameters
        ----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test labels
        save_path : Path, optional
            Path to save plot
        """
        logger.info("Plotting PR curve...")
        
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path is None:
            timestamp = get_timestamp()
            save_path = MODEL_DIR / f"pr_curve_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved PR curve: {save_path}")
        plt.close()
    
    def plot_roc_curve(self, X_test: pd.DataFrame, y_test: pd.Series, save_path: Optional[Path] = None):
        """
        Plot ROC curve.
        
        Parameters
        ----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test labels
        save_path : Path, optional
            Path to save plot
        """
        logger.info("Plotting ROC curve...")
        
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path is None:
            timestamp = get_timestamp()
            save_path = MODEL_DIR / f"roc_curve_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved ROC curve: {save_path}")
        plt.close()
    
    def plot_feature_importance(self, feature_names: list, top_n: int = 20, save_path: Optional[Path] = None):
        """
        Plot feature importance from model.
        
        Parameters
        ----------
        feature_names : list
            Feature names
        top_n : int
            Number of top features to display
        save_path : Path, optional
            Path to save plot
        """
        logger.info("Plotting feature importance...")
        
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importance[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path is None:
            timestamp = get_timestamp()
            save_path = MODEL_DIR / f"feature_importance_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved feature importance: {save_path}")
        plt.close()
    
    def save_results(self, suffix: str = ""):
        """
        Save evaluation results to JSON.
        
        Parameters
        ----------
        suffix : str
            Optional filename suffix
        """
        if not self.results:
            raise ValueError("Run evaluate() first")
        
        timestamp = get_timestamp()
        filename = MODEL_DIR / f"evaluation_results_{timestamp}{suffix}.json"
        
        save_json(self.results, filename)
        logger.info(f"✅ Results saved: {filename}")


def main():
    """Main execution function for testing."""
    from config import PROCESSED_DATA_DIR
    import glob
    
    # Load model
    model_files = sorted(glob.glob(str(MODEL_DIR / "xgb_fraud_*.json")))
    if not model_files:
        logger.error("No trained model found!")
        return
    
    latest_model = model_files[-1]
    logger.info(f"Loading model: {latest_model}")
    
    model = XGBClassifier()
    model.load_model(latest_model)
    
    # Load data
    data_file = PROCESSED_DATA_DIR / "features.csv"
    df = pd.read_csv(data_file)
    
    # Prepare test set
    feature_cols = [col for col in df.columns if col not in ['is_fraud', 'timestamp', 'hash', 'from', 'to', 'address']]
    X = df[feature_cols]
    y = df['is_fraud']
    
    split_idx = int(len(df) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    # Evaluate
    evaluator = ModelEvaluator(model)
    evaluator.evaluate(X_test, y_test)
    
    # Generate plots
    evaluator.plot_confusion_matrix()
    evaluator.plot_pr_curve(X_test, y_test)
    evaluator.plot_roc_curve(X_test, y_test)
    evaluator.plot_feature_importance(feature_cols)
    
    # Save results
    evaluator.save_results()
    
    logger.info("✅ Evaluation complete!")
    
    return evaluator


if __name__ == "__main__":
    main()
