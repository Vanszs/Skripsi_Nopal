"""
SHAP (SHapley Additive exPlanations) explainability for XGBoost model.
Provides both global and local explanations for fraud predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from pathlib import Path
from typing import Optional

from config import MODEL_DIR
from utils import setup_logger, get_timestamp

logger = setup_logger(__name__)


class ModelExplainer:
    """
    Generate SHAP explanations for fraud detection model.
    Provides interpretable insights into model predictions.
    """
    
    def __init__(self, model: XGBClassifier):
        """
        Initialize explainer with trained model.
        
        Parameters
        ----------
        model : XGBClassifier
            Trained XGBoost model
        
        References
        ----------
        SHAP: Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions"
        TreeExplainer: Optimized for tree-based models like XGBoost
        """
        self.model = model
        self.explainer = None
        self.shap_values = None
    
    def create_explainer(self, X_background: Optional[pd.DataFrame] = None):
        """
        Create SHAP TreeExplainer.
        
        Parameters
        ----------
        X_background : pd.DataFrame, optional
            Background dataset for explainer (uses full dataset if None)
        
        Notes
        -----
        TreeExplainer is optimized for tree models and much faster than KernelExplainer.
        """
        logger.info("Creating SHAP TreeExplainer...")
        
        # TreeExplainer for XGBoost (optimized)
        if X_background is not None:
            self.explainer = shap.TreeExplainer(self.model, X_background)
        else:
            self.explainer = shap.TreeExplainer(self.model)
        
        logger.info("✅ Explainer created")
    
    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for dataset.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        
        Returns
        -------
        np.ndarray
            SHAP values array
        """
        if self.explainer is None:
            self.create_explainer()
        
        logger.info(f"Computing SHAP values for {len(X)} samples...")
        self.shap_values = self.explainer.shap_values(X)
        logger.info("✅ SHAP values computed")
        
        return self.shap_values
    
    def plot_global_importance(
        self,
        X: pd.DataFrame,
        max_display: int = 20,
        save_path: Optional[Path] = None
    ):
        """
        Plot global feature importance using SHAP.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        max_display : int
            Maximum number of features to display
        save_path : Path, optional
            Path to save plot
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        logger.info("Generating global feature importance plot...")
        
        # Summary plot (bar)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, 
            X, 
            plot_type="bar",
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        
        if save_path is None:
            timestamp = get_timestamp()
            save_path = MODEL_DIR / f"shap_summary_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved global importance plot: {save_path}")
        plt.close()
        
        # Summary plot (beeswarm)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, 
            X,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        
        beeswarm_path = str(save_path).replace('.png', '_beeswarm.png')
        plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved beeswarm plot: {beeswarm_path}")
        plt.close()
    
    def plot_local_explanation(
        self,
        X: pd.DataFrame,
        sample_idx: int = 0,
        save_path: Optional[Path] = None
    ):
        """
        Plot local explanation for a single prediction.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        sample_idx : int
            Index of sample to explain
        save_path : Path, optional
            Path to save plot
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        logger.info(f"Generating local explanation for sample {sample_idx}...")
        
        # Waterfall plot
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[sample_idx],
                base_values=self.explainer.expected_value,
                data=X.iloc[sample_idx],
                feature_names=X.columns.tolist()
            ),
            show=False
        )
        
        if save_path is None:
            timestamp = get_timestamp()
            save_path = MODEL_DIR / f"shap_waterfall_{sample_idx}_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved local explanation: {save_path}")
        plt.close()
    
    def plot_dependence(
        self,
        X: pd.DataFrame,
        feature: str,
        interaction_feature: Optional[str] = None,
        save_path: Optional[Path] = None
    ):
        """
        Plot SHAP dependence for a feature.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        feature : str
            Feature to plot
        interaction_feature : str, optional
            Feature to color by (auto-detected if None)
        save_path : Path, optional
            Path to save plot
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        logger.info(f"Generating dependence plot for '{feature}'...")
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature,
            self.shap_values,
            X,
            interaction_index=interaction_feature,
            show=False
        )
        
        if save_path is None:
            timestamp = get_timestamp()
            save_path = MODEL_DIR / f"shap_dependence_{feature}_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved dependence plot: {save_path}")
        plt.close()
    
    def explain_predictions(
        self,
        X: pd.DataFrame,
        y_true: Optional[pd.Series] = None,
        n_samples: int = 5
    ):
        """
        Generate comprehensive explanations for multiple predictions.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y_true : pd.Series, optional
            True labels for comparison
        n_samples : int
            Number of samples to explain
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        predictions = self.model.predict_proba(X)[:, 1]
        
        print("\n=== SHAP Explanation Summary ===")
        
        for i in range(min(n_samples, len(X))):
            print(f"\n--- Sample {i} ---")
            print(f"Predicted fraud probability: {predictions[i]:.4f}")
            
            if y_true is not None:
                print(f"True label: {'Fraud' if y_true.iloc[i] == 1 else 'Normal'}")
            
            # Top contributing features
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'shap_value': self.shap_values[i]
            }).sort_values('shap_value', key=abs, ascending=False)
            
            print("\nTop 5 contributing features:")
            for _, row in feature_importance.head(5).iterrows():
                direction = "increases" if row['shap_value'] > 0 else "decreases"
                print(f"  {row['feature']}: {row['shap_value']:.4f} ({direction} fraud risk)")


def main():
    """Main execution function for testing."""
    from config import PROCESSED_DATA_DIR
    import glob
    
    # Load model
    model_files = sorted(glob.glob(str(MODEL_DIR / "xgb_fraud_*.json")))
    if not model_files:
        logger.error("No trained model found!")
        logger.info("Run model_train.py first!")
        return
    
    latest_model = model_files[-1]
    logger.info(f"Loading model: {latest_model}")
    
    model = XGBClassifier()
    model.load_model(latest_model)
    
    # Load test data
    data_file = PROCESSED_DATA_DIR / "features.csv"
    df = pd.read_csv(data_file)
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['is_fraud', 'timestamp', 'hash', 'from', 'to', 'address']]
    X = df[feature_cols]
    y = df['is_fraud']
    
    # Take test set (last 20%)
    split_idx = int(len(df) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    logger.info(f"Explaining {len(X_test)} test samples...")
    
    # Create explainer
    explainer = ModelExplainer(model)
    explainer.compute_shap_values(X_test)
    
    # Generate plots
    explainer.plot_global_importance(X_test)
    explainer.plot_local_explanation(X_test, sample_idx=0)
    
    # Explain predictions
    explainer.explain_predictions(X_test, y_test, n_samples=5)
    
    logger.info("✅ SHAP analysis complete")
    
    return explainer


if __name__ == "__main__":
    main()
