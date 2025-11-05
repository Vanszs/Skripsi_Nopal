"""
Handle imbalanced dataset using ADASYN oversampling.
Critical for fraud detection where fraud class is minority (~1-5%).
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE
from collections import Counter
from typing import Tuple, Dict

from config import RANDOM_STATE
from utils import setup_logger

logger = setup_logger(__name__)


class ImbalanceHandler:
    """
    Handle class imbalance in fraud detection dataset.
    Uses ADASYN (Adaptive Synthetic Sampling) for intelligent oversampling.
    """
    
    def __init__(self, method: str = 'adasyn', random_state: int = RANDOM_STATE):
        """
        Initialize imbalance handler.
        
        Parameters
        ----------
        method : str
            Sampling method: 'adasyn' or 'smote'
        random_state : int
            Random seed for reproducibility
        
        References
        ----------
        ADASYN: He et al. (2008). "ADASYN: Adaptive synthetic sampling approach"
        SMOTE: Chawla et al. (2002). "SMOTE: Synthetic minority over-sampling technique"
        """
        self.method = method
        self.random_state = random_state
        
        if method == 'adasyn':
            self.sampler = ADASYN(random_state=random_state)
        elif method == 'smote':
            self.sampler = SMOTE(random_state=random_state)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'adasyn' or 'smote'")
        
        logger.info(f"Initialized {method.upper()} sampler")
    
    def fit_resample(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply oversampling to balance dataset.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable (0=normal, 1=fraud)
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            Resampled X and y
        
        Notes
        -----
        CRITICAL: Only apply to training set to prevent data leakage!
        Test set should remain unchanged to reflect real-world distribution.
        """
        logger.info("Applying oversampling...")
        logger.info(f"   Original distribution: {Counter(y)}")
        
        try:
            X_resampled_array, y_resampled_array = self.sampler.fit_resample(X, y)
            
            logger.info(f"   Resampled distribution: {Counter(y_resampled_array)}")
            logger.info(f"✅ Oversampling complete")
            logger.info(f"   Original size: {len(y)} → Resampled: {len(y_resampled_array)}")
            
            # Convert back to DataFrame/Series
            if isinstance(X, pd.DataFrame):
                X_resampled = pd.DataFrame(X_resampled_array, columns=X.columns)
            else:
                X_resampled = pd.DataFrame(X_resampled_array)
                
            if isinstance(y, pd.Series):
                y_resampled = pd.Series(y_resampled_array, name=y.name)
            else:
                y_resampled = pd.Series(y_resampled_array)
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"Oversampling failed: {e}")
            logger.warning("Returning original data")
            return X, y
    
    def get_sample_strategy(self, y: pd.Series, target_ratio: float = 0.5) -> Dict[int, int]:
        """
        Calculate sampling strategy for custom ratio.
        
        Parameters
        ----------
        y : pd.Series
            Target variable
        target_ratio : float
            Desired ratio of minority to majority class
        
        Returns
        -------
        Dict[int, int]
            Sampling strategy {class: n_samples}
        """
        class_counts = Counter(y)
        majority_class = max(class_counts.keys(), key=lambda k: class_counts[k])
        minority_class = min(class_counts.keys(), key=lambda k: class_counts[k])
        
        n_majority = class_counts[majority_class]
        n_minority_target = int(n_majority * target_ratio)
        
        strategy = {
            majority_class: n_majority,
            minority_class: n_minority_target
        }
        
        logger.info(f"Custom strategy: {strategy}")
        return strategy


def main():
    """Main execution function for testing."""
    from sklearn.datasets import make_classification
    
    # Create imbalanced dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        weights=[0.95, 0.05],  # 95% normal, 5% fraud
        random_state=RANDOM_STATE
    )
    
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y, name='is_fraud')
    
    print("\n=== Testing Imbalance Handler ===")
    print(f"Original distribution: {Counter(y)}")
    
    # Test ADASYN
    handler = ImbalanceHandler(method='adasyn')
    X_resampled, y_resampled = handler.fit_resample(X, y)
    
    print(f"Resampled distribution: {Counter(y_resampled)}")
    print(f"Shape: {X_resampled.shape}")
    
    return X_resampled, y_resampled


if __name__ == "__main__":
    main()
