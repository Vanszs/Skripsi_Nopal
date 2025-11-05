"""
Ethereum Fraud Detection System
End-to-end ML pipeline for detecting fraudulent transactions on Ethereum Mainnet
using XGBoost with SHAP explainability and Network Graph Analysis.
"""

__version__ = "1.0.0"
__author__ = "Ethereum Fraud Detection Team"

from . import config
from . import utils
from . import fetch_transactions
from . import feature_engineering
from . import network_graph
from . import imbalance_handler
from . import model_train
from . import model_explain
from . import evaluate

__all__ = [
    'config',
    'utils',
    'fetch_transactions',
    'feature_engineering',
    'network_graph',
    'imbalance_handler',
    'model_train',
    'model_explain',
    'evaluate',
]
