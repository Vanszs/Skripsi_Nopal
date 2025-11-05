"""
Configuration loader for Ethereum fraud detection system.
Loads environment variables and provides centralized config access.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# API Configuration
MAINNET_RPC_URL = os.getenv("MAINNET_RPC_URL")
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

# Directory Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model Hyperparameters (from research best practices)
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", 300))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.05))
MAX_DEPTH = int(os.getenv("MAX_DEPTH", 6))
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
EVAL_METRIC = "aucpr"
TREE_METHOD = "hist"
EARLY_STOPPING_ROUNDS = 10

# Data Sampling Configuration (for mainnet)
START_BLOCK = int(os.getenv("START_BLOCK", 18000000))
END_BLOCK = os.getenv("END_BLOCK", "latest")
MAX_ADDRESSES = int(os.getenv("MAX_ADDRESSES", 1000))

# API Rate Limiting
ETHERSCAN_MAX_REQUESTS_PER_SEC = 5
ETHERSCAN_BACKOFF_TIME = 0.2  # seconds

# Network Graph Configuration
BETWEENNESS_SAMPLE_SIZE = 1000  # Sample size for large graphs
PAGERANK_ALPHA = 0.85
PAGERANK_MAX_ITER = 100

# Cross-Validation Configuration
N_SPLITS_CV = 5

# Feature Engineering Constants
LOG_TRANSFORM_FEATURES = ["value_eth", "gas_price", "gas_used"]

# Known Scam Address Sources
SCAM_ADDRESS_SOURCES = [
    "https://api.etherscan.io/api?module=account&action=tokentx&contractaddress=0x",
    "https://xblock.pro/api/phishing-addresses",
]

# Logging Configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

def validate_config():
    """Validate that required configuration is present."""
    if not MAINNET_RPC_URL:
        raise ValueError("MAINNET_RPC_URL not set in .env file")
    if not ETHERSCAN_API_KEY:
        raise ValueError("ETHERSCAN_API_KEY not set in .env file")
    
    print("✅ Configuration validated successfully")
    print(f"   - Data directory: {DATA_DIR}")
    print(f"   - Model directory: {MODEL_DIR}")
    print(f"   - Random state: {RANDOM_STATE}")

if __name__ == "__main__":
    validate_config()
