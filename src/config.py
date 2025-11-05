"""
Configuration loader for blockchain fraud detection system.
Supports multiple networks: BSC, opBNB, Ethereum Mainnet.
Loads environment variables and provides centralized config access.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# ============================================================
# NETWORK CONFIGURATION
# ============================================================

# Active Network (BSC, OPBNB, or ETH)
ACTIVE_NETWORK = os.getenv("ACTIVE_NETWORK", "BSC").upper()

# Network-specific RPC URLs
BSC_RPC_URL = os.getenv("BSC_RPC_URL")
OPBNB_RPC_URL = os.getenv("OPBNB_RPC_URL")
ETH_MAINNET_RPC_URL = os.getenv("ETH_MAINNET_RPC_URL")

# Network mapping
NETWORK_CONFIG = {
    "BSC": {
        "rpc_url": BSC_RPC_URL,
        "explorer_api": "https://api.bscscan.com/api",
        "api_key": os.getenv("BSCSCAN_API_KEY"),
        "chain_id": 56,
        "name": "BNB Smart Chain",
        "native_token": "BNB"
    },
    "OPBNB": {
        "rpc_url": OPBNB_RPC_URL,
        "explorer_api": "https://api-opbnb.bscscan.com/api",
        "api_key": os.getenv("BSCSCAN_API_KEY"),
        "chain_id": 204,
        "name": "opBNB Mainnet",
        "native_token": "BNB"
    },
    "ETH": {
        "rpc_url": ETH_MAINNET_RPC_URL,
        "explorer_api": "https://api.etherscan.io/v2/api",
        "api_key": os.getenv("ETHERSCAN_API_KEY"),
        "chain_id": 1,
        "name": "Ethereum Mainnet",
        "native_token": "ETH"
    }
}

# Active network configuration
CURRENT_NETWORK = NETWORK_CONFIG.get(ACTIVE_NETWORK, NETWORK_CONFIG["BSC"])
MAINNET_RPC_URL = CURRENT_NETWORK["rpc_url"]
EXPLORER_API_URL = CURRENT_NETWORK["explorer_api"]
EXPLORER_API_KEY = CURRENT_NETWORK["api_key"]
CHAIN_ID = CURRENT_NETWORK["chain_id"]

# Legacy compatibility (for existing code)
ETHERSCAN_API_KEY = EXPLORER_API_KEY

# ============================================================
# DIRECTORY CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================
# MODEL HYPERPARAMETERS
# ============================================================

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

# ============================================================
# DATA SAMPLING CONFIGURATION
# ============================================================

# Data Sampling Configuration
START_BLOCK = int(os.getenv("START_BLOCK", 35000000))
END_BLOCK = os.getenv("END_BLOCK", "latest")
MAX_ADDRESSES = int(os.getenv("MAX_ADDRESSES", 1000))

# ============================================================
# API RATE LIMITING
# ============================================================

# API Rate Limiting
ETHERSCAN_MAX_REQUESTS_PER_SEC = 5
ETHERSCAN_BACKOFF_TIME = 0.2  # seconds

# ============================================================
# NETWORK GRAPH CONFIGURATION
# ============================================================

# Network Graph Configuration
BETWEENNESS_SAMPLE_SIZE = 1000  # Sample size for large graphs
PAGERANK_ALPHA = 0.85
PAGERANK_MAX_ITER = 100

# Cross-Validation Configuration
N_SPLITS_CV = 5

# Feature Engineering Constants
LOG_TRANSFORM_FEATURES = ["value_eth", "gas_price", "gas_used"]

# ============================================================
# SCAM ADDRESS SOURCES (Network-specific)
# ============================================================

# Network-specific scam sources
SCAM_ADDRESS_SOURCES = {
    "BSC": [
        "https://api.bscscan.com/api?module=account&action=tokentx&apikey=" + str(EXPLORER_API_KEY or ""),
    ],
    "OPBNB": [
        "https://api-opbnb.bscscan.com/api?module=account&action=tokentx&apikey=" + str(EXPLORER_API_KEY or ""),
    ],
    "ETH": [
        f"https://api.etherscan.io/v2/api?chainid=1&module=account&action=tokentx&apikey=" + str(EXPLORER_API_KEY or ""),
        "https://xblock.pro/api/phishing-addresses",
    ]
}

# Active scam sources
ACTIVE_SCAM_SOURCES = SCAM_ADDRESS_SOURCES.get(ACTIVE_NETWORK, SCAM_ADDRESS_SOURCES["BSC"])

# ============================================================
# LOGGING CONFIGURATION
# ============================================================

# Logging Configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

def validate_config():
    """Validate that required configuration is present."""
    if not MAINNET_RPC_URL:
        raise ValueError(f"RPC URL not set for {ACTIVE_NETWORK} in .env file")
    if not EXPLORER_API_KEY:
        raise ValueError(f"API KEY not set for {ACTIVE_NETWORK} in .env file")
    
    print("✅ Configuration validated successfully")
    print(f"   - Active Network: {CURRENT_NETWORK['name']} (Chain ID: {CHAIN_ID})")
    print(f"   - Native Token: {CURRENT_NETWORK['native_token']}")
    print(f"   - RPC URL: {MAINNET_RPC_URL[:50]}...")
    print(f"   - Explorer API: {EXPLORER_API_URL}")
    print(f"   - Data directory: {DATA_DIR}")
    print(f"   - Model directory: {MODEL_DIR}")
    print(f"   - Random state: {RANDOM_STATE}")

if __name__ == "__main__":
    validate_config()
