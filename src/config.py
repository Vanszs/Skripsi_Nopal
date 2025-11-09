"""
Configuration loader for opBNB fraud detection system.
opBNB-only configuration for transaction fraud detection.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# ============================================================
# opBNB NETWORK CONFIGURATION
# ============================================================

# opBNB Mainnet Configuration
OPBNB_RPC_URL = os.getenv("OPBNB_RPC_URL")
OPBNB_API_KEY = os.getenv("OPBNB_API_KEY")

# Network constants
CHAIN_ID = 204
NETWORK_NAME = "opBNB Mainnet"
NATIVE_TOKEN = "BNB"
EXPLORER_API_URL = "https://api-opbnb.bscscan.com/api"

# Aliases for backward compatibility
MAINNET_RPC_URL = OPBNB_RPC_URL
EXPLORER_API_KEY = OPBNB_API_KEY

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

# Snapshot Configuration for Recent Transactions
SNAPSHOT_MODE = os.getenv("SNAPSHOT_MODE", "recent_blocks")
TARGET_TX_COUNT = int(os.getenv("TARGET_TX_COUNT", 5000))
MAX_BLOCKS_TO_SCAN = int(os.getenv("MAX_BLOCKS_TO_SCAN", 500))

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
# SCAM ADDRESS SOURCES (opBNB-specific)
# ============================================================

# opBNB scam address sources
SCAM_ADDRESS_SOURCES = [
    f"https://api-opbnb.bscscan.com/api?module=account&action=tokentx&apikey={OPBNB_API_KEY}",
    "https://chainabuse.com/api/scams?chain=opbnb",  # Community-reported scams
]

# ============================================================
# LOGGING CONFIGURATION
# ============================================================

# Logging Configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

def validate_config():
    """Validate that required configuration is present."""
    if not MAINNET_RPC_URL:
        raise ValueError("opBNB RPC URL not set in .env file (OPBNB_RPC_URL)")
    if not EXPLORER_API_KEY:
        raise ValueError("opBNB API KEY not set in .env file (OPBNB_API_KEY)")
    
    print("✅ Configuration validated successfully")
    print(f"   - Network: {NETWORK_NAME} (Chain ID: {CHAIN_ID})")
    print(f"   - Native Token: {NATIVE_TOKEN}")
    print(f"   - RPC URL: {MAINNET_RPC_URL[:50]}...")
    print(f"   - Explorer API: {EXPLORER_API_URL}")
    print(f"   - Snapshot Mode: {SNAPSHOT_MODE}")
    print(f"   - Target TX Count: {TARGET_TX_COUNT:,}")
    print(f"   - Max Blocks to Scan: {MAX_BLOCKS_TO_SCAN:,}")
    print(f"   - Data directory: {DATA_DIR}")
    print(f"   - Model directory: {MODEL_DIR}")
    print(f"   - Random state: {RANDOM_STATE}")

if __name__ == "__main__":
    validate_config()
