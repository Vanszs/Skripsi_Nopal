# 📚 Workspace Guide - opBNB Fraud Detection

**Last Updated:** November 9, 2025  
**Project:** opBNB Wallet Drainer & Fraud Detection  
**Status:** ✅ Production Ready

---

## 📁 Directory Structure

```
Skripsi_Nopal/
│
├── 📄 .env                          # Environment configuration (CREATE THIS!)
├── 📄 .env.example                  # Template for .env
├── 📄 .gitignore                    # Git ignore rules
├── 📄 README.md                     # Main documentation (START HERE)
├── 📄 WORKSPACE_GUIDE.md            # This file
├── 📄 requirements.txt              # Python dependencies
│
├── 📂 data/
│   ├── 📂 raw/                      # Raw blockchain data
│   │   └── transactions_raw.csv     # 5,000 transactions with fraud labels
│   └── 📂 processed/                # Engineered features
│       └── features.csv             # 30 features for ML
│
├── 📂 src/                          # Source code
│   ├── config.py                    # Configuration loader
│   ├── fetch_transactions.py        # Data fetching + drainer detection
│   ├── feature_engineering.py       # Feature extraction (30 features)
│   ├── network_graph.py             # NetworkX graph analysis
│   ├── imbalance_handler.py         # ADASYN oversampling
│   ├── model_train.py               # XGBoost training + Time-Series CV
│   ├── model_explain.py             # SHAP explainability
│   ├── evaluate.py                  # Performance metrics
│   ├── main.py                      # Pipeline orchestrator
│   └── 📂 utils/
│       └── test_connection.py       # Network connectivity test
│
├── 📂 models/                       # Trained models & outputs
│   ├── xgb_fraud_20251109_095333.json           # Trained XGBoost model
│   ├── params_20251109_095333.json              # Hyperparameters
│   ├── evaluation_results_20251109_095333.json  # Metrics
│   └── viz_*.png                                # Visualization outputs
│
├── 📂 notebooks/                    # Jupyter notebooks
│   ├── 01_EDA.md                    # Exploratory data analysis guide
│   ├── 02_Data_Analysis_Visualization.ipynb     # Main analysis notebook
│   └── README.md                    # Notebook documentation
│
├── 📂 logs/                         # Execution logs
│   └── pipeline_*.log               # Pipeline execution logs
│
└── 📂 venv/                         # Python virtual environment
    └── (Python packages)
```

---

## 🚀 Quick Commands

### Environment Setup
```powershell
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Data Pipeline
```powershell
# Test network connection
python src/utils/test_connection.py

# Fetch transactions (snapshot mode)
python src/fetch_transactions.py

# Full pipeline
python src/main.py --fetch-data --train --evaluate

# Time-Series Cross-Validation
python src/main.py --train --time-series-cv
```

### Analysis
```powershell
# Launch Jupyter
jupyter notebook notebooks/02_Data_Analysis_Visualization.ipynb

# Generate SHAP explanations
python src/model_explain.py --model models/xgb_fraud_*.json

# Evaluate model
python src/evaluate.py
```

---

## 📊 Current Dataset Status

**File:** `data/raw/transactions_raw.csv`

| Metric | Value |
|--------|-------|
| Total Transactions | 5,000 |
| Fraud Rate | 6.3% (315 transactions) |
| Drainer Detection | 23.8% (1,190 transactions) |
| Network | opBNB Mainnet (Chain ID: 204) |
| Blocks Scanned | ~500 blocks (~8 minutes) |
| Columns | 14 (includes `suspected_drainer`) |

**Wallet Drainer Patterns:**
1. ✅ Burst Draining (>5 tx in 60s)
2. ✅ High-Value Low-Gas (profit maximization)
3. ✅ Sequential Transfers (80%+ different recipients)
4. ✅ Approval Scam (zero-value → high-value)
5. ✅ Nonce Jumps (>5 manipulation)

---

## 🤖 Model Performance

**File:** `models/xgb_fraud_20251109_095333.json`

| Metric | Value |
|--------|-------|
| PR-AUC | 0.9891 ± 0.0131 |
| F1-Score | 0.9096 ± 0.0736 |
| Precision | Variable by fold |
| Recall | Variable by fold |
| Validation PR-AUC | 1.0000 (perfect) |

**Training Config:**
- Algorithm: XGBoost with ADASYN oversampling
- Cross-Validation: 5-fold Time-Series Split
- Features: 30 engineered features
- Hyperparameters: See `models/params_20251109_095333.json`

---

## 🎯 Key Files Explained

### Configuration Files

**`.env`** (YOU CREATE THIS)
```properties
# Required configuration
OPBNB_RPC_URL=https://opbnb-mainnet.nodereal.io/v1/YOUR_KEY
BSCSCAN_API_KEY=YOUR_BSCSCAN_KEY
SNAPSHOT_MODE=recent_blocks
TARGET_TX_COUNT=5000
MAX_BLOCKS_TO_SCAN=500
```

**`.env.example`**  
Template for creating `.env`. Copy and fill in your API keys.

### Source Code

**`src/main.py`** - Pipeline orchestrator
- Entry point for all pipeline stages
- Command-line interface
- Handles fetch → feature → train → explain → evaluate

**`src/fetch_transactions.py`** - Data acquisition
- Connects to opBNB RPC
- Fetches transactions via BscScan API
- Implements 5 wallet drainer detection patterns
- Labels fraud transactions

**`src/feature_engineering.py`** - Feature extraction
- Generates 30 features from raw transactions
- Temporal, value, gas, account behavior
- Handles missing data and outliers

**`src/network_graph.py`** - Graph analysis
- Builds directed transaction graph
- Computes centrality measures
- Community detection
- Risk propagation analysis

**`src/model_train.py`** - Model training
- XGBoost classifier with ADASYN
- Time-Series Cross-Validation
- Hyperparameter management
- Model persistence

**`src/model_explain.py`** - SHAP explainability
- TreeExplainer for XGBoost
- Global feature importance
- Local explanations
- Visualization generation

**`src/evaluate.py`** - Performance evaluation
- PR-AUC, ROC-AUC, F1-Score
- Confusion matrix
- Feature importance plots
- Results persistence

### Notebooks

**`notebooks/02_Data_Analysis_Visualization.ipynb`**  
Comprehensive analysis notebook with:
- Raw data exploration
- Wallet drainer pattern analysis
- Feature correlation analysis
- Network graph visualization
- Model performance analysis
- SHAP explainability plots
- Summary and conclusions

---

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Activate venv: `.\venv\Scripts\Activate.ps1` |
| API rate limit | Get free key: https://bscscan.com/myapikey |
| RPC connection failed | Try: `OPBNB_RPC_URL=https://opbnb-mainnet-rpc.bnbchain.org` |
| Out of memory | Reduce: `TARGET_TX_COUNT=1000` |
| Unicode errors | Safe to ignore - data still processes |

### Logs

Check logs for detailed error messages:
```powershell
Get-Content logs\pipeline_*.log | Select-Object -Last 50
```

---

## 📈 Development Workflow

### 1. Setup (First Time)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
notepad .env  # Add API keys
python src/utils/test_connection.py
```

### 2. Data Collection
```powershell
# Fetch fresh data
python src/fetch_transactions.py

# Or use pipeline
python src/main.py --fetch-data
```

### 3. Feature Engineering
```powershell
python src/feature_engineering.py

# Check features
python -c "import pandas as pd; print(pd.read_csv('data/processed/features.csv').columns.tolist())"
```

### 4. Model Training
```powershell
# Standard training
python src/main.py --train

# With Time-Series CV (recommended)
python src/main.py --train --time-series-cv
```

### 5. Analysis
```powershell
# Generate explanations
python src/model_explain.py

# Evaluate
python src/evaluate.py

# Visualize in Jupyter
jupyter notebook notebooks/02_Data_Analysis_Visualization.ipynb
```

### 6. Iteration
```powershell
# Full pipeline with new data
python src/main.py --fetch-data --train --time-series-cv --evaluate
```

---

## 🎓 For Thesis (Skripsi)

### File Organization for Thesis Chapters

**Chapter 1-2 (Introduction, Literature Review)**
- `README.md` - Project overview
- `notebooks/01_EDA.md` - Initial exploration

**Chapter 3 (Methodology)**
- `src/fetch_transactions.py` - Data collection methodology
- `src/feature_engineering.py` - Feature engineering process
- `models/params_*.json` - Hyperparameter documentation
- `.github/copilot-instructions.md` - Technical specifications

**Chapter 4 (Results)**
- `models/evaluation_results_*.json` - Performance metrics
- `models/viz_*.png` - All visualization outputs
- `notebooks/02_Data_Analysis_Visualization.ipynb` - Complete analysis

**Chapter 5 (Discussion, Conclusion)**
- Performance comparison with baselines
- SHAP interpretation for explainability
- Future work recommendations

### Key Outputs for Thesis

1. **Tables:**
   - Dataset statistics → From notebook Summary section
   - Model performance → `evaluation_results_*.json`
   - Feature importance → SHAP summary plots

2. **Figures:**
   - Network graph → `viz_06_network_centrality.png`
   - Confusion matrix → `confusion_matrix_*.png`
   - SHAP plots → `viz_09_shap_summary_bar.png`, `viz_10_shap_beeswarm.png`
   - Drainer analysis → `viz_00_drainer_analysis.png`

3. **Code Snippets:**
   - Use syntax highlighting from source files
   - Reference line numbers from GitHub repository

---

## 🔐 Security Checklist

- [ ] ✅ `.env` file created and configured
- [ ] ✅ `.env` added to `.gitignore`
- [ ] ✅ API keys never committed to Git
- [ ] ✅ Public RPC URLs only (no private keys)
- [ ] ⚠️ Get your own API keys (don't share)

---

## 📚 Additional Resources

- **Main README:** `README.md` - Comprehensive project documentation
- **Notebook Guide:** `notebooks/README.md` - Jupyter notebook instructions
- **Code Documentation:** Inline comments in all `.py` files
- **GitHub Copilot Instructions:** `.github/copilot-instructions.md`

---

## ✅ Workspace Health Check

Run this to verify everything is set up correctly:

```powershell
# 1. Check Python environment
python --version  # Should be 3.8+

# 2. Check packages
pip list | Select-String "pandas|numpy|xgboost|shap|web3"

# 3. Check data
Test-Path data\raw\transactions_raw.csv
Test-Path data\processed\features.csv

# 4. Check model
Test-Path models\xgb_fraud_*.json

# 5. Test connection
python src\utils\test_connection.py
```

Expected output:
```
✅ Python 3.x.x
✅ All packages installed
✅ Data files exist
✅ Model files exist
✅ Network connection OK
```

---

**Status:** 🟢 Workspace Clean & Organized  
**Ready for:** Production use and thesis writing  
**Last Cleanup:** November 9, 2025
