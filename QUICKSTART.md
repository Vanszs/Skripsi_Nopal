# 🚀 Quick Start Guide

## Ethereum Fraud Detection System

Panduan cepat untuk menjalankan proyek deteksi fraud pada Ethereum Mainnet.

---

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **Alchemy API Key** (get from https://www.alchemy.com/)
3. **Etherscan API Key** (get from https://etherscan.io/apis)
   - Proyek ini menggunakan Etherscan API V2 yang memerlukan parameter `chainid`
   - Free tier: 5 requests/second
4. **Git** (optional, for cloning)

---

## ⚡ Setup (5 menit)

### 1. Clone/Download Project

```bash
cd ~/Documents/Code/Skripsi_Nopal
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Fish shell)
source venv/bin/activate.fish

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Expected installation time: **2-3 minutes**

### 4. Configure API Keys

```bash
# Copy example env file
cp .env.example .env

# Edit .env file
nano .env  # or use your preferred editor
```

Add your API keys:
```env
MAINNET_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/YOUR_ALCHEMY_KEY
ETHERSCAN_API_KEY=YOUR_ETHERSCAN_API_KEY
```

---

## 🎯 Running the Pipeline

### Option A: Full Pipeline (Recommended for First Run)

```bash
python src/main.py --full-pipeline
```

This runs all stages:
1. ✅ Fetch data from Ethereum mainnet
2. ✅ Extract features (including network graph)
3. ✅ Train XGBoost model
4. ✅ Generate SHAP explanations
5. ✅ Evaluate model performance

**Estimated time**: 10-30 minutes (depending on data size)

### Option B: Stage-by-Stage

```bash
# Stage 1: Fetch Data
python src/main.py --fetch-data

# Stage 2: Feature Engineering
python src/main.py --feature-engineering

# Stage 3: Train Model (with time-series CV)
python src/main.py --train --time-series-cv

# Stage 4: SHAP Explanations
python src/main.py --explain

# Stage 5: Evaluation
python src/main.py --evaluate
```

### Option C: Skip Graph Features (Faster)

```bash
python src/main.py --full-pipeline --no-graph-features
```

Network graph analysis can be slow for large datasets. This skips it for faster execution.

---

## 📊 Checking Results

### Models
```bash
ls models/
# Output:
# xgb_fraud_20251105_143022.json
# params_20251105_143022.json
# evaluation_results_20251105_143022.json
# shap_summary_20251105_143022.png
# confusion_matrix_20251105_143022.png
```

### Logs
```bash
tail -f logs/pipeline_*.log
```

### Data
```bash
ls data/raw/
ls data/processed/
```

---

## 🧪 Testing Individual Components

### Test Configuration

```bash
python src/config.py
```

Expected output:
```
✅ Configuration validated successfully
   - Data directory: /path/to/data
   - Model directory: /path/to/models
   - Random state: 42
```

### Test Data Fetching

```bash
python src/fetch_transactions.py
```

### Test Feature Engineering

```bash
python src/feature_engineering.py
```

### Test Model Training

```bash
python src/model_train.py
```

---

## 📈 Exploring Data (Jupyter)

```bash
# Install Jupyter (if not already installed)
pip install jupyter

# Start Jupyter
jupyter notebook notebooks/
```

Open `01_EDA.ipynb` for exploratory data analysis.

---

## 🐛 Troubleshooting

### Problem: API Rate Limit Exceeded

**Solution**: Wait 1 minute and retry. The script automatically handles rate limiting.

### Problem: Out of Memory

**Solution**: 
- Reduce `MAX_ADDRESSES` in `.env`
- Use `--no-graph-features` flag
- Reduce block range in `.env`

### Problem: No Transactions Fetched

**Solution**:
- Check API keys in `.env`
- Verify RPC URL is correct
- Try different addresses (use popular contracts)

### Problem: Model Training Fails

**Solution**:
- Check if data has fraud labels (`is_fraud` column)
- Ensure processed features exist: `data/processed/features.csv`
- Check logs for specific error

---

## 📁 Project Structure Reference

```
Skripsi_Nopal/
├── .env                        # API credentials (YOU CREATE THIS)
├── .env.example               # Template
├── requirements.txt           # Dependencies
├── README.md                  # Full documentation
├── QUICKSTART.md             # This file
│
├── data/
│   ├── raw/                  # Raw transaction data
│   │   └── transactions_raw.csv
│   └── processed/            # Engineered features
│       └── features.csv
│
├── src/                      # Source code
│   ├── main.py              # Main pipeline
│   ├── config.py            # Configuration
│   ├── fetch_transactions.py
│   ├── feature_engineering.py
│   ├── network_graph.py
│   ├── model_train.py
│   ├── model_explain.py
│   └── evaluate.py
│
├── models/                   # Trained models & results
│   ├── xgb_fraud_*.json
│   ├── shap_summary_*.png
│   └── evaluation_results_*.json
│
├── logs/                     # Pipeline logs
└── notebooks/               # Jupyter notebooks
    └── 01_EDA.md
```

---

## 🎓 Next Steps

1. ✅ **Run full pipeline** to ensure everything works
2. 📊 **Explore results** in `models/` directory
3. 📈 **Analyze SHAP plots** for model interpretability
4. 📝 **Read evaluation metrics** for thesis writeup
5. 🔬 **Experiment** with different hyperparameters in `config.py`

---

## 💡 Tips for Thesis

### For Methodology Chapter

- Hyperparameters are in `models/params_*.json`
- Feature list is in `src/feature_engineering.py`
- CV results show model stability

### For Results Chapter

- Use plots from `models/` directory
- Copy metrics from `evaluation_results_*.json`
- SHAP plots show feature importance

### For Discussion Chapter

- Analyze top SHAP features
- Compare PR-AUC vs ROC-AUC (imbalanced data)
- Discuss time-series validation importance

---

## 📞 Need Help?

Check:
1. Logs in `logs/` directory
2. Documentation in README.md
3. Code comments in `src/` files

Good luck with your thesis! 🎉
