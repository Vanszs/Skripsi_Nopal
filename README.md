# 🔍 opBNB Fraud Detection System

**Deteksi Wallet Drainer & Scam pada opBNB Mainnet Menggunakan XGBoost + SHAP + Network Graph Analysis**

## 📋 Overview

Sistem deteksi fraud pada **opBNB Mainnet (Chain ID: 204)** - BNB Smart Chain Layer 2:
- **XGBoost** untuk klasifikasi transaksi (fraud/normal)  
- **SHAP** untuk explainability (Explainable AI)  
- **Wallet Drainer Detection** dengan 5 pola scam detection  
- **Network Graph Analysis** untuk analisis graf transaksi  
- **Time-Series Cross-Validation** untuk evaluasi robust  

**Why opBNB?**
- 💰 Gas fee ~$0.001 (99% lebih murah dari Ethereum)
- ⚡ Block time 1 detik (12x lebih cepat)
- 🚀 Throughput 4000+ TPS (266x lebih tinggi)
- 📈 Dataset besar untuk ML research

---

## 🚀 Quick Start

### 1. Setup Environment
```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure API Keys
```powershell
# Copy template dan edit
copy .env.example .env
notepad .env  # Add your API keys
```

Required variables:
```properties
OPBNB_RPC_URL=https://opbnb-mainnet.nodereal.io/v1/YOUR_KEY
BSCSCAN_API_KEY=YOUR_BSCSCAN_KEY
SNAPSHOT_MODE=recent_blocks
TARGET_TX_COUNT=5000
```

### 3. Test Connection
```powershell
python src/utils/test_connection.py
```

### 4. Run Full Pipeline
```powershell
# Fetch data → Feature engineering → Train → Evaluate
python src/main.py --fetch-data --train --evaluate

# Or with Time-Series CV
python src/main.py --fetch-data --train --time-series-cv --evaluate
```

### 5. Open Jupyter Notebook
```powershell
jupyter notebook notebooks/02_Data_Analysis_Visualization.ipynb
```

---

## 🎯 Features

### Wallet Drainer Detection (5 Patterns)
1. **Burst Draining** - >5 transaksi dalam 60 detik
2. **High-Value Low-Gas** - Maksimalkan profit, minimalkan cost
3. **Sequential Transfers** - 80%+ transfer ke alamat berbeda
4. **Approval Scam** - Zero-value approval → high-value transfer
5. **Nonce Jumps** - Manipulasi nonce >5 dalam window

### Transaction Features (30 Features Total)
- **Temporal**: `hour_of_day`, `day_of_week`, `tx_interval_mean`
- **Value**: `value_eth`, `log_value`, `value_to_gas_ratio`
- **Gas**: `gas_price`, `gas_used`, `gas_price_ratio`
- **Account Behavior**: `unique_receivers`, `total_tx_24h`, `burst_activity_flag`
- **Network Graph**: `degree_centrality`, `betweenness_centrality`, `pagerank_score`

---

## 📁 Project Structure

```
Skripsi_Nopal/
├── .env                      # API credentials (create from .env.example)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── data/
│   ├── raw/                  # Raw transactions + fraud labels
│   │   └── transactions_raw.csv
│   └── processed/            # Engineered features
│       └── features.csv
│
├── src/
│   ├── config.py             # Load environment variables
│   ├── fetch_transactions.py # opBNB data fetching + drainer detection
│   ├── feature_engineering.py # Feature extraction (30 features)
│   ├── network_graph.py      # NetworkX graph analysis
│   ├── imbalance_handler.py  # ADASYN oversampling
│   ├── model_train.py        # XGBoost training + Time-Series CV
│   ├── model_explain.py      # SHAP explainability
│   ├── evaluate.py           # Metrics (PR-AUC, F1, Confusion Matrix)
│   ├── main.py               # Pipeline orchestrator
│   └── utils/
│       └── test_connection.py # Network connectivity test
│
├── models/                   # Trained models + results
│   ├── xgb_fraud_*.json
│   ├── params_*.json
│   ├── evaluation_results_*.json
│   └── viz_*.png
│
├── notebooks/
│   └── 02_Data_Analysis_Visualization.ipynb  # Comprehensive EDA
│
└── logs/                     # Pipeline execution logs
```

---

## 🔬 Current Dataset Stats

```
Total Transactions: 5,000
Fraud Rate: 6.3% (315 transactions)
Drainer Detection: 23.8% (1,190 suspected drainers)
Network: opBNB Mainnet (Chain ID: 204)
Date Range: Recent 500 blocks (~8 minutes)
```

**Model Performance:**
- **PR-AUC**: 0.9891 ± 0.0131 (5-fold Time-Series CV)
- **F1-Score**: 0.9096 ± 0.0736
- **Validation PR-AUC**: 1.0000 (perfect on validation set)

---

## 🎓 Academic Context (Skripsi)

### Reproducibility Requirements
- ✅ All random seeds set (`random_state=42`)
- ✅ Time-Series Cross-Validation (prevent data leakage)
- ✅ Timestamp-based splitting (no stratification)
- ✅ Comprehensive logging (`logs/pipeline_*.log`)
- ✅ Model versioning (timestamped outputs)

### Novel Contributions
1. **opBNB Focus** - First fraud detection research on BNB Layer 2
2. **Wallet Drainer Patterns** - 5 specific scam detection methods
3. **Snapshot Strategy** - Efficient recent transaction sampling
4. **SHAP Integration** - Complete explainability pipeline

### Expected Outputs
1. **Trained Model**: `models/xgb_fraud_YYYYMMDD_HHMMSS.json`
2. **Hyperparameters**: `models/params_YYYYMMDD_HHMMSS.json`
3. **Evaluation Metrics**: `models/evaluation_results_YYYYMMDD_HHMMSS.json`
4. **SHAP Plots**: `models/viz_09_shap_summary_bar.png`
5. **Feature Importance**: `models/viz_08_feature_importance_xgboost.png`
6. **Confusion Matrix**: `models/confusion_matrix_YYYYMMDD_HHMMSS.png`

---

## 🔧 Troubleshooting

### Common Issues

**ModuleNotFoundError**
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**API Rate Limit**  
Get free API key: https://bscscan.com/myapikey  
Update `BSCSCAN_API_KEY` in `.env`

**RPC Connection Failed**  
Try alternative: `OPBNB_RPC_URL=https://opbnb-mainnet-rpc.bnbchain.org`

**Out of Memory**  
Reduce dataset: `TARGET_TX_COUNT=1000` and `MAX_BLOCKS_TO_SCAN=100` in `.env`

**Unicode Errors in Console**  
Safe to ignore - data processing continues normally

---

## 📚 References

- **opBNB Docs**: https://docs.bnbchain.org/opbnb-docs/
- **BscScan opBNB**: https://opbnb.bscscan.com/
- **XGBoost**: https://xgboost.readthedocs.io/
- **SHAP**: https://shap.readthedocs.io/
- **NetworkX**: https://networkx.org/

---

## 📝 License

Academic research project for thesis purposes.

**Last Updated:** November 9, 2025  
**Network:** opBNB Mainnet (Chain ID: 204)  
**Status:** ✅ Production Ready
