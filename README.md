# 🔍 Ethereum Fraud Detection System

**Deteksi Anomali Transaksi pada Jaringan Ethereum Menggunakan Algoritma XGBoost dan Explainable AI untuk Pencegahan Aktivitas Fraud dan Scam**

## 📋 Overview

Sistem deteksi fraud pada Ethereum Mainnet menggunakan:
- **XGBoost** untuk klasifikasi transaksi (fraud/normal)
- **SHAP** untuk explainability (Explainable AI)
- **NetworkX** untuk analisis graf transaksi
- **Time-Series Cross-Validation** untuk evaluasi robust

## 🏗️ Architecture

```
Ethereum Mainnet → Data Fetching → Feature Engineering → Graph Analysis
                                         ↓
                                  Imbalance Handling
                                         ↓
                         XGBoost Training (Time-Series CV)
                                         ↓
                              SHAP Explanations
                                         ↓
                                   Evaluation
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and navigate to project
cd Skripsi_Nopal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # fish: source venv/bin/activate.fish

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Copy example and edit with your API keys
cp .env.example .env
nano .env  # Add your Alchemy and Etherscan API keys
```

### 3. Run Pipeline

```bash
# Full pipeline: fetch → train → evaluate
python src/main.py --fetch-data --train --evaluate

# Or step by step:
python src/main.py --fetch-data              # Fetch data only
python src/main.py --train                   # Train model only
python src/main.py --evaluate                # Evaluate only
```

## 📊 Features

### Transaction Features
- **Temporal**: hour_of_day, day_of_week, tx_interval_mean
- **Value**: value_eth, log_value, value_to_gas_ratio
- **Gas**: gas_price, gas_used, gas_price_ratio
- **Account**: unique_receivers, total_tx_24h, tx_success_rate

### Network Graph Features
- **Centrality**: degree_centrality, betweenness_centrality, pagerank
- **Structure**: clustering_coefficient, community_id
- **Risk**: shortest_path_to_known_scam

## 📁 Project Structure

```
Skripsi_Nopal/
├── .env                      # API credentials (DO NOT COMMIT)
├── requirements.txt          # Python dependencies
├── README.md                # This file
│
├── data/
│   ├── raw/                 # Raw transaction data
│   └── processed/           # Engineered features
│
├── src/
│   ├── config.py            # Load environment variables
│   ├── fetch_transactions.py # Etherscan/Web3 data fetching
│   ├── feature_engineering.py # Feature extraction
│   ├── network_graph.py     # Graph analysis (NetworkX)
│   ├── imbalance_handler.py # ADASYN/SMOTE
│   ├── model_train.py       # XGBoost training
│   ├── model_explain.py     # SHAP explainability
│   ├── evaluate.py          # Metrics calculation
│   ├── utils.py             # Helper functions
│   └── main.py              # Main pipeline
│
├── models/                  # Trained models & results
├── logs/                    # Pipeline logs
└── notebooks/               # Jupyter notebooks for EDA
```

## 🎓 Academic Context

Proyek ini adalah skripsi yang memerlukan:
- ✅ Reproducibility (random seeds, versioning)
- ✅ Time-Series Cross-Validation (prevent data leakage)
- ✅ Explainable AI (SHAP values)
- ✅ Network Analysis (transaction graphs)
- ✅ Comprehensive logging & documentation

## 📈 Expected Outputs

1. **Trained Model**: `models/xgb_fraud_YYYYMMDD_HHMMSS.json`
2. **Evaluation Metrics**: `models/results_YYYYMMDD_HHMMSS.json`
3. **SHAP Plots**: `models/shap_summary_YYYYMMDD_HHMMSS.png`
4. **Feature Importance**: `models/feature_importance_YYYYMMDD_HHMMSS.png`
5. **Network Graphs**: `models/network_analysis_YYYYMMDD_HHMMSS.png`

## 🔧 Troubleshooting

**API Rate Limits**: Etherscan allows max 5 req/sec. Script includes automatic backoff.

**Memory Issues**: Mainnet data is large. Adjust `MAX_ADDRESSES` and block range in `.env`.

**Graph Computation**: For large graphs (>100k nodes), betweenness centrality uses sampling (k=1000).

## 📚 References

- XGBoost: https://xgboost.readthedocs.io/
- SHAP: https://shap.readthedocs.io/
- NetworkX: https://networkx.org/
- Etherscan API: https://docs.etherscan.io/

## 📝 License

Academic project for thesis purposes.
