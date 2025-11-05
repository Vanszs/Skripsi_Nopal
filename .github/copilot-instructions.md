# Ethereum Fraud Detection System - AI Agent Guide

## 🎯 Project Mission
Building an **end-to-end ML pipeline** for detecting fraudulent/scam transactions on Ethereum Mainnet using **XGBoost** with **SHAP explainability** and **Network Graph Analysis**. This is an academic research project (skripsi) requiring reproducibility, documentation, and scientific rigor.

## 🏗️ Architecture Overview

### Data Pipeline Flow
```
Ethereum Mainnet (Alchemy/Etherscan) 
  → Raw Transactions (data/raw/) 
  → Feature Engineering (temporal, gas, account behavior, network graph)
  → Imbalance Handling (ADASYN/SMOTE)
  → XGBoost Training (models/) with Time-Series CV
  → SHAP Explanations
  → Evaluation Metrics & Visualizations
```

### Core Components
- **`src/fetch_transactions.py`**: Web3.py + Etherscan API integration for Ethereum Mainnet
- **`src/feature_engineering.py`**: Transform raw txs → 15+ numerical features (gas ratios, temporal patterns, account metrics)
- **`src/network_graph.py`**: NetworkX-based graph analysis (centrality, clustering, shortest paths)
- **`src/imbalance_handler.py`**: ADASYN oversampling (fraud class is minority ~1-5%)
- **`src/model_train.py`**: XGBoost classifier with `scale_pos_weight` and time-series cross-validation
- **`src/model_explain.py`**: SHAP TreeExplainer for per-prediction and global feature importance
- **`src/evaluate.py`**: Precision/Recall/F1/AUC-PR metrics (optimized for imbalanced data)

## 🔧 Development Patterns

### Environment Setup
Always load credentials from `.env` file (NEVER hardcode):
```python
from dotenv import load_dotenv
import os

load_dotenv()
MAINNET_RPC_URL = os.getenv("MAINNET_RPC_URL")  # Alchemy/Infura mainnet URL
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
```

### Web3.py Transaction Fetching (Mainnet)
Use batch requests with rate limit handling (Etherscan: max 5 req/sec):
```python
from web3 import Web3
import time
import requests
import logging

w3 = Web3(Web3.HTTPProvider(MAINNET_RPC_URL))

def fetch_tx_batch(address, start_block, end_block):
    """Fetch with exponential backoff on rate limit"""
    try:
        response = requests.get(
            f"https://api.etherscan.io/api?module=account&action=txlist"
            f"&address={address}&startblock={start_block}&endblock={end_block}"
            f"&apikey={ETHERSCAN_API_KEY}"
        )
        if response.status_code == 429:
            time.sleep(0.2)  # Rate limit backoff
            return fetch_tx_batch(address, start_block, end_block)
        return response.json()['result']
    except Exception as e:
        logging.error(f"Fetch error: {e}")
        return []
```

### Feature Engineering Conventions
All features must be **numeric** and **reproducible**. Key feature categories:

**Temporal Features:**
- `hour_of_day`, `day_of_week`, `tx_interval_mean`, `time_since_last_tx`

**Value Features:**
- `value_eth`, `log_value`, `value_to_gas_ratio`

**Gas Features:**
- `gas_price`, `gas_used`, `gas_price_ratio = gas_price / median_gas_block`

**Account Behavior:**
- `unique_receivers`, `total_tx_24h`, `tx_success_rate`, `burst_activity_flag`

**Network Graph Features (from src/network_graph.py):**
- `degree_centrality`, `betweenness_centrality`, `pagerank_score`
- `clustering_coefficient`, `avg_neighbor_degree`
- `shortest_path_to_known_scam`, `community_id`

Always use `np.log1p()` for value transformations to handle zero values.

### XGBoost Training Pattern
Use these hyperparameters as baseline (from research best practices):
```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="aucpr",  # PR-AUC is critical for imbalanced data
    scale_pos_weight=ratio_majority_minority,  # Calculate from data
    tree_method="hist",  # Faster for large datasets
    early_stopping_rounds=10,
    random_state=42  # Reproducibility requirement
)

model.fit(
    X_train, y_train, 
    eval_set=[(X_test, y_test)],
    verbose=True
)
```

### SHAP Explainability Integration
Always use `TreeExplainer` for XGBoost (optimized for tree models):
```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global explanation (save for thesis)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig("models/shap_summary.png", dpi=300, bbox_inches='tight')

# Local explanation for single transaction
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0],
    feature_names=X_test.columns.tolist()
))
```

### Network Graph Analysis Pattern
Use NetworkX to extract graph-based features from transaction networks:
```python
import networkx as nx
import pandas as pd

def build_transaction_graph(df):
    """Build directed graph from transactions"""
    G = nx.DiGraph()
    
    for _, row in df.iterrows():
        G.add_edge(
            row['from'], 
            row['to'], 
            weight=row['value_eth'],
            timestamp=row['timestamp']
        )
    
    return G

def extract_graph_features(G, addresses):
    """Extract centrality and community features"""
    # Centrality measures
    degree_cent = nx.degree_centrality(G)
    between_cent = nx.betweenness_centrality(G, k=1000)  # Sample for large graphs
    pagerank = nx.pagerank(G, alpha=0.85)
    
    # Clustering coefficient (undirected version)
    G_undirected = G.to_undirected()
    clustering = nx.clustering(G_undirected)
    
    # Community detection (Louvain algorithm recommended)
    from networkx.algorithms import community
    communities = community.greedy_modularity_communities(G_undirected)
    
    # Map addresses to features
    features = []
    for addr in addresses:
        features.append({
            'address': addr,
            'degree_centrality': degree_cent.get(addr, 0),
            'betweenness_centrality': between_cent.get(addr, 0),
            'pagerank_score': pagerank.get(addr, 0),
            'clustering_coefficient': clustering.get(addr, 0),
            'community_id': _get_community_id(addr, communities)
        })
    
    return pd.DataFrame(features)
```

### Time-Series Cross-Validation Strategy
For temporal data, use **TimeSeriesSplit** to prevent data leakage:
```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

def time_series_cv_evaluate(X, y, timestamps, n_splits=5):
    """
    Perform time-series cross-validation.
    CRITICAL: Sort data by timestamp before splitting!
    """
    # Sort by timestamp
    sort_idx = np.argsort(timestamps)
    X_sorted = X.iloc[sort_idx]
    y_sorted = y.iloc[sort_idx]
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_scores = {
        'pr_auc': [],
        'f1': [],
        'recall': [],
        'precision': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_sorted)):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")
        
        X_train, X_test = X_sorted.iloc[train_idx], X_sorted.iloc[test_idx]
        y_train, y_test = y_sorted.iloc[train_idx], y_sorted.iloc[test_idx]
        
        # Apply ADASYN only to training set
        from imblearn.over_sampling import ADASYN
        adasyn = ADASYN(random_state=42)
        X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)
        
        # Train model
        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
            tree_method="hist",
            eval_metric="aucpr",
            early_stopping_rounds=10,
            random_state=42
        )
        
        model.fit(
            X_train_res, y_train_res,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        from sklearn.metrics import precision_recall_curve, auc, f1_score, recall_score, precision_score
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        cv_scores['pr_auc'].append(pr_auc)
        cv_scores['f1'].append(f1_score(y_test, y_pred))
        cv_scores['recall'].append(recall_score(y_test, y_pred))
        cv_scores['precision'].append(precision_score(y_test, y_pred))
        
        print(f"PR-AUC: {pr_auc:.4f}, F1: {cv_scores['f1'][-1]:.4f}")
    
    # Print summary
    print("\n=== Cross-Validation Summary ===")
    for metric, scores in cv_scores.items():
        print(f"{metric.upper()}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    return cv_scores
```

### Evaluation Metrics Priority
For **imbalanced fraud detection**, prioritize these metrics:
1. **PR-AUC** (Precision-Recall AUC) - more informative than ROC-AUC
2. **Recall@0.8_Precision** - operational threshold for alerts
3. **F1-Score** - harmonic mean balance
4. **Confusion Matrix** - false positive analysis

```python
from sklearn.metrics import precision_recall_curve, auc, classification_report

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

print(f"PR-AUC: {pr_auc:.4f}")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
```

## 🐛 Common Pitfalls

### Data Leakage Prevention (CRITICAL for Time-Series)
- **NEVER** include future information in features (e.g., `tx_count_next_24h`)
- **ALWAYS** sort by timestamp before splitting: `df.sort_values('timestamp', inplace=True)`
- Split train/test **BEFORE** SMOTE/ADASYN (only oversample training set)
- Use `TimeSeriesSplit` instead of `KFold` for cross-validation
- DO NOT use `stratify=y` with time-series data (breaks temporal order)
- Graph features must only use edges/nodes from training period

### Gas Price Normalization
Ethereum gas prices fluctuate wildly. Always normalize by block median:
```python
df['gas_price_ratio'] = df['gasPrice'] / df.groupby('blockNumber')['gasPrice'].transform('median')
```

### Phishing Address Labeling (Mainnet)
Pull known scam addresses from:
- Etherscan labeled addresses API: `https://api.etherscan.io/api?module=account&action=txlist&address={scam_address}`
- XBlock Phishing dataset (https://xblock.pro)
- EtherScamDB (backup source)
- Chainabuse.com reports

Merge labels with `pd.merge(df, scam_addresses, left_on='to', right_on='address')`

**IMPORTANT**: Mainnet has significantly more data than testnet. Consider:
- Sampling strategies (by time window or address clusters)
- Using recent blocks (last 6 months) for manageable dataset size
- Caching aggressively to avoid re-fetching

## 📊 Output Requirements (for Thesis)

All outputs must be saved with timestamp for reproducibility:
```python
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save trained model
model.save_model(f"models/xgb_fraud_{timestamp}.json")

# Save evaluation results
results = {
    'pr_auc': pr_auc,
    'f1_score': f1,
    'timestamp': timestamp,
    'hyperparameters': model.get_params()
}
with open(f"models/results_{timestamp}.json", 'w') as f:
    json.dump(results, f, indent=2)

# Save feature importance plot
plt.savefig(f"models/feature_importance_{timestamp}.png", dpi=300)
```

## 🚀 Quick Start Commands

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # or venv/bin/activate.fish for fish shell
pip install -r requirements.txt

# Run full pipeline
python src/main.py --fetch-data --train --evaluate

# Run only training (with cached data)
python src/main.py --train --data-path data/processed/features.csv

# Generate SHAP explanations for test set
python src/model_explain.py --model models/xgb_fraud.json --data data/processed/features.csv
```

## 🔐 Security & Best Practices

- Store API keys in `.env` file (add to `.gitignore`)
- Cache fetched transactions to avoid redundant API calls (`data/raw/tx_cache.pkl`)
- Use `try/except` with exponential backoff for all external API calls
- Log all pipeline steps to `logs/pipeline_{timestamp}.log`
- Set `random_state=42` everywhere for reproducibility
- Document all hyperparameter choices in comments (for thesis methodology section)

## 📝 Code Style Conventions

- Use `snake_case` for functions and variables
- Prefix private functions with `_` (e.g., `_calculate_gas_ratio()`)
- Add docstrings following NumPy style for all functions:
  ```python
  def fetch_transactions(address: str, start_block: int) -> pd.DataFrame:
      """
      Fetch transaction history from Etherscan API.
      
      Parameters
      ----------
      address : str
          Ethereum address (0x prefixed)
      start_block : int
          Starting block number
      
      Returns
      -------
      pd.DataFrame
          Transaction dataframe with columns: hash, from, to, value, gasPrice, etc.
      """
  ```
- Include references to papers/documentation in comments for algorithm choices

## 🎓 Academic Context

This is a **computer science thesis (skripsi)** requiring:
- Reproducible experiments (save all random seeds, versions, datasets)
- Detailed logging (track training time, iterations, memory usage)
- Ablation studies (compare with/without ADASYN, different hyperparameters)
- Comparison baselines (Random Forest, Logistic Regression for benchmarking)
- Statistical significance testing (use cross-validation, report std deviation)

When implementing new features, always consider: "How will I explain this in the methodology chapter?"
