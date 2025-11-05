# Exploratory Data Analysis (EDA) - Ethereum Fraud Detection

## Overview
This notebook performs exploratory data analysis on Ethereum transaction data for fraud detection.

## 1. Setup and Imports

```python
import sys
sys.path.append('../src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import project modules
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils import setup_logger

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

logger = setup_logger(__name__)
print("✅ Setup complete")
```

## 2. Load Data

```python
# Load raw transaction data
raw_file = RAW_DATA_DIR / "transactions_raw.csv"

if raw_file.exists():
    df = pd.read_csv(raw_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Loaded {len(df)} transactions")
    print(f"Columns: {df.columns.tolist()}")
else:
    print(f"❌ Data file not found: {raw_file}")
    print("Run: python src/main.py --fetch-data")
```

## 3. Basic Statistics

```python
# Dataset shape
print(f"Dataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Data types
print("\nData types:")
print(df.dtypes)
```

## 4. Class Distribution

```python
# Fraud vs Normal distribution
class_dist = df['is_fraud'].value_counts()
print(f"\nClass distribution:")
print(f"Normal: {class_dist[0]} ({100*class_dist[0]/len(df):.2f}%)")
print(f"Fraud: {class_dist[1]} ({100*class_dist[1]/len(df):.2f}%)")

# Plot
plt.figure(figsize=(8, 6))
class_dist.plot(kind='bar', color=['green', 'red'])
plt.title('Transaction Class Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Class (0=Normal, 1=Fraud)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```

## 5. Temporal Analysis

```python
# Transactions over time
df['date'] = df['timestamp'].dt.date

plt.figure(figsize=(14, 6))
df.groupby('date').size().plot(kind='line', linewidth=2)
plt.title('Transactions Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Number of Transactions')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Hour of day distribution
plt.figure(figsize=(12, 5))
df['hour'] = df['timestamp'].dt.hour
df['hour'].hist(bins=24, edgecolor='black')
plt.title('Transaction Distribution by Hour of Day', fontsize=14, fontweight='bold')
plt.xlabel('Hour of Day')
plt.ylabel('Count')
plt.xticks(range(0, 24))
plt.tight_layout()
plt.show()
```

## 6. Value Analysis

```python
# Transaction value distribution
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
df[df['value_eth'] > 0]['value_eth'].hist(bins=50, edgecolor='black')
plt.title('Value Distribution (ETH)', fontsize=12, fontweight='bold')
plt.xlabel('Value (ETH)')
plt.ylabel('Count')
plt.yscale('log')

plt.subplot(1, 2, 2)
df[df['value_eth'] > 0]['value_eth'].apply(np.log10).hist(bins=50, edgecolor='black')
plt.title('Log10(Value) Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Log10(Value)')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# Compare fraud vs normal values
print("\nValue statistics by class:")
print(df.groupby('is_fraud')['value_eth'].describe())
```

## 7. Gas Analysis

```python
# Gas price distribution
df['gas_price_gwei'] = df['gasPrice'] / 1e9

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
df['gas_price_gwei'].hist(bins=50, edgecolor='black')
plt.title('Gas Price Distribution (Gwei)', fontsize=12, fontweight='bold')
plt.xlabel('Gas Price (Gwei)')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
df.boxplot(column='gas_price_gwei', by='is_fraud')
plt.title('Gas Price by Class', fontsize=12, fontweight='bold')
plt.suptitle('')
plt.xlabel('Class (0=Normal, 1=Fraud)')
plt.ylabel('Gas Price (Gwei)')

plt.tight_layout()
plt.show()
```

## 8. Address Analysis

```python
# Most active addresses
print("Top 10 sending addresses:")
print(df['from'].value_counts().head(10))

print("\nTop 10 receiving addresses:")
print(df['to'].value_counts().head(10))

# Unique addresses
n_unique_from = df['from'].nunique()
n_unique_to = df['to'].nunique()
print(f"\nUnique addresses:")
print(f"Senders: {n_unique_from}")
print(f"Receivers: {n_unique_to}")
```

## 9. Correlation Analysis (if features exist)

```python
# Load processed features if available
feature_file = PROCESSED_DATA_DIR / "features.csv"

if feature_file.exists():
    df_features = pd.read_csv(feature_file)
    
    # Select numeric columns only
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    
    # Correlation matrix
    corr_matrix = df_features[numeric_cols].corr()
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Top correlations with target
    if 'is_fraud' in numeric_cols:
        fraud_corr = corr_matrix['is_fraud'].sort_values(ascending=False)
        print("\nTop 10 features correlated with fraud:")
        print(fraud_corr.head(11)[1:])  # Exclude self-correlation
else:
    print(f"Feature file not found. Run feature engineering first.")
```

## 10. Summary

```python
print("\n" + "="*60)
print("EDA SUMMARY")
print("="*60)
print(f"Total transactions: {len(df)}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Fraud rate: {100*df['is_fraud'].mean():.2f}%")
print(f"Unique senders: {df['from'].nunique()}")
print(f"Unique receivers: {df['to'].nunique()}")
print(f"Average value: {df['value_eth'].mean():.4f} ETH")
print(f"Median gas price: {df['gas_price_gwei'].median():.2f} Gwei")
print("="*60)
```
