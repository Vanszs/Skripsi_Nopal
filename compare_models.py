"""
Model Comparison Script
Compares XGBoost with other ML/DL algorithms for fraud detection:
- Random Forest
- Logistic Regression
- LSTM (Deep Learning)
- TCN (Temporal Convolutional Network)
- LightGBM

Author: Fraud Detection System
Date: November 2025
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical

# XGBoost and LightGBM
import xgboost as xgb
import lightgbm as lgb

# Utilities
import joblib
from datetime import datetime

print("="*80)
print(" "*25 + "🔬 MODEL COMPARISON")
print("="*80)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

print("\n📁 Loading dataset...")
data_path = Path('data/processed/ethereum_fraud_balanced.csv')
df = pd.read_csv(data_path)

print(f"Dataset shape: {df.shape}")
print(f"Fraud cases: {df['label'].sum()} ({df['label'].mean()*100:.2f}%)")

# Prepare features and labels
X = df.drop(columns=['label'])
y = df['label']

# Train-test split (80-20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain samples: {len(X_train):,}")
print(f"Test samples: {len(X_test):,}")

# Apply SMOTE to training data
print("\n⚖️  Applying SMOTE to balance training data...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"After SMOTE - Train samples: {len(X_train_balanced):,}")
print(f"  - Fraud: {y_train_balanced.sum():,}")
print(f"  - Normal: {len(y_train_balanced) - y_train_balanced.sum():,}")

# Normalize data for deep learning models
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Save scaler for later use
joblib.dump(scaler, 'models/scaler_comparison.pkl')

# ============================================================================
# 2. DEFINE EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model, X_test, y_test, model_name, training_time):
    """
    Comprehensive model evaluation
    """
    print(f"\n📊 Evaluating {model_name}...")
    
    # Predictions
    start_time = time.time()
    
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        # For models without predict_proba (like TCN custom)
        y_pred_proba = model.predict(X_test).ravel()
    
    y_pred = (y_pred_proba >= 0.5).astype(int)
    inference_time = time.time() - start_time
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    results = {
        'model_name': model_name,
        'metrics': {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc)
        },
        'confusion_matrix': cm.tolist(),
        'training_time': training_time,
        'inference_time': inference_time,
        'samples_per_second': len(X_test) / inference_time
    }
    
    print(f"  ✅ PR-AUC: {pr_auc:.4f}")
    print(f"  ✅ ROC-AUC: {roc_auc:.4f}")
    print(f"  ✅ F1-Score: {f1:.4f}")
    print(f"  ⏱️  Training time: {training_time:.2f}s")
    print(f"  ⏱️  Inference time: {inference_time:.2f}s")
    
    return results

# ============================================================================
# 3. TRAIN MODELS
# ============================================================================

all_results = []

# -----------------------------------------------------------------------------
# Model 1: XGBoost (Baseline - already trained)
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("1️⃣  XGBoost Classifier (Baseline)")
print("="*80)

start_time = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='aucpr',
    scale_pos_weight=1,  # Already balanced with SMOTE
    tree_method='hist',
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train_balanced, y_train_balanced,
    eval_set=[(X_test, y_test)],
    verbose=False
)
xgb_time = time.time() - start_time

xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost", xgb_time)
all_results.append(xgb_results)

# Save model
joblib.dump(xgb_model, 'models/comparison_xgboost.pkl')

# -----------------------------------------------------------------------------
# Model 2: LightGBM
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("2️⃣  LightGBM Classifier")
print("="*80)

start_time = time.time()
lgb_model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_model.fit(X_train_balanced, y_train_balanced)
lgb_time = time.time() - start_time

lgb_results = evaluate_model(lgb_model, X_test, y_test, "LightGBM", lgb_time)
all_results.append(lgb_results)

joblib.dump(lgb_model, 'models/comparison_lightgbm.pkl')

# -----------------------------------------------------------------------------
# Model 3: Random Forest
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("3️⃣  Random Forest Classifier")
print("="*80)

start_time = time.time()
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

rf_model.fit(X_train_balanced, y_train_balanced)
rf_time = time.time() - start_time

rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest", rf_time)
all_results.append(rf_results)

joblib.dump(rf_model, 'models/comparison_randomforest.pkl')

# -----------------------------------------------------------------------------
# Model 4: Logistic Regression
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("4️⃣  Logistic Regression")
print("="*80)

start_time = time.time()
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    n_jobs=-1,
    solver='saga'
)

lr_model.fit(X_train_scaled, y_train_balanced)
lr_time = time.time() - start_time

lr_results = evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression", lr_time)
all_results.append(lr_results)

joblib.dump(lr_model, 'models/comparison_logistic.pkl')

# -----------------------------------------------------------------------------
# Model 5: LSTM (Deep Learning)
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("5️⃣  LSTM Neural Network")
print("="*80)

# Reshape for LSTM (samples, timesteps, features)
X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

start_time = time.time()

lstm_model = models.Sequential([
    layers.LSTM(128, input_shape=(1, X_train_scaled.shape[1]), return_sequences=True),
    layers.Dropout(0.3),
    layers.LSTM(64, return_sequences=False),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

lstm_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['AUC', 'Precision', 'Recall']
)

# Early stopping
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=0
)

# Train
history = lstm_model.fit(
    X_train_lstm, y_train_balanced,
    epochs=50,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

lstm_time = time.time() - start_time

lstm_results = evaluate_model(lstm_model, X_test_lstm, y_test, "LSTM", lstm_time)
all_results.append(lstm_results)

lstm_model.save('models/comparison_lstm.h5')

# -----------------------------------------------------------------------------
# Model 6: TCN (Temporal Convolutional Network)
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("6️⃣  TCN (Temporal Convolutional Network)")
print("="*80)

# Build TCN architecture
def build_tcn_model(input_shape, num_filters=64, kernel_size=3, num_layers=3):
    """
    Build Temporal Convolutional Network
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # TCN layers with dilated convolutions
    for i in range(num_layers):
        dilation_rate = 2 ** i
        
        # Causal convolution
        conv = layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu'
        )(x)
        
        conv = layers.BatchNormalization()(conv)
        conv = layers.Dropout(0.3)(conv)
        
        # Residual connection
        if x.shape[-1] != num_filters:
            x = layers.Conv1D(filters=num_filters, kernel_size=1)(x)
        x = layers.Add()([x, conv])
    
    # Global pooling and output
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

start_time = time.time()

tcn_model = build_tcn_model(input_shape=(1, X_train_scaled.shape[1]))

tcn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['AUC', 'Precision', 'Recall']
)

# Train
history_tcn = tcn_model.fit(
    X_train_lstm, y_train_balanced,
    epochs=50,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

tcn_time = time.time() - start_time

tcn_results = evaluate_model(tcn_model, X_test_lstm, y_test, "TCN", tcn_time)
all_results.append(tcn_results)

tcn_model.save('models/comparison_tcn.h5')

# ============================================================================
# 4. COMPARE RESULTS
# ============================================================================

print("\n" + "="*80)
print(" "*25 + "📊 COMPARISON RESULTS")
print("="*80)

# Create comparison dataframe
comparison_df = pd.DataFrame([
    {
        'Model': r['model_name'],
        'PR-AUC': r['metrics']['pr_auc'],
        'ROC-AUC': r['metrics']['roc_auc'],
        'Precision': r['metrics']['precision'],
        'Recall': r['metrics']['recall'],
        'F1-Score': r['metrics']['f1'],
        'Train Time (s)': r['training_time'],
        'Inference Time (s)': r['inference_time'],
        'Samples/sec': r['samples_per_second']
    }
    for r in all_results
])

# Sort by PR-AUC (most important for imbalanced data)
comparison_df = comparison_df.sort_values('PR-AUC', ascending=False).reset_index(drop=True)

print("\n📈 Performance Metrics (sorted by PR-AUC):")
print(comparison_df.to_string(index=False))

# Find best model
best_model = comparison_df.iloc[0]
print(f"\n🏆 BEST MODEL: {best_model['Model']}")
print(f"   PR-AUC: {best_model['PR-AUC']:.4f}")
print(f"   ROC-AUC: {best_model['ROC-AUC']:.4f}")
print(f"   F1-Score: {best_model['F1-Score']:.4f}")

# ============================================================================
# 5. SAVE RESULTS
# ============================================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save detailed results
output = {
    'timestamp': timestamp,
    'dataset': {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_samples_after_smote': len(X_train_balanced),
        'features': len(X.columns),
        'fraud_rate': float(y.mean())
    },
    'models': all_results,
    'comparison': comparison_df.to_dict('records')
}

results_path = Path(f'models/model_comparison_{timestamp}.json')
with open(results_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n💾 Results saved to: {results_path}")

# Save comparison CSV
csv_path = Path('models/model_comparison.csv')
comparison_df.to_csv(csv_path, index=False)
print(f"💾 Comparison CSV saved to: {csv_path}")

# Save detailed metrics for each model
for result in all_results:
    model_name = result['model_name'].replace(' ', '_').lower()
    model_path = Path(f'models/metrics_{model_name}.json')
    with open(model_path, 'w') as f:
        json.dump(result, f, indent=2)

print("\n" + "="*80)
print("✅ Model comparison complete!")
print("="*80)
