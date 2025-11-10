"""
End-to-End Training Pipeline for Ethereum Fraud Detection
Simplified script that runs complete training flow
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    precision_recall_curve, 
    auc,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import json

print("="*80)
print("ETHEREUM FRAUD DETECTION - END-TO-END TRAINING PIPELINE")
print("="*80)

# Step 1: Load processed dataset
print("\n[1/6] Loading dataset...")
data_path = Path('data/processed/ethereum_fraud_balanced.csv')

if not data_path.exists():
    print(f"ERROR: Dataset not found at {data_path}")
    print("Run: python process_ethereum_dataset.py first")
    sys.exit(1)

df = pd.read_csv(data_path)
print(f"   Loaded {len(df):,} samples")
print(f"   Features: {len(df.columns)-1}")

# Step 2: Prepare features and labels
print("\n[2/6] Preparing features and labels...")
X = df.drop(columns=['label'])
y = df['label']

print(f"   Features shape: {X.shape}")
print(f"   Fraud cases: {y.sum():,} ({y.mean()*100:.2f}%)")
print(f"   Normal cases: {(y==0).sum():,} ({(y==0).mean()*100:.2f}%)")

# Step 3: Split train/test
print("\n[3/6] Splitting train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"   Train: {len(X_train):,} samples")
print(f"   Test: {len(X_test):,} samples")

# Step 4: Handle class imbalance with SMOTE
print("\n[4/6] Applying SMOTE to balance training data...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"   Before SMOTE - Fraud: {y_train.sum():,}, Normal: {(y_train==0).sum():,}")
print(f"   After SMOTE  - Fraud: {y_train_res.sum():,}, Normal: {(y_train_res==0).sum():,}")

# Step 5: Train XGBoost model
print("\n[5/6] Training XGBoost model...")
print("   Hyperparameters:")
print(f"      n_estimators: 300")
print(f"      learning_rate: 0.05")
print(f"      max_depth: 6")
print(f"      eval_metric: aucpr")

# Calculate scale_pos_weight for original imbalance
scale_pos_weight = (y_train == 0).sum() / y_train.sum()

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='aucpr',
    scale_pos_weight=scale_pos_weight,
    tree_method='hist',
    early_stopping_rounds=10,
    random_state=42,
    verbosity=1
)

print("\n   Training in progress...")
model.fit(
    X_train_res, 
    y_train_res,
    eval_set=[(X_test, y_test)],
    verbose=False
)

print(f"   Training complete!")
print(f"   Best iteration: {model.best_iteration}")
print(f"   Best score: {model.best_score:.4f}")

# Step 6: Evaluate model
print("\n[6/6] Evaluating model...")

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Metrics
print("\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud'], digits=4))

# PR-AUC (better for imbalanced data)
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
print(f"\n   PR-AUC Score: {pr_auc:.4f}")

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"   ROC-AUC Score: {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n   Confusion Matrix:")
print(f"                 Predicted")
print(f"                 Normal  Fraud")
print(f"   Actual Normal  {cm[0,0]:5d}  {cm[0,1]:5d}")
print(f"   Actual Fraud   {cm[1,0]:5d}  {cm[1,1]:5d}")

# Feature Importance (Top 10)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n   Top 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"      {row['feature'][:40]:40s}: {row['importance']:.4f}")

# Step 7: Save model and results
print("\n[7/7] Saving model and results...")

# Create models directory
model_dir = Path('models')
model_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save model
model_path = model_dir / f'xgb_fraud_ethereum_{timestamp}.json'
model.save_model(model_path)
print(f"   Model saved: {model_path}")

# Save with joblib (alternative format)
joblib_path = model_dir / f'xgb_fraud_ethereum_{timestamp}.pkl'
joblib.dump(model, joblib_path)
print(f"   Model saved (joblib): {joblib_path}")

# Save evaluation results
results = {
    'timestamp': timestamp,
    'dataset': 'ethereum_fraud_balanced.csv',
    'samples': {
        'total': len(df),
        'train': len(X_train),
        'test': len(X_test),
        'train_after_smote': len(X_train_res)
    },
    'metrics': {
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred))
    },
    'confusion_matrix': cm.tolist(),
    'hyperparameters': {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 6,
        'scale_pos_weight': float(scale_pos_weight)
    },
    'feature_count': len(X.columns),
    'top_features': feature_importance.head(20).to_dict('records')
}

results_path = model_dir / f'results_{timestamp}.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"   Results saved: {results_path}")

# Save feature importance
importance_path = model_dir / f'feature_importance_{timestamp}.csv'
feature_importance.to_csv(importance_path, index=False)
print(f"   Feature importance saved: {importance_path}")

# Summary
print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\n   Model Performance:")
print(f"      PR-AUC:    {pr_auc:.4f}")
print(f"      ROC-AUC:   {roc_auc:.4f}")
print(f"      Precision: {precision_score(y_test, y_pred):.4f}")
print(f"      Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"      F1-Score:  {f1_score(y_test, y_pred):.4f}")

print(f"\n   Files Created:")
print(f"      1. {model_path.name}")
print(f"      2. {joblib_path.name}")
print(f"      3. {results_path.name}")
print(f"      4. {importance_path.name}")

print(f"\n   Next Steps:")
print(f"      1. Generate SHAP explanations")
print(f"      2. Visualize results in notebook")
print(f"      3. Test on new transactions")

print("\n" + "="*80)
