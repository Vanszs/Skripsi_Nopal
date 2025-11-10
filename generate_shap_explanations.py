"""
Generate SHAP Explanations for Trained Model
Visualize feature importance and decision-making process
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("SHAP EXPLAINABILITY ANALYSIS")
print("="*80)

# Step 1: Load trained model
print("\n[1/5] Loading trained model...")
model_dir = Path('models')
model_files = list(model_dir.glob('xgb_fraud_ethereum_*.pkl'))

if not model_files:
    print("ERROR: No trained model found!")
    print("Run: python train_end_to_end.py first")
    sys.exit(1)

latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
print(f"   Loading: {latest_model.name}")

model = joblib.load(latest_model)
print(f"   Model loaded successfully")

# Step 2: Load test data
print("\n[2/5] Loading test data...")
data_path = Path('data/processed/ethereum_fraud_balanced.csv')
df = pd.read_csv(data_path)

X = df.drop(columns=['label'])
y = df['label']

# Sample for SHAP (use smaller subset for speed)
sample_size = min(500, len(X))
X_sample = X.sample(sample_size, random_state=42)
y_sample = y.loc[X_sample.index]

print(f"   Using {sample_size} samples for SHAP analysis")
print(f"   Features: {X_sample.shape[1]}")

# Step 3: Create SHAP explainer
print("\n[3/5] Creating SHAP TreeExplainer...")
explainer = shap.TreeExplainer(model)
print("   Explainer created")

print("\n   Computing SHAP values...")
shap_values = explainer.shap_values(X_sample)
print("   SHAP values computed")

# Step 4: Generate visualizations
print("\n[4/5] Generating visualizations...")
output_dir = Path('models/shap_plots')
output_dir.mkdir(exist_ok=True)

# Plot 1: Summary Plot (Bar)
print("\n   [1/4] Feature Importance (Bar Plot)...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
plt.tight_layout()
bar_plot_path = output_dir / 'shap_summary_bar.png'
plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"         Saved: {bar_plot_path}")

# Plot 2: Summary Plot (Beeswarm)
print("   [2/4] Feature Impact (Beeswarm Plot)...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
plt.tight_layout()
beeswarm_path = output_dir / 'shap_summary_beeswarm.png'
plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"         Saved: {beeswarm_path}")

# Plot 3: Waterfall plot for a fraud case
print("   [3/4] Single Prediction Explanation (Waterfall)...")
fraud_idx = y_sample[y_sample == 1].index[0]
fraud_idx_pos = X_sample.index.get_loc(fraud_idx)

plt.figure(figsize=(10, 8))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[fraud_idx_pos],
        base_values=explainer.expected_value,
        data=X_sample.iloc[fraud_idx_pos],
        feature_names=X_sample.columns.tolist()
    ),
    show=False,
    max_display=15
)
plt.tight_layout()
waterfall_path = output_dir / 'shap_waterfall_fraud_example.png'
plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"         Saved: {waterfall_path}")

# Plot 4: Force plot for fraud case
print("   [4/4] Decision Force Plot...")
plt.figure(figsize=(20, 3))
shap.force_plot(
    explainer.expected_value,
    shap_values[fraud_idx_pos],
    X_sample.iloc[fraud_idx_pos],
    matplotlib=True,
    show=False
)
plt.tight_layout()
force_path = output_dir / 'shap_force_fraud_example.png'
plt.savefig(force_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"         Saved: {force_path}")

# Step 5: Save SHAP values
print("\n[5/5] Saving SHAP values...")

# Create summary dataframe
shap_df = pd.DataFrame(
    shap_values,
    columns=X_sample.columns
)
shap_df['actual_label'] = y_sample.values
shap_df['predicted_label'] = model.predict(X_sample)
shap_df['predicted_proba'] = model.predict_proba(X_sample)[:, 1]

shap_csv_path = output_dir / 'shap_values.csv'
shap_df.to_csv(shap_csv_path, index=False)
print(f"   SHAP values saved: {shap_csv_path}")

# Feature importance summary
feature_importance = pd.DataFrame({
    'feature': X_sample.columns,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

importance_path = output_dir / 'shap_feature_importance.csv'
feature_importance.to_csv(importance_path, index=False)
print(f"   Feature importance saved: {importance_path}")

# Summary
print("\n" + "="*80)
print("SHAP ANALYSIS COMPLETE!")
print("="*80)
print(f"\n   Top 10 Most Important Features (by SHAP):")
for idx, row in feature_importance.head(10).iterrows():
    print(f"      {row['feature'][:45]:45s}: {row['mean_abs_shap']:.4f}")

print(f"\n   Visualizations Created:")
print(f"      1. {bar_plot_path.name}")
print(f"      2. {beeswarm_path.name}")
print(f"      3. {waterfall_path.name}")
print(f"      4. {force_path.name}")

print(f"\n   Data Files:")
print(f"      1. {shap_csv_path.name}")
print(f"      2. {importance_path.name}")

print(f"\n   All files saved to: {output_dir}")
print("\n" + "="*80)
