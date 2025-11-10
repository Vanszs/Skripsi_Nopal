"""
Quick script to display model comparison results in terminal
Run this to see results without opening Jupyter notebook
"""

import pandas as pd
from pathlib import Path

print("="*100)
print(" "*35 + "🔬 MODEL COMPARISON RESULTS")
print("="*100)

# Load comparison data
csv_path = Path('models/model_comparison.csv')

if csv_path.exists():
    df = pd.read_csv(csv_path)
    
    print("\n📊 PERFORMANCE METRICS COMPARISON")
    print("-"*100)
    print(df[['Model', 'PR-AUC', 'ROC-AUC', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))
    
    print("\n\n⏱️  SPEED COMPARISON")
    print("-"*100)
    print(df[['Model', 'Train Time (s)', 'Inference Time (s)', 'Samples/sec']].to_string(index=False))
    
    print("\n\n🏆 RANKINGS")
    print("-"*100)
    
    print("\n1. Best PR-AUC (Primary Metric):")
    top_3_pr = df.nlargest(3, 'PR-AUC')
    for idx, (_, row) in enumerate(top_3_pr.iterrows(), 1):
        medal = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉"
        print(f"   {medal} {idx}. {row['Model']:20s} - PR-AUC: {row['PR-AUC']:.4f}, F1: {row['F1-Score']:.4f}")
    
    print("\n2. Fastest Training:")
    top_3_train = df.nsmallest(3, 'Train Time (s)')
    for idx, (_, row) in enumerate(top_3_train.iterrows(), 1):
        medal = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉"
        print(f"   {medal} {idx}. {row['Model']:20s} - {row['Train Time (s)']:.2f}s (PR-AUC: {row['PR-AUC']:.4f})")
    
    print("\n3. Fastest Inference:")
    top_3_inf = df.nlargest(3, 'Samples/sec')
    for idx, (_, row) in enumerate(top_3_inf.iterrows(), 1):
        medal = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉"
        print(f"   {medal} {idx}. {row['Model']:20s} - {row['Samples/sec']:.0f} samples/s")
    
    print("\n\n💡 RECOMMENDATIONS")
    print("-"*100)
    
    best_accuracy = df.loc[df['PR-AUC'].idxmax()]
    print(f"✅ Best Accuracy: {best_accuracy['Model']} (PR-AUC: {best_accuracy['PR-AUC']:.4f})")
    
    # Calculate efficiency score (PR-AUC / Training Time)
    df['Efficiency'] = df['PR-AUC'] / df['Train Time (s)']
    best_efficiency = df.loc[df['Efficiency'].idxmax()]
    print(f"⚡ Best Efficiency: {best_efficiency['Model']} (Score: {best_efficiency['Efficiency']:.4f})")
    
    best_speed = df.loc[df['Samples/sec'].idxmax()]
    print(f"🚀 Fastest Inference: {best_speed['Model']} ({best_speed['Samples/sec']:.0f} samples/s)")
    
    print("\n" + "="*100)
    print("📊 For interactive visualizations, open:")
    print("   notebooks/04_Ethereum_Fraud_Results_Visualization.ipynb")
    print("   and run Part 10: Model Comparison Analysis")
    print("="*100)

else:
    print("\n⚠️  Comparison data not found!")
    print("Please run: python compare_models_simple.py")
