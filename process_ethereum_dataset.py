"""
Process Ethereum Fraud Detection Dataset
Dataset: 9,841 transactions with 51 features
Labels: FLAG (1=fraud, 0=normal)

This script:
1. Loads Ethereum fraud dataset
2. Analyzes features
3. Prepares data for XGBoost training
4. Exports cleaned dataset for model training
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_ethereum_dataset():
    """Load Ethereum fraud dataset from extracted archive"""
    dataset_path = Path('data/external/ethereum_fraud/transaction_dataset.csv')
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        logger.info("Please extract archive.zip to data/external/ethereum_fraud/")
        return None
    
    logger.info(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    logger.info(f"Loaded {len(df):,} transactions with {len(df.columns)} columns")
    
    return df


def analyze_dataset(df):
    """Analyze dataset structure and quality"""
    logger.info("\n" + "="*60)
    logger.info("DATASET ANALYSIS")
    logger.info("="*60)
    
    # Basic info
    logger.info(f"\nShape: {df.shape}")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Label distribution
    if 'FLAG' in df.columns:
        fraud_count = df['FLAG'].sum()
        normal_count = len(df) - fraud_count
        
        logger.info(f"\nLabel Distribution:")
        logger.info(f"  Fraud: {fraud_count:,} ({fraud_count/len(df)*100:.2f}%)")
        logger.info(f"  Normal: {normal_count:,} ({normal_count/len(df)*100:.2f}%)")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.info(f"\nMissing Values:")
        for col in missing[missing > 0].index:
            pct = missing[col] / len(df) * 100
            logger.info(f"  {col}: {missing[col]:,} ({pct:.1f}%)")
    else:
        logger.info(f"\nNo missing values found")
    
    # Data types
    logger.info(f"\nData Types:")
    logger.info(f"  Numeric: {df.select_dtypes(include=[np.number]).shape[1]}")
    logger.info(f"  Object: {df.select_dtypes(include=['object']).shape[1]}")
    
    # Feature categories
    logger.info(f"\nFeature Categories:")
    logger.info(f"  Time-based: {len([c for c in df.columns if 'time' in c.lower() or 'min' in c.lower()])}")
    logger.info(f"  Value-based: {len([c for c in df.columns if 'val' in c.lower() or 'ether' in c.lower()])}")
    logger.info(f"  ERC20-based: {len([c for c in df.columns if 'erc20' in c.lower()])}")
    logger.info(f"  Count-based: {len([c for c in df.columns if 'number' in c.lower() or 'total' in c.lower()])}")


def clean_dataset(df):
    """Clean and prepare dataset for training"""
    logger.info("\n" + "="*60)
    logger.info("CLEANING DATASET")
    logger.info("="*60)
    
    df_clean = df.copy()
    
    # Drop unnecessary columns
    drop_cols = ['Unnamed: 0', 'Index', 'Address']
    existing_drop_cols = [col for col in drop_cols if col in df_clean.columns]
    
    if existing_drop_cols:
        df_clean = df_clean.drop(columns=existing_drop_cols)
        logger.info(f"\nDropped columns: {existing_drop_cols}")
    
    # Handle missing values
    # For numeric columns, fill with 0 (represents no activity)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(0)
    
    # For object columns, fill with 'Unknown'
    object_cols = df_clean.select_dtypes(include=['object']).columns
    object_cols = [col for col in object_cols if col != 'FLAG']  # Don't touch label column
    
    for col in object_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna('Unknown')
    
    logger.info(f"\nFilled missing values")
    
    # Handle infinite values
    df_clean = df_clean.replace([np.inf, -np.inf], 0)
    logger.info(f"Replaced infinite values with 0")
    
    # Ensure FLAG is integer
    if 'FLAG' in df_clean.columns:
        df_clean['FLAG'] = df_clean['FLAG'].astype(int)
    
    logger.info(f"\nCleaned dataset shape: {df_clean.shape}")
    
    return df_clean


def prepare_training_data(df_clean):
    """Prepare features and labels for model training"""
    logger.info("\n" + "="*60)
    logger.info("PREPARING TRAINING DATA")
    logger.info("="*60)
    
    # Separate features and labels
    if 'FLAG' not in df_clean.columns:
        logger.error("FLAG column not found!")
        return None, None
    
    y = df_clean['FLAG']
    X = df_clean.drop(columns=['FLAG'])
    
    # Convert object columns to numeric (one-hot encoding or label encoding)
    object_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    if object_cols:
        logger.info(f"\nConverting {len(object_cols)} object columns to numeric:")
        for col in object_cols:
            logger.info(f"  - {col}")
            # Simple label encoding (or use pd.get_dummies for one-hot)
            X[col] = pd.Categorical(X[col]).codes
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    logger.info(f"\nFinal feature matrix:")
    logger.info(f"  Shape: {X.shape}")
    logger.info(f"  Features: {len(feature_names)}")
    logger.info(f"  All numeric: {X.select_dtypes(include=[np.number]).shape[1] == len(feature_names)}")
    
    logger.info(f"\nLabel distribution:")
    logger.info(f"  Fraud: {y.sum():,} ({y.mean()*100:.2f}%)")
    logger.info(f"  Normal: {(y==0).sum():,} ({(y==0).mean()*100:.2f}%)")
    
    return X, y


def export_dataset(X, y):
    """Export cleaned dataset for training"""
    logger.info("\n" + "="*60)
    logger.info("EXPORTING DATASET")
    logger.info("="*60)
    
    # Create processed directory
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine X and y
    df_export = X.copy()
    df_export['label'] = y
    
    # Export full dataset
    full_path = output_dir / 'ethereum_fraud_full.csv'
    df_export.to_csv(full_path, index=False)
    logger.info(f"\nExported full dataset: {full_path}")
    logger.info(f"  Rows: {len(df_export):,}")
    logger.info(f"  Columns: {len(df_export.columns)}")
    logger.info(f"  Size: {full_path.stat().st_size / 1024**2:.2f} MB")
    
    # Export balanced sample for quick training
    fraud_samples = df_export[df_export['label'] == 1]
    normal_samples = df_export[df_export['label'] == 0].sample(len(fraud_samples) * 2, random_state=42)
    
    df_balanced = pd.concat([fraud_samples, normal_samples], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    balanced_path = output_dir / 'ethereum_fraud_balanced.csv'
    df_balanced.to_csv(balanced_path, index=False)
    logger.info(f"\nExported balanced sample: {balanced_path}")
    logger.info(f"  Rows: {len(df_balanced):,}")
    logger.info(f"  Fraud: {df_balanced['label'].sum():,}")
    logger.info(f"  Normal: {(df_balanced['label']==0).sum():,}")
    logger.info(f"  Size: {balanced_path.stat().st_size / 1024**2:.2f} MB")
    
    return full_path, balanced_path


def main():
    """Main processing pipeline"""
    logger.info("="*60)
    logger.info("ETHEREUM FRAUD DATASET PROCESSOR")
    logger.info("="*60)
    
    # Step 1: Load dataset
    df = load_ethereum_dataset()
    if df is None:
        return
    
    # Step 2: Analyze
    analyze_dataset(df)
    
    # Step 3: Clean
    df_clean = clean_dataset(df)
    
    # Step 4: Prepare training data
    X, y = prepare_training_data(df_clean)
    if X is None:
        return
    
    # Step 5: Export
    full_path, balanced_path = export_dataset(X, y)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("PROCESSING COMPLETE!")
    logger.info("="*60)
    logger.info("\nDatasets created:")
    logger.info(f"  1. {full_path.name} - Full dataset (9,841 samples)")
    logger.info(f"  2. {balanced_path.name} - Balanced sample (~6,500 samples)")
    
    logger.info("\nNext steps:")
    logger.info("  1. Train XGBoost model: python src/model_train.py")
    logger.info("  2. Evaluate performance: python src/evaluate.py")
    logger.info("  3. Generate SHAP explanations: python src/model_explain.py")
    logger.info("  4. Update notebook: notebooks/03_OpBNB_Fraud_End_to_End.ipynb")


if __name__ == "__main__":
    main()
