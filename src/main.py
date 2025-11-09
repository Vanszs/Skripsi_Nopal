"""
Main pipeline for Ethereum fraud detection system.
Orchestrates data fetching, feature engineering, training, and evaluation.
"""

import argparse
import sys
from pathlib import Path

# Fix Windows console encoding for emoji support
if sys.platform == 'win32':
    import codecs
    sys.stdout.reconfigure(encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR, 
    validate_config, TARGET_TX_COUNT, MAX_BLOCKS_TO_SCAN
)
from utils import setup_logger
from fetch_transactions import OpBNBDataFetcher
from feature_engineering import FeatureEngineer
from network_graph import TransactionGraphAnalyzer
from model_train import FraudDetectionTrainer
from model_explain import ModelExplainer
from evaluate import ModelEvaluator

logger = setup_logger(__name__)


from typing import List, Optional

def fetch_data_stage(
    mode: str = "snapshot",
    target_count: int = TARGET_TX_COUNT,
    sample_addresses: Optional[List[str]] = None
):
    """
    Stage 1: Fetch transaction data from opBNB mainnet.
    
    Parameters
    ----------
    mode : str
        Fetch mode: 'snapshot' (recent tx), 'blocks', or 'addresses'
    target_count : int
        Target number of transactions for snapshot mode
    sample_addresses : list, optional
        List of addresses to fetch (for addresses mode)
    """
    logger.info("\n" + "="*60)
    logger.info("STAGE 1: DATA FETCHING (opBNB Mainnet)")
    logger.info("="*60)
    
    fetcher = OpBNBDataFetcher()
    
    # Mode selection
    if mode == "snapshot":
        logger.info(f"Mode: Snapshot (fetching last {target_count:,} transactions)")
        df = fetcher.fetch_recent_transactions_snapshot(
            target_count=target_count,
            max_blocks=MAX_BLOCKS_TO_SCAN
        )
    
    elif mode == "addresses":
        # Use sample addresses if none provided
        if sample_addresses is None:
            sample_addresses = [
                "0x4200000000000000000000000000000000000006",  # WBNB on opBNB
                "0x4200000000000000000000000000000000000010",  # Gas Price Oracle
                "0x0000000000000000000000000000000000001000",  # System contract
            ]
            logger.info(f"Using {len(sample_addresses)} sample addresses")
        
        df = fetcher.fetch_transactions_for_addresses(
            addresses=sample_addresses,
            use_cache=True
        )
    
    else:
        logger.error(f"Unknown mode: {mode}")
        return None
    
    if df is None or df.empty:
        logger.error("No transactions fetched!")
        return None
    
    # Get known scam addresses and label
    scam_addresses = fetcher.fetch_known_scam_addresses()
    df = fetcher.label_scam_transactions(
        df, 
        scam_addresses,
        use_synthetic=True,  # Enable synthetic labels for research
        contamination=0.05   # 5% fraud rate
    )
    
    # Save raw data
    output_file = RAW_DATA_DIR / "transactions_raw.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"[OK] Stage 1 complete: {len(df):,} transactions saved to {output_file}")
    
    return df


def feature_engineering_stage(include_graph_features: bool = True):
    """
    Stage 2: Feature engineering and graph analysis.
    
    Parameters
    ----------
    include_graph_features : bool
        Whether to include network graph features
    """
    logger.info("\n" + "="*60)
    logger.info("STAGE 2: FEATURE ENGINEERING")
    logger.info("="*60)
    
    # Load raw data
    raw_file = RAW_DATA_DIR / "transactions_raw.csv"
    if not raw_file.exists():
        logger.error(f"Raw data not found: {raw_file}")
        logger.info("Run with --fetch-data first!")
        return None
    
    import pandas as pd
    df = pd.read_csv(raw_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    logger.info(f"Loaded {len(df)} transactions")
    
    # Basic feature engineering
    engineer = FeatureEngineer()
    df_features = engineer.create_feature_matrix(df)
    
    # Network graph features
    if include_graph_features:
        logger.info("\nAdding network graph features...")
        analyzer = TransactionGraphAnalyzer()
        scam_addresses = df[df['is_fraud'] == 1]['to'].unique().tolist()
        df_features = analyzer.extract_all_graph_features(df_features, scam_addresses)
    
    # Save processed data
    output_file = PROCESSED_DATA_DIR / "features.csv"
    df_features.to_csv(output_file, index=False)
    logger.info(f"[OK] Stage 2 complete: {df_features.shape[1]} features saved to {output_file}")
    
    return df_features


def training_stage(use_cv: bool = False):
    """
    Stage 3: Model training with time-series validation.
    
    Parameters
    ----------
    use_cv : bool
        Whether to use time-series cross-validation
    """
    logger.info("\n" + "="*60)
    logger.info("STAGE 3: MODEL TRAINING")
    logger.info("="*60)
    
    # Load processed data
    data_file = PROCESSED_DATA_DIR / "features.csv"
    if not data_file.exists():
        logger.error(f"Processed data not found: {data_file}")
        logger.info("Run with --feature-engineering first!")
        return None
    
    import pandas as pd
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    logger.info(f"Loaded {len(df)} samples with {df.shape[1]} features")
    
    # Initialize trainer
    trainer = FraudDetectionTrainer()
    
    # Prepare data
    X_train, X_test, y_train, y_test, ts_train, ts_test = trainer.prepare_data(df)
    
    # Time-series cross-validation
    if use_cv:
        logger.info("\nPerforming time-series cross-validation...")
        cv_scores = trainer.time_series_cv_evaluate(X_train, y_train, ts_train)
    
    # Train final model
    logger.info("\nTraining final model on full training set...")
    trainer.train_model(X_train, y_train, X_test, y_test)
    
    # Save model
    model_path = trainer.save_model()
    logger.info(f"✅ Stage 3 complete: Model saved to {model_path}")
    
    return trainer, X_test, y_test


def explanation_stage():
    """
    Stage 4: Generate SHAP explanations.
    """
    logger.info("\n" + "="*60)
    logger.info("STAGE 4: MODEL EXPLAINABILITY")
    logger.info("="*60)
    
    import glob
    from xgboost import XGBClassifier
    import pandas as pd
    
    # Load model
    from config import MODEL_DIR
    model_files = sorted(glob.glob(str(MODEL_DIR / "xgb_fraud_*.json")))
    if not model_files:
        logger.error("No trained model found!")
        return None
    
    latest_model = model_files[-1]
    logger.info(f"Loading model: {latest_model}")
    
    model = XGBClassifier()
    model.load_model(latest_model)
    
    # Load test data
    data_file = PROCESSED_DATA_DIR / "features.csv"
    df = pd.read_csv(data_file)
    
    feature_cols = [col for col in df.columns if col not in ['is_fraud', 'timestamp', 'hash', 'from', 'to', 'address']]
    X = df[feature_cols]
    y = df['is_fraud']
    
    # Test set (last 20%)
    split_idx = int(len(df) * 0.8)
    X_test = X.iloc[split_idx:split_idx + 100]  # Use 100 samples for speed
    y_test = y.iloc[split_idx:split_idx + 100]
    
    # Create explainer
    explainer = ModelExplainer(model)
    explainer.compute_shap_values(X_test)
    
    # Generate plots
    explainer.plot_global_importance(X_test)
    explainer.plot_local_explanation(X_test, sample_idx=0)
    
    # Text explanations
    explainer.explain_predictions(X_test, y_test, n_samples=3)
    
    logger.info("✅ Stage 4 complete: SHAP explanations generated")
    
    return explainer


def evaluation_stage():
    """
    Stage 5: Comprehensive model evaluation.
    """
    logger.info("\n" + "="*60)
    logger.info("STAGE 5: MODEL EVALUATION")
    logger.info("="*60)
    
    import glob
    from xgboost import XGBClassifier
    import pandas as pd
    
    # Load model
    from config import MODEL_DIR
    model_files = sorted(glob.glob(str(MODEL_DIR / "xgb_fraud_*.json")))
    if not model_files:
        logger.error("No trained model found!")
        return None
    
    latest_model = model_files[-1]
    logger.info(f"Loading model: {latest_model}")
    
    model = XGBClassifier()
    model.load_model(latest_model)
    
    # Load test data
    data_file = PROCESSED_DATA_DIR / "features.csv"
    df = pd.read_csv(data_file)
    
    feature_cols = [col for col in df.columns if col not in ['is_fraud', 'timestamp', 'hash', 'from', 'to', 'address']]
    X = df[feature_cols]
    y = df['is_fraud']
    
    # Test set
    split_idx = int(len(df) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    # Evaluate
    evaluator = ModelEvaluator(model)
    evaluator.evaluate(X_test, y_test)
    
    # Generate plots
    evaluator.plot_confusion_matrix()
    evaluator.plot_pr_curve(X_test, y_test)
    evaluator.plot_roc_curve(X_test, y_test)
    evaluator.plot_feature_importance(feature_cols)
    
    # Save results
    evaluator.save_results()
    
    logger.info("✅ Stage 5 complete: Evaluation results saved")
    
    return evaluator


def main():
    """
    Main pipeline orchestration.
    """
    parser = argparse.ArgumentParser(
        description='Ethereum Fraud Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (recommended for first run)
  python src/main.py --full-pipeline
  
  # Run with custom snapshot size
  python src/main.py --fetch-data --fetch-mode snapshot --target-count 10000
  
  # Run specific stages
  python src/main.py --fetch-data
  python src/main.py --feature-engineering
  python src/main.py --train --time-series-cv
  python src/main.py --explain --evaluate
        """
    )
    
    parser.add_argument('--fetch-data', action='store_true',
                       help='Fetch transaction data from opBNB mainnet')
    parser.add_argument('--fetch-mode', type=str, default='snapshot',
                       choices=['snapshot', 'blocks', 'addresses'],
                       help='Data fetching mode (default: snapshot)')
    parser.add_argument('--target-count', type=int, default=TARGET_TX_COUNT,
                       help=f'Target tx count for snapshot mode (default: {TARGET_TX_COUNT})')
    parser.add_argument('--feature-engineering', action='store_true',
                       help='Extract features from raw transactions')
    parser.add_argument('--no-graph-features', action='store_true',
                       help='Skip network graph features (faster)')
    parser.add_argument('--train', action='store_true',
                       help='Train XGBoost model')
    parser.add_argument('--time-series-cv', action='store_true',
                       help='Use time-series cross-validation')
    parser.add_argument('--explain', action='store_true',
                       help='Generate SHAP explanations')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model performance')
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Run complete pipeline (all stages)')
    
    args = parser.parse_args()
    
    # Validate configuration
    try:
        validate_config()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.info("Please set up your .env file with API credentials")
        return 1
    
    # Banner
    print("\n" + "="*60)
    print("  opBNB FRAUD DETECTION SYSTEM")
    print("  XGBoost + SHAP + Network Graph Analysis")
    print("  Network: opBNB Mainnet (Chain ID: 204)")
    print("="*60 + "\n")
    
    # Run stages based on arguments
    if args.full_pipeline or not any([args.fetch_data, args.feature_engineering, 
                                      args.train, args.explain, args.evaluate]):
        args.fetch_data = True
        args.feature_engineering = True
        args.train = True
        args.explain = True
        args.evaluate = True
    
    try:
        if args.fetch_data:
            fetch_data_stage(
                mode=args.fetch_mode,
                target_count=args.target_count
            )
        
        # Check if feature engineering is needed before training
        features_file = PROCESSED_DATA_DIR / "features.csv"
        if args.train and not features_file.exists():
            logger.info("Features not found, running feature engineering first...")
            args.feature_engineering = True
        
        if args.feature_engineering:
            feature_engineering_stage(include_graph_features=not args.no_graph_features)
        
        if args.train:
            training_stage(use_cv=args.time_series_cv)
        
        if args.explain:
            explanation_stage()
        
        if args.evaluate:
            evaluation_stage()
        
        print("\n" + "="*60)
        print("  ✅ PIPELINE COMPLETE!")
        print("="*60 + "\n")
        
        logger.info("Check the following directories for outputs:")
        logger.info(f"  - Models: {MODEL_DIR}")
        logger.info(f"  - Data: {PROCESSED_DATA_DIR}")
        logger.info(f"  - Logs: {Path(__file__).parent.parent / 'logs'}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
