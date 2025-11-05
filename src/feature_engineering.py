"""
Feature engineering for Ethereum transactions.
Extracts temporal, value, gas, and account behavior features from raw transaction data.
"""

import pandas as pd
import numpy as np
from typing import List
from datetime import timedelta

from config import PROCESSED_DATA_DIR, LOG_TRANSFORM_FEATURES
from utils import setup_logger, reduce_mem_usage

logger = setup_logger(__name__)


class FeatureEngineer:
    """
    Feature extraction from Ethereum transactions.
    Generates numeric features for ML model training.
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_names = []
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time-based features from transactions.
        
        Parameters
        ----------
        df : pd.DataFrame
            Transaction dataframe with 'timestamp' column
        
        Returns
        -------
        pd.DataFrame
            DataFrame with added temporal features
        
        Features Generated
        ------------------
        - hour_of_day: Hour when transaction occurred (0-23)
        - day_of_week: Day of week (0=Monday, 6=Sunday)
        - is_weekend: Binary flag for weekend transactions
        - tx_interval_mean: Mean time between transactions for address
        - time_since_last_tx: Time since previous transaction (seconds)
        """
        logger.info("Extracting temporal features...")
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Basic temporal features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Sort by address and timestamp for sequential features
        df = df.sort_values(['from', 'timestamp']).reset_index(drop=True)
        
        # Time since last transaction per address
        df['time_since_last_tx'] = df.groupby('from')['timestamp'].diff().dt.total_seconds()
        df['time_since_last_tx'] = df['time_since_last_tx'].fillna(0)
        
        # Mean transaction interval per address
        df['tx_interval_mean'] = df.groupby('from')['time_since_last_tx'].transform('mean')
        
        self.feature_names.extend([
            'hour_of_day', 'day_of_week', 'is_weekend',
            'tx_interval_mean', 'time_since_last_tx'
        ])
        
        logger.info(f"   Added {5} temporal features")
        return df
    
    def extract_value_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract value-related features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Transaction dataframe with 'value_eth' column
        
        Returns
        -------
        pd.DataFrame
            DataFrame with added value features
        
        Features Generated
        ------------------
        - log_value: Log-transformed value (handles zeros with log1p)
        - value_to_gas_ratio: Ratio of transaction value to gas cost
        - is_zero_value: Binary flag for zero-value transactions
        """
        logger.info("Extracting value features...")
        
        # Log transform (handles zeros)
        df['log_value'] = np.log1p(df['value_eth'])
        
        # Value to gas ratio
        df['gas_cost_eth'] = df['gasUsed'] * df['gasPrice'] / 1e18
        df['value_to_gas_ratio'] = df['value_eth'] / (df['gas_cost_eth'] + 1e-10)
        
        # Zero value flag
        df['is_zero_value'] = (df['value_eth'] == 0).astype(int)
        
        self.feature_names.extend([
            'value_eth', 'log_value', 'value_to_gas_ratio', 'is_zero_value'
        ])
        
        logger.info(f"   Added {4} value features")
        return df
    
    def extract_gas_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract gas-related features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Transaction dataframe with gas columns
        
        Returns
        -------
        pd.DataFrame
            DataFrame with added gas features
        
        Features Generated
        ------------------
        - gas_price: Gas price in Gwei
        - gas_used: Gas units used
        - gas_price_ratio: Gas price relative to block median (normalized)
        - gas_efficiency: Ratio of gas used to gas limit
        
        References
        ----------
        Gas normalization is critical due to Ethereum's volatile gas market.
        See: https://etherscan.io/gastracker
        """
        logger.info("Extracting gas features...")
        
        # Convert gas price to Gwei for better scale
        df['gas_price_gwei'] = df['gasPrice'] / 1e9
        
        # Normalize gas price by block median
        df['block_median_gas'] = df.groupby('blockNumber')['gas_price_gwei'].transform('median')
        df['gas_price_ratio'] = df['gas_price_gwei'] / (df['block_median_gas'] + 1e-10)
        
        # Gas efficiency (used vs limit)
        df['gas_efficiency'] = df['gasUsed'] / (df['gas'] + 1)
        
        self.feature_names.extend([
            'gas_price_gwei', 'gasUsed', 'gas_price_ratio', 'gas_efficiency'
        ])
        
        logger.info(f"   Added {4} gas features")
        return df
    
    def extract_account_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract account behavior features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Transaction dataframe
        
        Returns
        -------
        pd.DataFrame
            DataFrame with added account behavior features
        
        Features Generated
        ------------------
        - unique_receivers: Number of unique addresses sent to
        - total_tx_24h: Transaction count in last 24 hours
        - tx_success_rate: Ratio of successful transactions
        - burst_activity_flag: Flag for unusually high activity
        """
        logger.info("Extracting account behavior features...")
        
        # Unique receivers per sender
        df['unique_receivers'] = df.groupby('from')['to'].transform('nunique')
        
        # Rolling 24h transaction count
        df = df.sort_values(['from', 'timestamp']).reset_index(drop=True)
        
        # Use count instead of rolling window (simpler approach)
        # For each transaction, count how many tx from same address in last 24h
        def count_tx_24h(group):
            counts = []
            for idx, row in group.iterrows():
                mask = (group['timestamp'] >= row['timestamp'] - pd.Timedelta(hours=24)) & \
                       (group['timestamp'] <= row['timestamp'])
                counts.append(mask.sum())
            return pd.Series(counts, index=group.index)
        
        tx_24h_results = df.groupby('from', group_keys=False).apply(count_tx_24h)
        df['tx_24h_window'] = tx_24h_results.values
        
        # Success rate (assuming txreceipt_status or isError column)
        if 'isError' in df.columns:
            df['tx_success_rate'] = df.groupby('from')['isError'].transform(
                lambda x: 1 - x.mean()
            )
        else:
            df['tx_success_rate'] = 1.0  # Assume all successful if no error data
        
        # Burst activity detection (>10 tx in 1 hour)
        def count_tx_1h(group):
            counts = []
            for idx, row in group.iterrows():
                mask = (group['timestamp'] >= row['timestamp'] - pd.Timedelta(hours=1)) & \
                       (group['timestamp'] <= row['timestamp'])
                counts.append(mask.sum())
            return pd.Series(counts, index=group.index)
        
        tx_1h_results = df.groupby('from', group_keys=False).apply(count_tx_1h)
        df['tx_1h_window'] = tx_1h_results.values
        df['burst_activity_flag'] = (df['tx_1h_window'] > 10).astype(int)
        
        self.feature_names.extend([
            'unique_receivers', 'tx_24h_window', 'tx_success_rate', 'burst_activity_flag'
        ])
        
        logger.info(f"   Added {4} account behavior features")
        return df
    
    def create_feature_matrix(
        self, 
        df: pd.DataFrame,
        include_graph_features: bool = False
    ) -> pd.DataFrame:
        """
        Create full feature matrix from raw transactions.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw transaction dataframe
        include_graph_features : bool
            Whether to wait for graph features (added by network_graph.py)
        
        Returns
        -------
        pd.DataFrame
            Complete feature matrix ready for modeling
        """
        logger.info("Creating feature matrix...")
        
        # Extract all feature types
        df = self.extract_temporal_features(df)
        df = self.extract_value_features(df)
        df = self.extract_gas_features(df)
        df = self.extract_account_behavior_features(df)
        
        # Handle missing values
        df = df.fillna(0)
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], 0)
        
        # Drop non-numeric columns (except labels and metadata)
        # Keep only: numeric features + is_fraud + timestamp + hash (for tracking)
        non_feature_cols = ['hash', 'from', 'to', 'timestamp', 'is_fraud']
        
        # Identify object/string columns to drop
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        cols_to_drop = [col for col in object_cols if col not in non_feature_cols]
        
        if cols_to_drop:
            logger.info(f"   Dropping {len(cols_to_drop)} non-numeric columns: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        
        # Memory optimization
        df = reduce_mem_usage(df)
        
        logger.info(f"✅ Feature matrix created: {df.shape}")
        logger.info(f"   Total features: {len(self.feature_names)}")
        
        return df


def main():
    """Main execution function for testing."""
    from config import RAW_DATA_DIR
    
    # Load raw data
    raw_file = RAW_DATA_DIR / "transactions_raw.csv"
    if not raw_file.exists():
        logger.error(f"Raw data not found: {raw_file}")
        logger.info("Run fetch_transactions.py first!")
        return
    
    df = pd.read_csv(raw_file)
    logger.info(f"Loaded {len(df)} transactions")
    
    # Create features
    engineer = FeatureEngineer()
    df_features = engineer.create_feature_matrix(df)
    
    # Save processed data
    output_file = PROCESSED_DATA_DIR / "features.csv"
    df_features.to_csv(output_file, index=False)
    logger.info(f"✅ Saved features to {output_file}")
    
    # Print feature summary
    print("\n=== Feature Summary ===")
    print(f"Total features: {len(engineer.feature_names)}")
    print(f"Feature names: {engineer.feature_names}")
    
    return df_features


if __name__ == "__main__":
    main()
