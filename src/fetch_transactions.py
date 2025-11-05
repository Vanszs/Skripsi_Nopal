"""
Fetch transactions from blockchain networks (BSC, opBNB, Ethereum) using Web3.py and Block Explorer APIs.
Implements rate limiting, caching, and exponential backoff for robust data retrieval.
"""

import time
import requests
import pandas as pd
from web3 import Web3
from typing import List, Dict, Optional, Union
from tqdm import tqdm

from config import (
    MAINNET_RPC_URL,
    EXPLORER_API_URL,
    EXPLORER_API_KEY,
    RAW_DATA_DIR,
    ETHERSCAN_MAX_REQUESTS_PER_SEC,
    ETHERSCAN_BACKOFF_TIME,
    START_BLOCK,
    END_BLOCK,
    ACTIVE_NETWORK,
    CURRENT_NETWORK,
    CHAIN_ID,
)
from utils import setup_logger, save_pickle, load_pickle

logger = setup_logger(__name__)


class EthereumDataFetcher:
    """
    Fetcher for blockchain transaction data.
    Supports BSC, opBNB, and Ethereum Mainnet.
    Uses Block Explorer APIs for historical data and Web3.py for real-time data.
    """
    
    def __init__(self):
        """Initialize Web3 connection and API endpoints."""
        self.w3 = Web3(Web3.HTTPProvider(MAINNET_RPC_URL))
        self.explorer_api = EXPLORER_API_URL
        self.api_key = EXPLORER_API_KEY
        self.chain_id = CHAIN_ID
        self.network_name = CURRENT_NETWORK['name']
        self.native_token = CURRENT_NETWORK['native_token']
        self.cache_file = RAW_DATA_DIR / f"tx_cache_{ACTIVE_NETWORK.lower()}.pkl"
        
        # Validate connection
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to {self.network_name} RPC")
        
        logger.info(f"[OK] Connected to {self.network_name}")
        logger.info(f"   Chain ID: {self.chain_id}")
        logger.info(f"   Native Token: {self.native_token}")
        logger.info(f"   Current block: {self.w3.eth.block_number}")
    
    def fetch_tx_batch(
        self, 
        address: str, 
        start_block: int, 
        end_block: int,
        max_retries: int = 3
    ) -> List[Dict]:
        """
        Fetch transaction batch for an address with rate limit handling.
        
        Parameters
        ----------
        address : str
            Blockchain address (0x prefixed)
        start_block : int
            Starting block number
        end_block : int
            Ending block number
        max_retries : int
            Maximum number of retry attempts
        
        Returns
        -------
        list of dict
            Transaction records
        
        References
        ----------
        BscScan API: https://docs.bscscan.com/api-endpoints/accounts
        Etherscan API: https://docs.etherscan.io/api-endpoints/accounts
        """
        # Build API URL (BSC/opBNB uses different format than Ethereum)
        if ACTIVE_NETWORK in ["BSC", "OPBNB"]:
            url = (
                f"{self.explorer_api}?module=account&action=txlist"
                f"&address={address}&startblock={start_block}&endblock={end_block}"
                f"&sort=asc&apikey={self.api_key}"
            )
        else:  # Ethereum
            url = (
                f"{self.explorer_api}?chainid={self.chain_id}&module=account&action=txlist"
                f"&address={address}&startblock={start_block}&endblock={end_block}"
                f"&sort=asc&apikey={self.api_key}"
            )
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30)
                
                # Handle rate limiting (HTTP 429)
                if response.status_code == 429:
                    wait_time = ETHERSCAN_BACKOFF_TIME * (2 ** attempt)
                    logger.warning(f"Rate limit hit, waiting {wait_time:.2f}s...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                if data.get('status') == '1' and data.get('result'):
                    return data['result']
                elif data.get('message') == 'No transactions found':
                    return []
                else:
                    logger.error(f"API error: {data.get('message', 'Unknown error')}")
                    return []
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(ETHERSCAN_BACKOFF_TIME * (2 ** attempt))
                else:
                    return []
        
        return []
    
    def fetch_known_scam_addresses(self) -> List[str]:
        """
        Fetch known scam/phishing addresses from Etherscan.
        
        Returns
        -------
        list of str
            List of known scam addresses
        
        References
        ----------
        Etherscan labeled addresses can be found via:
        - Manual compilation from Etherscan's phishing database
        - XBlock dataset: https://xblock.pro
        - Chainabuse reports: https://chainabuse.com
        """
        # Hardcoded known scam addresses (example - expand from real sources)
        # In production, fetch from external API or database
        scam_addresses = [
            "0x1234567890123456789012345678901234567890",  # Example
            # Add more from real sources
        ]
        
        logger.info(f"Loaded {len(scam_addresses)} known scam addresses")
        return [addr.lower() for addr in scam_addresses]
    
    def fetch_transactions_for_addresses(
        self,
        addresses: List[str],
        start_block: Optional[int] = None,
        end_block: Optional[Union[str, int]] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch transactions for multiple addresses.
        
        Parameters
        ----------
        addresses : list of str
            Ethereum addresses to fetch
        start_block : int, optional
            Starting block number
        end_block : str or int, optional
            Ending block number or 'latest'
        use_cache : bool
            Whether to use cached data
        
        Returns
        -------
        pd.DataFrame
            Transaction dataframe with columns:
            hash, from, to, value, gasPrice, gasUsed, timestamp, blockNumber, etc.
        """
        if start_block is None:
            start_block = START_BLOCK
        
        end_block_int: int
        if end_block is None or end_block == 'latest':
            end_block_int = self.w3.eth.block_number
        else:
            end_block_int = int(end_block)
        
        # Check cache
        if use_cache and self.cache_file.exists():
            logger.info(f"Loading from cache: {self.cache_file}")
            return load_pickle(self.cache_file)
        
        all_transactions = []
        
        logger.info(f"Fetching transactions for {len(addresses)} addresses...")
        logger.info(f"Block range: {start_block} to {end_block_int}")
        
        for address in tqdm(addresses, desc="Fetching addresses"):
            transactions = self.fetch_tx_batch(address, start_block, end_block_int)
            all_transactions.extend(transactions)
            
            # Rate limiting: max 5 req/sec
            time.sleep(1 / ETHERSCAN_MAX_REQUESTS_PER_SEC)
        
        if not all_transactions:
            logger.warning("No transactions fetched!")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_transactions)
        
        # Convert data types
        numeric_columns = ['value', 'gas', 'gasPrice', 'gasUsed', 
                          'blockNumber', 'timeStamp', 'transactionIndex']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert Wei to ETH
        if 'value' in df.columns:
            df['value_eth'] = df['value'] / 1e18
        
        # Convert timestamp to datetime
        if 'timeStamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timeStamp'], unit='s')
        
        # Normalize addresses to lowercase
        df['from'] = df['from'].str.lower()
        df['to'] = df['to'].str.lower()
        
        # Save to cache
        save_pickle(df, self.cache_file)
        
        logger.info(f"[OK] Fetched {len(df)} transactions")
        logger.info(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def label_scam_transactions(
        self, 
        df: pd.DataFrame, 
        scam_addresses: List[str]
    ) -> pd.DataFrame:
        """
        Label transactions as fraud (1) or normal (0) based on scam addresses.
        
        Parameters
        ----------
        df : pd.DataFrame
            Transaction dataframe
        scam_addresses : list of str
            Known scam addresses
        
        Returns
        -------
        pd.DataFrame
            Dataframe with 'is_fraud' column added
        """
        scam_set = set(addr.lower() for addr in scam_addresses)
        
        # Mark as fraud if 'to' address is in scam list
        df['is_fraud'] = df['to'].apply(lambda x: 1 if x in scam_set else 0)
        
        fraud_count = df['is_fraud'].sum()
        normal_count = len(df) - fraud_count
        
        logger.info(f"Labeled transactions:")
        logger.info(f"   Normal: {normal_count} ({100*normal_count/len(df):.1f}%)")
        logger.info(f"   Fraud: {fraud_count} ({100*fraud_count/len(df):.1f}%)")
        
        return df


def main():
    """Main execution function for testing."""
    fetcher = EthereumDataFetcher()
    
    # Example: Fetch transactions for well-known addresses with activity
    # Using Vitalik's address and other known active addresses
    sample_addresses = [
        "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",  # Vitalik Buterin
        "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",  # Known exchange address
    ]
    
    # Fetch transactions
    df = fetcher.fetch_transactions_for_addresses(
        addresses=sample_addresses,
        use_cache=True
    )
    
    if df is None or len(df) == 0:
        logger.error("No transactions fetched! Check:")
        logger.error("  1. RPC URL is valid and complete")
        logger.error("  2. Etherscan API key is valid")
        logger.error("  3. Addresses have transaction history")
        return None
    
    # Get known scam addresses
    scam_addresses = fetcher.fetch_known_scam_addresses()
    
    # Label transactions
    df = fetcher.label_scam_transactions(df, scam_addresses)
    
    # Save raw data
    output_file = RAW_DATA_DIR / "transactions_raw.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"[OK] Saved to {output_file}")
    
    return df


if __name__ == "__main__":
    main()
