# -*- coding: utf-8 -*-
"""Test connection to opBNB/BSC/ETH networks."""
from web3 import Web3
import requests
import os
from dotenv import load_dotenv

load_dotenv()
ACTIVE = os.getenv('ACTIVE_NETWORK', 'OPBNB').upper()

CFG = {
    'BSC': {
        'rpc': os.getenv('BSC_RPC_URL'),
        'api': 'https://api.etherscan.io/v2/api',  # V2 unified
        'key': os.getenv('BSCSCAN_API_KEY'),
        'chain': 56,
        'addr': '0x8894e0a0c962cb723c1976a4421c95949be2d4e3'
    },
    'OPBNB': {
        'rpc': os.getenv('OPBNB_RPC_URL'),
        'api': 'https://api.etherscan.io/v2/api',  # V2 unified
        'key': os.getenv('BSCSCAN_API_KEY'),
        'chain': 204,
        'addr': '0x4200000000000000000000000000000000000006'
    },
    'ETH': {
        'rpc': os.getenv('ETH_MAINNET_RPC_URL'),
        'api': 'https://api.etherscan.io/v2/api',  # V2 unified
        'key': os.getenv('ETHERSCAN_API_KEY'),
        'chain': 1,
        'addr': '0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045'
    }
}

c = CFG.get(ACTIVE, CFG['OPBNB'])

print(f"\n{'='*60}")
print(f"Testing {ACTIVE} Network (Chain ID: {c['chain']})")
print(f"{'='*60}")

# Test RPC
print("\n[1] RPC Connection:")
try:
    w3 = Web3(Web3.HTTPProvider(c['rpc']))
    if w3.is_connected():
        block = w3.eth.block_number
        print(f"    OK - Current block: {block:,}")
    else:
        print("    FAILED - Could not connect")
except Exception as e:
    print(f"    ERROR: {e}")

# Test API
print("\n[2] Explorer API:")
try:
    test_address = c['addr']
    api_key = c['key']
    
    # API V2 format - unified endpoint with chainid parameter for all chains
    url = f"{c['api']}?chainid={c['chain']}&module=account&action=balance&address={test_address}&tag=latest&apikey={api_key}"
    
    r = requests.get(url, timeout=10)
    d = r.json()
    
    if d.get('status') == '1':
        balance = int(d['result']) / 1e18
        print(f"    OK - Balance: {balance:.4f}")
    else:
        print(f"    ERROR: {d.get('message', 'Unknown')}")
except Exception as e:
    print(f"    ERROR: {e}")

print(f"\n{'='*60}")
print("Test Complete!")
print(f"{'='*60}\n")
