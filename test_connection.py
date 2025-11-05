"""
Quick test to verify RPC and Etherscan API connectivity.
"""

import os
from dotenv import load_dotenv
from web3 import Web3
import requests

load_dotenv()

print("=" * 60)
print("TESTING ETHEREUM CONNECTION")
print("=" * 60)

# Test 1: RPC Connection
print("\n1️⃣ Testing RPC Connection...")
rpc_url = os.getenv("MAINNET_RPC_URL")
print(f"   RPC URL: {rpc_url[:50]}...")

try:
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if w3.is_connected():
        print("   ✅ RPC Connected!")
        print(f"   Current block: {w3.eth.block_number}")
        print(f"   Chain ID: {w3.eth.chain_id}")
    else:
        print("   ❌ RPC Connection FAILED!")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Etherscan API
print("\n2️⃣ Testing Etherscan API...")
api_key = os.getenv("ETHERSCAN_API_KEY")
print(f"   API Key: {api_key[:10]}...")

# Test with Vitalik's address (known to have transactions)
test_address = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
url = (
    f"https://api.etherscan.io/v2/api?chainid=1&module=account&action=txlist"
    f"&address={test_address}&startblock=18000000&endblock=18001000"
    f"&page=1&offset=10&sort=asc&apikey={api_key}"
)

try:
    response = requests.get(url, timeout=30)
    data = response.json()
    
    print(f"   Status Code: {response.status_code}")
    print(f"   API Status: {data.get('status')}")
    print(f"   API Message: {data.get('message')}")
    
    if data.get('status') == '1':
        result = data.get('result', [])
        print(f"   ✅ Etherscan API Working!")
        print(f"   Transactions found: {len(result)}")
        if result:
            print(f"   First TX Hash: {result[0].get('hash', 'N/A')[:20]}...")
    else:
        print(f"   ❌ API Error: {data.get('message')}")
        print(f"   Result: {data.get('result')}")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
