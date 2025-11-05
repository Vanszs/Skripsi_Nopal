# 🌐 Multi-Network Support Guide

## Supported Networks

The system now supports multiple blockchain networks:

### 1. **BNB Smart Chain (BSC)** - Default
- **Chain ID**: 56
- **Native Token**: BNB
- **RPC URL**: `https://bnb-mainnet.g.alchemy.com/v2/{API_KEY}`
- **Explorer**: BscScan (https://bscscan.com)
- **API**: https://api.bscscan.com/api

### 2. **opBNB (Layer 2)**
- **Chain ID**: 204
- **Native Token**: BNB
- **RPC URL**: `https://opbnb-mainnet.g.alchemy.com/v2/{API_KEY}`
- **Explorer**: opBNB BscScan (https://opbnb.bscscan.com)
- **API**: https://api-opbnb.bscscan.com/api

### 3. **Ethereum Mainnet**
- **Chain ID**: 1
- **Native Token**: ETH
- **RPC URL**: `https://eth-mainnet.g.alchemy.com/v2/{API_KEY}`
- **Explorer**: Etherscan (https://etherscan.io)
- **API**: https://api.etherscan.io/v2/api

---

## Configuration

### 📝 `.env` File Setup

```bash
# ============================================================
# BLOCKCHAIN NETWORK CONFIGURATION
# ============================================================

# Primary Network: BNB Smart Chain (BSC)
BSC_RPC_URL=https://bnb-mainnet.g.alchemy.com/v2/yyhCbLJY_uIHUhD7HQ4JG

# Secondary Network: opBNB (Layer 2)
OPBNB_RPC_URL=https://opbnb-mainnet.g.alchemy.com/v2/yyhCbLJY_uIHUhD7HQ4JG

# Legacy Ethereum Mainnet (Backup)
ETH_MAINNET_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/yyhCbLJY_uIHUhD7HQ4JG

# Active Network (switch between: BSC, OPBNB, ETH)
ACTIVE_NETWORK=BSC

# API Keys
BSCSCAN_API_KEY=your_bscscan_api_key
ETHERSCAN_API_KEY=your_etherscan_api_key
```

---

## Switching Networks

### Method 1: Update `.env` File

Edit the `ACTIVE_NETWORK` variable in `.env`:

```bash
# Use BNB Smart Chain
ACTIVE_NETWORK=BSC

# Use opBNB
ACTIVE_NETWORK=OPBNB

# Use Ethereum Mainnet
ACTIVE_NETWORK=ETH
```

### Method 2: Environment Variable (Temporary)

```bash
# Linux/macOS/Fish Shell
export ACTIVE_NETWORK=BSC
python src/main.py --full-pipeline

# Or inline
ACTIVE_NETWORK=OPBNB python src/main.py --fetch-data
```

---

## Network-Specific Features

### 🔗 BSC (Binance Smart Chain)
- **Lower gas fees** compared to Ethereum
- **Faster block times** (~3 seconds)
- **High transaction volume** (DeFi, meme tokens)
- **Block range**: Start from ~35,000,000 (recent blocks)

### ⚡ opBNB (Layer 2)
- **Ultra-low gas fees** (~$0.001 per tx)
- **Very fast** (<1 second finality)
- **High throughput** (optimized for gaming, NFTs)
- **Block range**: Start from recent blocks (launched Aug 2023)

### 🏦 Ethereum Mainnet
- **Most established** smart contract platform
- **Higher gas fees**
- **Slowest block times** (~12 seconds)
- **Most mature fraud detection datasets**
- **Block range**: Start from ~18,000,000+

---

## Data Fetching Examples

### Fetch from BSC

```bash
# Set active network
export ACTIVE_NETWORK=BSC

# Update start block for BSC
export START_BLOCK=35000000

# Run pipeline
python src/main.py --fetch-data
```

### Fetch from opBNB

```bash
export ACTIVE_NETWORK=OPBNB
export START_BLOCK=1000000  # Recent blocks
python src/main.py --fetch-data
```

### Fetch from Ethereum

```bash
export ACTIVE_NETWORK=ETH
export START_BLOCK=18000000  # Recent Ethereum blocks
python src/main.py --fetch-data
```

---

## Cache Management

Each network has its own cache file:

```
data/raw/
├── tx_cache_bsc.pkl      # BSC transactions
├── tx_cache_opbnb.pkl    # opBNB transactions
└── tx_cache_eth.pkl      # Ethereum transactions
```

This allows you to maintain separate datasets for each network.

---

## API Rate Limits

### BscScan API
- **Free Tier**: 5 requests/second
- **Pro Tier**: Higher limits available

### Etherscan API
- **Free Tier**: 5 requests/second
- **Pro Tier**: Higher limits available

### Alchemy RPC
- **Free Tier**: 25 requests/second
- **Growth Tier**: 300 requests/second

---

## Block Number Recommendations

### For Quick Testing (Recent Data)

```bash
# BSC - Last 1M blocks (~3.5 days)
START_BLOCK=40000000
END_BLOCK=latest

# opBNB - Last 1M blocks (~11 days, faster blocks)
START_BLOCK=20000000
END_BLOCK=latest

# Ethereum - Last 100k blocks (~14 days)
START_BLOCK=21000000
END_BLOCK=latest
```

### For Research (More Data)

```bash
# BSC - Last 10M blocks (~35 days)
START_BLOCK=35000000
END_BLOCK=latest

# opBNB - Last 10M blocks (~110 days)
START_BLOCK=10000000
END_BLOCK=latest

# Ethereum - Last 1M blocks (~140 days)
START_BLOCK=18000000
END_BLOCK=latest
```

---

## Scam Address Sources

### BSC
- BscScan Labeled Addresses
- Community reports
- DeFi hack databases

### opBNB
- opBNB BscScan Labels
- Cross-chain scam tracking

### Ethereum
- Etherscan Phishing Labels
- XBlock Phishing Database
- ChainAbuse Reports
- EtherScamDB

---

## Example: Full Pipeline on BSC

```bash
# 1. Configure network
cat > .env << EOF
ACTIVE_NETWORK=BSC
BSC_RPC_URL=https://bnb-mainnet.g.alchemy.com/v2/your_key_here
BSCSCAN_API_KEY=your_bscscan_key_here
START_BLOCK=40000000
END_BLOCK=latest
MAX_ADDRESSES=1000
EOF

# 2. Validate configuration
python src/config.py

# 3. Run full pipeline
python src/main.py --full-pipeline

# 4. Check results
ls -lh models/
```

---

## Troubleshooting

### Issue: "Failed to connect to RPC"
**Solution**: Check RPC URL is correct for the active network

### Issue: "API error: Invalid API Key"
**Solution**: Verify you're using the correct API key (BscScan for BSC/opBNB, Etherscan for ETH)

### Issue: "No transactions found"
**Solution**: Adjust START_BLOCK to a more recent block number

### Issue: "Rate limit exceeded"
**Solution**: Reduce MAX_ADDRESSES or add delays between requests

---

## Performance Comparison

| Network | Block Time | Avg Gas Fee | TPS | Data Fetch Speed |
|---------|-----------|-------------|-----|------------------|
| BSC     | ~3s       | ~$0.10      | ~60 | ⭐⭐⭐⭐         |
| opBNB   | <1s       | ~$0.001     | ~4000 | ⭐⭐⭐⭐⭐       |
| Ethereum| ~12s      | ~$5-50      | ~15 | ⭐⭐⭐           |

---

## Best Practices

1. **Start with Recent Blocks**: Faster fetching, more relevant data
2. **Use Network-Specific Caches**: Keep datasets separate
3. **Monitor API Rate Limits**: Implement exponential backoff
4. **Validate Configuration**: Run `python src/config.py` before pipeline
5. **Test with Small Samples**: Use `MAX_ADDRESSES=10` for testing

---

## Next Steps

1. **Expand Scam Address Database**: Add network-specific known scam addresses
2. **Cross-Network Analysis**: Compare fraud patterns across networks
3. **Multi-Network Training**: Train models on combined datasets
4. **Real-Time Detection**: Implement WebSocket monitoring

---

**For more details, see the main README.md and documentation.**
