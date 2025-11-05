# 📓 Ethereum Fraud Detection - Notebooks

This directory contains Jupyter notebooks for comprehensive data analysis and visualization.

## 📚 Available Notebooks

### 02_Data_Analysis_Visualization.ipynb
**Complete Data Analysis & Visualization Pipeline**

A comprehensive notebook covering the entire ML pipeline from raw data to model evaluation.

#### 📊 Contents:

**Part 1: Raw Data Exploration**
- Transaction timeline analysis
- Value & gas distribution
- Temporal patterns (heatmap: day × hour)

**Part 2: Feature Engineering Analysis**
- Feature correlation matrix
- Feature distribution plots (16 features)
- PCA & t-SNE dimensionality reduction

**Part 3: Network Graph Analysis**
- Network statistics (nodes, edges, density)
- Centrality measures (degree, betweenness, PageRank)
- Interactive network visualization (top 50 nodes)
- Graph feature distributions

**Part 4: Model Training & Performance**
- XGBoost model analysis
- Feature importance (weight, gain, cover)

**Part 5: SHAP Explainability**
- SHAP summary plots (bar & beeswarm)
- SHAP dependence plots (top 4 features)
- Global & local explanations

**Part 6: Model Evaluation Results**
- Performance metrics dashboard
- Confusion matrix
- PR-AUC & ROC curves
- Feature importance visualization

**Part 7: Summary & Conclusions**
- Key findings
- Recommendations

#### 🎨 Generated Visualizations:

1. `viz_01_value_distribution.png` - Transaction value & gas analysis (4 subplots)
2. `viz_02_temporal_heatmap.png` - Activity heatmap (day × hour)
3. `viz_03_correlation_matrix.png` - Feature correlation matrix
4. `viz_04_feature_distributions.png` - 16 feature distributions
5. `viz_05_dimensionality_reduction.png` - PCA & t-SNE plots
6. `viz_06_network_centrality.png` - Centrality distributions (3 types)
7. `viz_07_graph_features.png` - Graph feature distributions
8. `viz_08_feature_importance_xgboost.png` - XGBoost importance (3 types)
9. `viz_09_shap_summary_bar.png` - SHAP global importance (bar)
10. `viz_10_shap_beeswarm.png` - SHAP impact visualization
11. `viz_11_shap_dependence.png` - SHAP dependence plots (top 4)

All visualizations are saved to `../models/` directory.

#### 🚀 Usage:

```bash
# Start Jupyter from notebooks directory
cd notebooks
jupyter notebook 02_Data_Analysis_Visualization.ipynb

# Or use VS Code Jupyter extension
# Open the .ipynb file directly in VS Code

# Or use Jupyter Lab
jupyter lab 02_Data_Analysis_Visualization.ipynb
```

#### 📦 Requirements:

All dependencies are installed from `../requirements.txt`:
- pandas, numpy
- matplotlib, seaborn, plotly
- networkx
- scikit-learn
- xgboost, shap

#### ⚡ Performance Notes:

- t-SNE uses max 1000 samples for speed
- Network visualization shows top 50 nodes
- SHAP analysis uses 100 samples
- Betweenness centrality samples k=100 nodes

#### 📝 Academic Use:

This notebook is designed for thesis/research documentation:
- ✅ All plots saved in high resolution (300 DPI)
- ✅ Comprehensive statistics printed
- ✅ Reproducible results (random_state=42)
- ✅ Publication-ready visualizations
- ✅ Step-by-step explanations
- ✅ Interactive plots with Plotly

#### 🎯 Recommended Workflow:

1. **Run Full Pipeline First**:
   ```bash
   python src/main.py --full-pipeline
   ```

2. **Open Notebook**:
   ```bash
   jupyter notebook notebooks/02_Data_Analysis_Visualization.ipynb
   ```

3. **Execute Cells**: Run all cells sequentially (Cell → Run All)

4. **Review Visualizations**: Check `models/viz_*.png` files

5. **Export for Thesis**: Use File → Download as → PDF/HTML

---

## 📊 Visualization Examples

### Transaction Timeline
Shows daily and cumulative transaction counts over time.

### Network Graph
Interactive visualization of top 50 nodes with centrality-based sizing and coloring.

### SHAP Analysis
Comprehensive feature importance with global and local explanations.

### Feature Correlations
Heatmap showing relationships between engineered features.

---

**For questions or issues, refer to the main README.md in the project root.**
