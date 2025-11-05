"""
Network graph analysis for Ethereum transaction networks.
Uses NetworkX to extract graph-based features from transaction flows.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Dict, Set, Tuple, Optional
from tqdm import tqdm

from config import (
    PROCESSED_DATA_DIR,
    BETWEENNESS_SAMPLE_SIZE,
    PAGERANK_ALPHA,
    PAGERANK_MAX_ITER,
)
from utils import setup_logger, save_pickle, load_pickle

logger = setup_logger(__name__)


class TransactionGraphAnalyzer:
    """
    Analyze Ethereum transaction networks using graph theory.
    Extracts centrality measures, community detection, and risk propagation features.
    """
    
    def __init__(self):
        """Initialize graph analyzer."""
        self.G = None
        self.scam_addresses = set()
    
    def build_transaction_graph(
        self, 
        df: pd.DataFrame,
        weight_col: str = 'value_eth'
    ) -> nx.DiGraph:
        """
        Build directed graph from transaction dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            Transaction dataframe with 'from', 'to', and value columns
        weight_col : str
            Column to use as edge weight
        
        Returns
        -------
        nx.DiGraph
            Directed transaction graph
        
        References
        ----------
        NetworkX DiGraph: https://networkx.org/documentation/stable/reference/classes/digraph.html
        """
        logger.info("Building transaction graph...")
        
        G = nx.DiGraph()
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding edges"):
            G.add_edge(
                row['from'],
                row['to'],
                weight=row.get(weight_col, 1.0),
                timestamp=row.get('timestamp'),
                tx_hash=row.get('hash')
            )
        
        self.G = G
        
        logger.info(f"✅ Graph created:")
        logger.info(f"   Nodes (addresses): {G.number_of_nodes()}")
        logger.info(f"   Edges (transactions): {G.number_of_edges()}")
        logger.info(f"   Density: {nx.density(G):.6f}")
        
        return G
    
    def extract_centrality_features(
        self, 
        addresses: List[str]
    ) -> pd.DataFrame:
        """
        Extract centrality measures for addresses.
        
        Parameters
        ----------
        addresses : list of str
            Addresses to compute features for
        
        Returns
        -------
        pd.DataFrame
            DataFrame with centrality features
        
        Features Generated
        ------------------
        - degree_centrality: Normalized degree (connections / total nodes)
        - in_degree_centrality: Incoming connections centrality
        - out_degree_centrality: Outgoing connections centrality
        - betweenness_centrality: Node betweenness (bridge importance)
        - pagerank_score: Google PageRank score
        
        References
        ----------
        Centrality measures: https://networkx.org/documentation/stable/reference/algorithms/centrality.html
        PageRank: Page, L., et al. (1999). "The PageRank Citation Ranking"
        """
        if self.G is None:
            logger.error("Graph not built. Call build_transaction_graph() first.")
            return pd.DataFrame()
            
        logger.info("Computing centrality features...")
        
        if self.G is None:
            raise ValueError("Graph not built. Call build_transaction_graph() first.")
        
        # Degree centrality
        logger.info("   Computing degree centrality...")
        degree_cent = nx.degree_centrality(self.G)
        in_degree_cent = nx.in_degree_centrality(self.G)
        out_degree_cent = nx.out_degree_centrality(self.G)
        
        # Betweenness centrality (sample for large graphs)
        logger.info(f"   Computing betweenness centrality (k={BETWEENNESS_SAMPLE_SIZE})...")
        if self.G.number_of_nodes() > BETWEENNESS_SAMPLE_SIZE:
            between_cent = nx.betweenness_centrality(
                self.G, 
                k=BETWEENNESS_SAMPLE_SIZE,
                seed=42
            )
        else:
            between_cent = nx.betweenness_centrality(self.G)
        
        # PageRank
        logger.info("   Computing PageRank...")
        pagerank = nx.pagerank(
            self.G, 
            alpha=PAGERANK_ALPHA,
            max_iter=PAGERANK_MAX_ITER
        )
        
        # Map to addresses
        features = []
        for addr in addresses:
            features.append({
                'address': addr,
                'degree_centrality': degree_cent.get(addr, 0),
                'in_degree_centrality': in_degree_cent.get(addr, 0),
                'out_degree_centrality': out_degree_cent.get(addr, 0),
                'betweenness_centrality': between_cent.get(addr, 0),
                'pagerank_score': pagerank.get(addr, 0),
            })
        
        df_centrality = pd.DataFrame(features)
        logger.info(f"✅ Computed centrality for {len(df_centrality)} addresses")
        
        return df_centrality
    
    def extract_clustering_features(
        self, 
        addresses: List[str]
    ) -> pd.DataFrame:
        """
        Extract clustering and local structure features.
        
        Parameters
        ----------
        addresses : list of str
            Addresses to compute features for
        
        Returns
        -------
        pd.DataFrame
            DataFrame with clustering features
        
        Features Generated
        ------------------
        - clustering_coefficient: Local clustering coefficient
        - avg_neighbor_degree: Average degree of neighbors
        - triangles: Number of triangles node participates in
        
        References
        ----------
        Clustering: https://networkx.org/documentation/stable/reference/algorithms/clustering.html
        """
        if self.G is None:
            logger.error("Graph not built. Call build_transaction_graph() first.")
            return pd.DataFrame()
            
        logger.info("Computing clustering features...")
        
        # Convert to undirected for clustering
        G_undirected = self.G.to_undirected()
        
        # Clustering coefficient
        logger.info("   Computing clustering coefficient...")
        clustering_dict = nx.clustering(G_undirected)
        
        # Average neighbor degree
        logger.info("   Computing average neighbor degree...")
        avg_neighbor_deg = nx.average_neighbor_degree(G_undirected)
        
        # Triangles
        logger.info("   Computing triangles...")
        triangles = nx.triangles(G_undirected)
        
        # Map to addresses
        features = []
        for addr in addresses:
            features.append({
                'address': addr,
                'clustering_coefficient': clustering_dict.get(addr, 0),
                'avg_neighbor_degree': avg_neighbor_deg.get(addr, 0),
                'triangles': triangles.get(addr, 0),
            })
        
        df_clustering = pd.DataFrame(features)
        logger.info(f"✅ Computed clustering for {len(df_clustering)} addresses")
        
        return df_clustering
    
    def extract_community_features(
        self, 
        addresses: List[str]
    ) -> pd.DataFrame:
        """
        Extract community detection features.
        
        Parameters
        ----------
        addresses : list of str
            Addresses to compute features for
        
        Returns
        -------
        pd.DataFrame
            DataFrame with community features
        
        Features Generated
        ------------------
        - community_id: Community identifier from Louvain algorithm
        - community_size: Size of node's community
        
        References
        ----------
        Louvain method: https://networkx.org/documentation/stable/reference/algorithms/community.html
        Blondel et al. (2008). "Fast unfolding of communities in large networks"
        """
        if self.G is None:
            logger.error("Graph not built. Call build_transaction_graph() first.")
            return pd.DataFrame()
            
        logger.info("Detecting communities (Louvain algorithm)...")
        
        # Convert to undirected
        G_undirected = self.G.to_undirected()
        
        # Greedy modularity communities
        from networkx.algorithms import community
        communities_gen = community.greedy_modularity_communities(G_undirected)
        communities = list(communities_gen)  # Convert generator to list
        
        logger.info(f"   Found {len(communities)} communities")
        
        # Create mapping
        addr_to_community = {}
        community_sizes = {}
        
        for comm_id, comm in enumerate(communities):
            community_sizes[comm_id] = len(comm)
            for addr in comm:
                addr_to_community[addr] = comm_id
        
        # Map to addresses
        features = []
        for addr in addresses:
            comm_id = addr_to_community.get(addr, -1)
            features.append({
                'address': addr,
                'community_id': comm_id,
                'community_size': community_sizes.get(comm_id, 0),
            })
        
        df_community = pd.DataFrame(features)
        logger.info(f"✅ Computed communities for {len(df_community)} addresses")
        
        return df_community
    
    def extract_risk_propagation_features(
        self, 
        addresses: List[str], 
        known_scam_addresses: List[str]
    ) -> pd.DataFrame:
        """
        Extract risk propagation features based on proximity to known scam addresses.
        
        Parameters
        ----------
        addresses : List[str]
            List of Ethereum addresses to analyze
        known_scam_addresses : List[str]
            List of known scam/fraud addresses
            
        Returns
        -------
        pd.DataFrame
            Features with columns: address, shortest_path_to_scam, scam_neighbor_count, etc.
        """
        if self.G is None:
            logger.error("Graph not built. Call build_transaction_graph() first.")
            return pd.DataFrame()
            
        logger.info(f"Extracting risk propagation features for {len(addresses)} addresses...")
        
        # Filter scam addresses that exist in graph
        scam_in_graph = [addr for addr in known_scam_addresses if addr in self.G]
        logger.info(f"   Found {len(scam_in_graph)} scam addresses in graph")
        
        if len(scam_in_graph) == 0:
            logger.warning("No scam addresses in graph. Returning default features.")
            # Return default features
            features = []
            for addr in addresses:
                features.append({
                    'address': addr,
                    'shortest_path_to_scam': 999,  # Large number = no connection
                    'scam_neighbor_count': 0,
                    'direct_scam_connection': 0,
                })
            return pd.DataFrame(features)
        
        # Extract risk features
        features = []
        for addr in addresses:
            # Shortest path to any scam address
            min_path_length = 999
            for scam_addr in scam_in_graph:
                try:
                    if nx.has_path(self.G, addr, scam_addr):
                        path_len = nx.shortest_path_length(self.G, addr, scam_addr)
                        min_path_length = min(min_path_length, path_len)
                except nx.NetworkXNoPath:
                    pass
            
            # Count scam neighbors
            neighbors = list(self.G.successors(addr)) if addr in self.G else []
            scam_neighbor_count = len(set(neighbors) & set(scam_in_graph))
            
            # Direct connection to scam
            direct_scam = 1 if scam_neighbor_count > 0 else 0
            
            features.append({
                'address': addr,
                'shortest_path_to_scam': min_path_length,
                'scam_neighbor_count': scam_neighbor_count,
                'direct_scam_connection': direct_scam,
            })
        
        df_risk = pd.DataFrame(features)
        logger.info(f"✅ Computed risk propagation for {len(df_risk)} addresses")
        
        return df_risk
    
    def extract_all_graph_features(
        self,
        df: pd.DataFrame,
        scam_addresses: List[str]
    ) -> pd.DataFrame:
        """
        Extract all graph-based features and merge with transaction data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Transaction dataframe
        scam_addresses : list of str
            Known scam addresses
        
        Returns
        -------
        pd.DataFrame
            DataFrame with all graph features added
        """
        logger.info("Extracting all graph features...")
        
        # Build graph
        self.build_transaction_graph(df)
        
        # Get unique addresses
        unique_addresses = list(set(df['from'].unique()) | set(df['to'].unique()))
        logger.info(f"Processing {len(unique_addresses)} unique addresses")
        
        # Extract all feature types
        df_centrality = self.extract_centrality_features(unique_addresses)
        df_clustering = self.extract_clustering_features(unique_addresses)
        df_community = self.extract_community_features(unique_addresses)
        df_risk = self.extract_risk_propagation_features(unique_addresses, scam_addresses)
        
        # Merge all features
        df_graph = df_centrality.copy()
        for df_feature in [df_clustering, df_community, df_risk]:
            df_graph = df_graph.merge(df_feature, on='address', how='left')
        
        # Merge with original dataframe (on 'from' address)
        df_final = df.merge(
            df_graph, 
            left_on='from', 
            right_on='address', 
            how='left',
            suffixes=('', '_graph')
        )
        
        # Fill NaN values
        graph_feature_cols = [col for col in df_graph.columns if col != 'address']
        df_final[graph_feature_cols] = df_final[graph_feature_cols].fillna(0)
        
        logger.info(f"✅ Added {len(graph_feature_cols)} graph features")
        
        return df_final


def main():
    """Main execution function for testing."""
    from config import RAW_DATA_DIR
    
    # Load data
    raw_file = RAW_DATA_DIR / "transactions_raw.csv"
    if not raw_file.exists():
        logger.error(f"Raw data not found: {raw_file}")
        return
    
    df = pd.read_csv(raw_file)
    logger.info(f"Loaded {len(df)} transactions")
    
    # Example scam addresses
    scam_addresses = [
        "0x1234567890123456789012345678901234567890",
    ]
    
    # Extract graph features
    analyzer = TransactionGraphAnalyzer()
    df_with_graph = analyzer.extract_all_graph_features(df, scam_addresses)
    
    # Save
    output_file = PROCESSED_DATA_DIR / "features_with_graph.csv"
    df_with_graph.to_csv(output_file, index=False)
    logger.info(f"✅ Saved to {output_file}")
    
    return df_with_graph


if __name__ == "__main__":
    main()
