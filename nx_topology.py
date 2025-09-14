# nx_topology.py
from __future__ import annotations
import numpy as np
import networkx as nx

def build_ring_knn_graph(X: np.ndarray, k: int = 3) -> nx.Graph:
    """
    Build an undirected graph with ring + symmetric kNN edges over nodes 0..N-1.
    X: (N, F) telemetry feature matrix used only for neighbor selection.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (N, F)")
    N = X.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))
    
    # Ring
    if N >= 2:
        for i in range(N):
            G.add_edge(i, (i + 1) % N)
    # kNN by Euclidean distance on X (row-wise)
    if N >= 2 and k > 0:
        from sklearn.metrics.pairwise import euclidean_distances
        D = euclidean_distances(X)  # (N, N) distances
        for i in range(N):
            nbrs = np.argsort(D[i])[1:min(k+1, N)]  # skip self
            for j in nbrs:
                G.add_edge(i, j)
    return G

def compute_nx_features(G: nx.Graph) -> np.ndarray:
    """
    Return (N, 4) matrix with columns:
    [degree_centrality, betweenness_centrality, eigenvector_centrality, clustering].
    """
    N = G.number_of_nodes()
    deg = nx.degree_centrality(G)                           # normalized degree
    btw = nx.betweenness_centrality(G, normalized=True)     # standard betweenness
    try:
        eig = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)  # spectral influence
    except Exception:
        eig = {n: 0.0 for n in G.nodes()}
    clus = nx.clustering(G)                                  # triangle-based clustering
    M = np.zeros((N, 4), dtype=np.float32)
    for n in range(N):
        M[n, 0] = float(deg.get(n, 0.0))
        M[n, 1] = float(btw.get(n, 0.0))
        M[n, 2] = float(eig.get(n, 0.0))
        M[n, 3] = float(clus.get(n, 0.0))
    return M

def nx_to_edge_index(G: nx.Graph) -> np.ndarray:
    """
    Convert to COO edge_index with both directions: shape (2, E).
    """
    edges = []
    for u, v in G.edges():
        edges.append((u, v))
        edges.append((v, u))
    if not edges:
        return np.empty((2, 0), dtype=np.int64)
    return np.array(edges, dtype=np.int64).T
