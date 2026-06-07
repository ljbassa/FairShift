import numpy as np
import scipy.sparse as sp
import networkx as nx


import numpy as np
import scipy.sparse as sp
import networkx as nx
import powerlaw


def max_degree(A):
    """
    Compute the maximum degree.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Maximum degree.
    """
    try:
        degrees = A.sum(axis=-1)
        return np.max(degrees)
    except:
        return np.nan


def min_degree(A):
    """
    Compute the minimum degree.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Minimum degree.
    """
    try:
        degrees = A.sum(axis=-1)
        return np.min(degrees)
    except:
        return np.nan


def average_degree(A):
    """
    Compute the average degree.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Average degree.
    """
    try:
        degrees = A.sum(axis=-1)
        return np.mean(degrees)
    except:
        return np.nan


def LCC(A):
    """
    Compute the size of the largest connected component (LCC).
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Size of the largest connected component.
    """
    try:
        G = nx.from_scipy_sparse_array(A)
        return max([len(c) for c in nx.connected_components(G)])
    except:
        return np.nan


def wedge_count(A):
    """
    Compute the wedge count.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Wedge count.
    """
    try:
        degrees = np.array(A.sum(axis=-1))
        return 0.5 * np.dot(degrees.T, degrees - 1).reshape([]) 
    except:
        return np.nan


def claw_count(A):
    """
    Compute the claw count.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Claw count.
    """
    try:
        degrees = np.array(A.sum(axis=-1))
        return 1 / 6 * np.sum(degrees * (degrees - 1) * (degrees - 2))
    except:
        return np.nan


def triangle_count(A):
    """
    Compute the triangle count.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Triangle count.
    """
    try:
        A_graph = nx.from_scipy_sparse_array(A)
        triangles = nx.triangles(A_graph)
        t = np.sum(list(triangles.values())) / 3
        return int(t)
    except:
        return np.nan


def square_count(A):
    """
    Compute the square count.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Square count.
    """
    try:
        A_squared = A @ A
        common_neighbors = sp.triu(A_squared, k=1).tocsr()
        num_common_neighbors = np.array(
            common_neighbors[common_neighbors.nonzero()]
        ).reshape(-1)
        return np.dot(num_common_neighbors, num_common_neighbors - 1) / 4
    except:
        return np.nan


def power_law_alpha(A):
    """
    Compute the power law coefficient of the degree distribution of the input graph.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Power law coefficient.
    """
    try:
        degrees = np.array(A.sum(axis=-1)).flatten()
        return powerlaw.Fit(
            degrees, xmin=max(np.min(degrees), 1), verbose=False
        ).power_law.alpha
    except:
        return np.nan


def gini(A):
    """
    Compute the Gini coefficient of the degree distribution of the input graph.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Gini coefficient.
    """
    try:
        N = A.shape[0]
        degrees_sorted = np.sort(np.array(A.sum(axis=-1)).flatten())
        return (
            2 * np.dot(degrees_sorted, np.arange(1, N + 1)) / (N * np.sum(degrees_sorted))
            - (N + 1) / N
        )
    except:
        return np.nan


def edge_distribution_entropy(A):
    """
    Compute the relative edge distribution entropy of the input graph.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Relative edge distribution entropy.
    """
    try:
        N = A.shape[0]
        degrees = np.array(A.sum(axis=-1)).flatten()
        degrees /= degrees.sum()
        return -np.dot(np.log(degrees), degrees) / np.log(N)
    except:
        return np.nan


def assortativity(A):
    """
    Compute the assortativity of the input graph.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Assortativity.
    """
    try:
        G = nx.from_scipy_sparse_array(A)
        return nx.degree_assortativity_coefficient(G)
    except:
        return np.nan


def clustering_coefficient(A):
    """
    Compute the clustering coefficient of the input graph.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Clustering coefficient.
    """
    try:
        return 3 * triangle_count(A) / wedge_count(A)   
    except:
        return np.nan


def cpl(A):
    """
    Compute the characteristic path length of the input graph.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Characteristic path length.
    """
    try:
        P = sp.csgraph.shortest_path(A)
        return P[((1 - np.isinf(P)) * (1 - np.eye(P.shape[0]))).astype(np.bool)].mean()
    except:
        return np.nan


def compute_graph_statistics(A):
    """
    Compute a selection of graph statistics for the input graph.
    
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
          
    Returns:
        Dictionary containing the following statistics:
                 * Maximum, minimum, mean degree of nodes
                 * Size of the largest connected component (LCC)
                 * Wedge count
                 * Claw count
                 * Triangle count
                 * Square count
                 * Power law exponent
                 * Gini coefficient
                 * Relative edge distribution entropy
                 * Assortativity
                 * Clustering coefficient
                 * Characteristic path length
    """
    # statistics = {"d_max": max_degree(A),
    #               "d_min": min_degree(A),
    #               "d": average_degree(A),
    #               "LCC": LCC(A),
    #               "wedge_count": wedge_count(A),
    #               "claw_count": claw_count(A),
    #               "triangle_count": triangle_count(A),
    #               "square_count": square_count(A),
    #               "power_law_exp": power_law_alpha(A),
    #               "gini": gini(A),
    #               "rel_edge_distr_entropy": edge_distribution_entropy(A),
    #               "assortativity": assortativity(A),
    #               "clustering_coefficient": clustering_coefficient(A),
    #               "cpl": cpl(A)}

    statistics = {#"d_max": max_degree(A),
                  #"d_min": min_degree(A),
                  "d": average_degree(A),
                  #"LCC": LCC(A),
                  #"wedge_count": wedge_count(A),
                  #"claw_count": claw_count(A),
                  "triangle_count": triangle_count(A),
                  #"square_count": square_count(A),
                  #"power_law_exp": power_law_alpha(A),
                  #"gini": gini(A),
                  #"rel_edge_distr_entropy": edge_distribution_entropy(A),
                  #"assortativity": assortativity(A),
                  "clustering_coefficient": clustering_coefficient(A),
                  #"cpl": cpl(A)} 
    }
    return statistics
    
def edge_overlap(A, B):
    """
    Compute edge overlap between two graphs (amount of shared edges).
    Args:
        A (sp.csr.csr_matrix): First input adjacency matrix.
        B (sp.csr.csr_matrix): Second input adjacency matrix.
    Returns:
        Edge overlap.
    """
    try:
        return A.multiply(B).sum() / 2
    except:
        return np.nan
    
# Additional statistic: AUC

def _auc_binary_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    AUC for binary scores in {0,1} without sklearn.
    AUC = P(score_pos > score_neg) + 0.5 P(score_pos == score_neg)
    """
    labels = labels.astype(np.int32)
    scores = scores.astype(np.int32)

    P = int(labels.sum())
    N = int(len(labels) - P)
    if P == 0 or N == 0:
        # undefined; return nan or 0.5
        return float("nan")

    tp = int(scores[labels == 1].sum())  # positives predicted 1
    fp = int(scores[labels == 0].sum())  # negatives predicted 1

    # pos=1,neg=0 => tp*(N-fp)
    # ties: (pos=1,neg=1) => tp*fp, (pos=0,neg=0) => (P-tp)*(N-fp)
    auc = (tp * (N - fp) + 0.5 * (tp * fp + (P - tp) * (N - fp))) / (P * N)
    return float(auc)


def linkpred_auc_from_pairs(g_nx, edge_pairs, labels) -> float:
    """
    g_nx: generated NetworkX graph (undirected expected)
    edge_pairs: list[tuple(u,v)] in node labels space
    labels: np.ndarray of 0/1 (1 if (u,v) is an edge in reference graph)
    score is 1 if edge exists in generated graph, else 0.
    """
    # normalize undirected edges
    gen_edge_set = set()
    for u, v in g_nx.edges():
        if u == v:
            continue
        if u < v:
            gen_edge_set.add((u, v))
        else:
            gen_edge_set.add((v, u))

    scores = np.zeros(len(edge_pairs), dtype=np.int32)
    for i, (u, v) in enumerate(edge_pairs):
        if u == v:
            scores[i] = 0
            continue
        e = (u, v) if u < v else (v, u)
        scores[i] = 1 if e in gen_edge_set else 0

    labels = np.asarray(labels, dtype=np.int32)
    return _auc_binary_scores(labels, scores)
