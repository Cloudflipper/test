"""Graph adjacency matrix construction and normalization."""
import torch
import numpy as np
from typing import Optional, Literal


def build_synthetic_adjacency(num_nodes: int, seed: int = 42) -> torch.Tensor:
    """Build synthetic adjacency matrix.
    
    Creates a random graph with some connectivity.
    
    Args:
        num_nodes: Number of nodes
        seed: Random seed
        
    Returns:
        Adjacency matrix [num_nodes, num_nodes]
    """
    np.random.seed(seed)
    
    # Create random sparse adjacency matrix
    adj = np.random.rand(num_nodes, num_nodes)
    # Make it symmetric
    adj = (adj + adj.T) / 2
    # Threshold to create sparsity
    threshold = np.percentile(adj, 85)
    adj = (adj > threshold).astype(float)
    
    return torch.FloatTensor(adj)


def normalize_adjacency(
    adj: torch.Tensor,
    self_loop: bool = True,
    normalize: Optional[Literal["symmetric", "asymmetric"]] = "symmetric"
) -> torch.Tensor:
    """Normalize adjacency matrix.
    
    Args:
        adj: Adjacency matrix [N, N]
        self_loop: Whether to add self-loops
        normalize: Normalization method ("symmetric", "asymmetric", or None)
        
    Returns:
        Normalized adjacency matrix [N, N]
    """
    adj = adj.clone()
    
    # Add self-loops
    if self_loop:
        adj = adj + torch.eye(adj.shape[0], device=adj.device)
    
    # Normalize
    if normalize == "symmetric":
        # D^(-1/2) * A * D^(-1/2)
        degree = torch.sum(adj, dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        degree_matrix = torch.diag(degree_inv_sqrt)
        adj_normalized = torch.matmul(torch.matmul(degree_matrix, adj), degree_matrix)
    elif normalize == "asymmetric":
        # D^(-1) * A
        degree = torch.sum(adj, dim=1, keepdim=True)
        degree = torch.clamp(degree, min=1e-8)
        adj_normalized = adj / degree
    elif normalize is None:
        adj_normalized = adj
    else:
        raise ValueError(f"Unknown normalization method: {normalize}")
    
    return adj_normalized


def build_graph(
    num_nodes: Optional[int] = None,
    graph_type: str = "synthetic",
    self_loop: bool = True,
    normalize: Optional[Literal["symmetric", "asymmetric"]] = "symmetric",
    seed: int = 42,
    return_raw: bool = False,
    adj_path: Optional[str] = None
) -> torch.Tensor:
    """Build and normalize graph adjacency matrix.
    
    Args:
        num_nodes: Number of nodes (for synthetic graphs)
        graph_type: Type of graph to build ("synthetic" or "los")
        self_loop: Whether to add self-loops (for synthetic graphs)
        normalize: Normalization method (for synthetic graphs)
        seed: Random seed (for synthetic graphs)
        return_raw: If True, return raw adjacency without normalization (for synthetic graphs)
        adj_path: Path to adjacency CSV file (for real datasets like LOS)
        
    Returns:
        Adjacency matrix [N, N]
    """
    if graph_type == "synthetic":
        if num_nodes is None:
            raise ValueError("num_nodes required for synthetic graph")
        adj = build_synthetic_adjacency(num_nodes, seed=seed)
        if return_raw:
            # Return raw adjacency (no normalization, no self-loops)
            return adj
        adj_normalized = normalize_adjacency(adj, self_loop=self_loop, normalize=normalize)
        return adj_normalized
    elif graph_type == "los":
        if adj_path is None:
            adj_path = "./data/los_adj.csv"
        # Load pre-normalized adjacency from CSV (no further normalization)
        import pandas as pd
        adj_df = pd.read_csv(adj_path, header=None)
        adj = torch.FloatTensor(adj_df.values)  # [N, N]
        # LOS adjacency is already normalized, use directly
        return adj
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

