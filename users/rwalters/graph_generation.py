import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import networkx as nx
import numpy as np


def generate_random_directed_graph(n: int, m: int, allow_self_loops: bool = False) -> nx.DiGraph:
    """
    Generate a random directed graph with n nodes and m edges.

    Args:
        n: Number of nodes
        m: Number of edges
        allow_self_loops: Whether to allow self-loops (edges from a node to itself)

    Returns:
        A NetworkX DiGraph
    """
    # Validate inputs
    max_edges = n * n if allow_self_loops else n * (n - 1)
    if m > max_edges:
        raise ValueError(f"Too many edges requested. Maximum is {max_edges} for {n} nodes.")

    # Create empty directed graph with n nodes
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # Add m random edges
    edges_added = 0
    attempts = 0
    max_attempts = m * 10  # Avoid infinite loops

    while edges_added < m and attempts < max_attempts:
        source = random.randint(0, n - 1)
        target = random.randint(0, n - 1)

        # Skip self-loops if not allowed
        if not allow_self_loops and source == target:
            attempts += 1
            continue

        # Skip if edge already exists
        if G.has_edge(source, target):
            attempts += 1
            continue

        # Add edge
        G.add_edge(source, target)
        edges_added += 1

    if edges_added < m:
        print(f"Warning: Could only add {edges_added} edges out of {m} requested.")

    return G


def generate_erdos_renyi_directed_graph(n: int, m: int) -> nx.DiGraph:
    """
    Generate a directed graph using the Erdős–Rényi model with n nodes and approximately m edges.

    Args:
        n: Number of nodes
        m: Target number of edges
        allow_self_loops: Whether to allow self-loops

    Returns:
        A NetworkX DiGraph
    """
    # Calculate probability
    max_edges = n * (n - 1)
    p = m / max_edges

    # Generate graph
    G = nx.gnp_random_graph(n, p, directed=True)

    print(f"Generated graph with {G.number_of_edges()} edges (target was {m}).")
    return G


def generate_preferential_attachment_directed_graph(n: int, m_per_node: int) -> nx.DiGraph:
    """
    Generate a scale-free directed graph using preferential attachment.

    Args:
        n: Final number of nodes
        m_per_node: Number of edges to attach from each new node

    Returns:
        A NetworkX DiGraph
    """
    if m_per_node >= n:
        raise ValueError("m_per_node must be less than n")

    # Start with a complete directed graph of m_per_node+1 nodes
    G = nx.complete_graph(m_per_node + 1, create_using=nx.DiGraph())

    # Add remaining nodes with preferential attachment
    for i in range(m_per_node + 1, n):
        # Calculate the current in-degrees for existing nodes
        in_degrees = [G.in_degree(node) + 1 for node in G.nodes()]  # +1 to avoid zero probability
        total_in_degree = sum(in_degrees)

        # Normalize to get probabilities
        probs = [d / total_in_degree for d in in_degrees]

        # Select targets for the new node's edges
        targets = np.random.choice(list(G.nodes()), size=min(m_per_node, len(G.nodes())), replace=False, p=probs)

        # Add the new node and its edges
        G.add_node(i)
        for target in targets:
            G.add_edge(i, target)

    print(f"Generated graph with {G.number_of_edges()} edges.")
    return G


def ensure_sources(G: nx.DiGraph, n: int) -> nx.DiGraph:
    """
    Ensure the graph has exactly N source nodes (nodes with no incoming edges).
    If there are fewer than N sources, create more by removing appropriate edges.
    If there are more than N sources, add edges to reduce the number to N.

    Args:
        G: NetworkX directed graph
        n: Desired number of source nodes

    Returns:
        Modified NetworkX DiGraph with exactly N sources
    """
    # Find nodes with no incoming edges (current sources)
    sources = [node for node in G.nodes() if G.in_degree(node) == 0]

    # If we already have exactly N sources, we're done
    if len(sources) == n:
        print(f"Graph already has exactly {n} sources: {sources}")
        return G

    # If we have fewer than N sources, create more
    elif len(sources) < n:
        needed = n - len(sources)
        print(f"Need to create {needed} more sources")

        # Find non-source nodes sorted by in-degree (fewer edges to remove first)
        non_sources = sorted([node for node in G.nodes() if node not in sources], key=lambda n: G.in_degree(n))

        # Create new sources from the nodes with fewest incoming edges
        for i in range(min(needed, len(non_sources))):
            node = non_sources[i]
            incoming_edges = list(G.in_edges(node))
            G.remove_edges_from(incoming_edges)
            print(f"Created source at node {node} by removing {len(incoming_edges)} incoming edges")

    # If we have more than N sources, reduce them
    else:
        excess = len(sources) - n
        print(f"Need to remove {excess} sources")

        # Sort sources by out-degree (more edges = more important to keep as source)
        sorted_sources = sorted(sources, key=lambda n: G.out_degree(n))

        # For each excess source, add an edge from another node
        for i in range(excess):
            node_to_modify = sorted_sources[i]

            # Find a potential target node (that won't create a cycle)
            potential_targets = [n for n in G.nodes() if n != node_to_modify and not nx.has_path(G, node_to_modify, n)]

            if potential_targets:
                target = random.choice(potential_targets)
                G.add_edge(target, node_to_modify)
                print(f"Added edge from {target} to {node_to_modify} to remove a source")
            else:
                print(f"Warning: Could not find a suitable target for source {node_to_modify}")

    # Verify we have the desired number of sources
    final_sources = [node for node in G.nodes() if G.in_degree(node) == 0]
    print(f"Final number of sources: {len(final_sources)}")

    return G


def ensure_sinks(G: nx.DiGraph, n: int) -> nx.DiGraph:
    """
    Ensure the graph has exactly N sink nodes (nodes with no outgoing edges).
    If there are fewer than N sinks, create more by removing appropriate edges.
    If there are more than N sinks, add edges to reduce the number to N.

    Args:
        G: NetworkX directed graph
        n: Desired number of sink nodes

    Returns:
        Modified NetworkX DiGraph with exactly N sinks
    """
    # Find nodes with no outgoing edges (current sinks)
    sinks = [node for node in G.nodes() if G.out_degree(node) == 0]

    # If we already have exactly N sinks, we're done
    if len(sinks) == n:
        print(f"Graph already has exactly {n} sinks: {sinks}")
        return G

    # If we have fewer than N sinks, create more
    elif len(sinks) < n:
        needed = n - len(sinks)
        print(f"Need to create {needed} more sinks")

        # Find non-sink nodes sorted by out-degree (fewer edges to remove first)
        non_sinks = sorted([node for node in G.nodes() if node not in sinks], key=lambda n: G.out_degree(n))

        # Create new sinks from the nodes with fewest outgoing edges
        for i in range(min(needed, len(non_sinks))):
            node = non_sinks[i]
            outgoing_edges = list(G.out_edges(node))
            G.remove_edges_from(outgoing_edges)
            print(f"Created sink at node {node} by removing {len(outgoing_edges)} outgoing edges")

    # If we have more than N sinks, reduce them
    else:
        excess = len(sinks) - n
        print(f"Need to remove {excess} sinks")

        # Sort sinks by in-degree (more edges = more important to keep as sink)
        sorted_sinks = sorted(sinks, key=lambda n: G.in_degree(n))

        # For each excess sink, add an edge to another node
        for i in range(excess):
            node_to_modify = sorted_sinks[i]

            # Find a potential source node (that won't create a cycle)
            potential_sources = [n for n in G.nodes() if n != node_to_modify and not nx.has_path(G, n, node_to_modify)]

            if potential_sources:
                source = random.choice(potential_sources)
                G.add_edge(node_to_modify, source)
                print(f"Added edge from {node_to_modify} to {source} to remove a sink")
            else:
                print(f"Warning: Could not find a suitable source for sink {node_to_modify}")

    # Verify we have the desired number of sinks
    final_sinks = [node for node in G.nodes() if G.out_degree(node) == 0]
    print(f"Final number of sinks: {len(final_sinks)}")

    return G


class GraphType(Enum):
    RANDOM = "random"
    ERDOS_RENYI = "erdos_renyi"
    PREFERENTIAL_ATTACHMENT = "preferential_attachment"


@dataclass
class GraphConfig:
    """Configuration for graph generation"""

    # Common parameters
    graph_type: GraphType
    n_nodes: int
    n_edges: int  # For RANDOM and ERDOS_RENYI
    m_per_node: int  # For PREFERENTIAL_ATTACHMENT
    allow_self_loops: bool = False
    n_sources: Optional[int] = None
    n_sinks: Optional[int] = None

    def __post_init__(self):
        """Validate configuration parameters"""
        # Validate node count
        if self.n_nodes <= 0:
            raise ValueError("Number of nodes must be positive")

        # Validate type-specific parameters
        if self.graph_type in [GraphType.RANDOM, GraphType.ERDOS_RENYI]:
            if self.n_edges is None:
                raise ValueError(f"{self.graph_type.value} graph requires n_edges parameter")

            if self.n_edges < 0:
                raise ValueError("Number of edges must be non-negative")

            max_edges = self.n_nodes * self.n_nodes if self.allow_self_loops else self.n_nodes * (self.n_nodes - 1)
            if self.n_edges > max_edges:
                raise ValueError(f"Too many edges requested. Maximum is {max_edges} for {self.n_nodes} nodes.")

        elif self.graph_type == GraphType.PREFERENTIAL_ATTACHMENT:
            if self.m_per_node is None:
                raise ValueError("Preferential attachment graph requires m_per_node parameter")

            if self.m_per_node <= 0:
                raise ValueError("m_per_node must be positive")

            if self.m_per_node >= self.n_nodes:
                raise ValueError("m_per_node must be less than n_nodes")

        # Validate sources and sinks
        if self.n_sources is not None and (self.n_sources < 0 or self.n_sources > self.n_nodes):
            raise ValueError(f"Invalid number of sources: {self.n_sources}. Must be between 0 and {self.n_nodes}")

        if self.n_sinks is not None and (self.n_sinks < 0 or self.n_sinks > self.n_nodes):
            raise ValueError(f"Invalid number of sinks: {self.n_sinks}. Must be between 0 and {self.n_nodes}")


def generate_graph_from_config(config: GraphConfig) -> nx.DiGraph:
    """
    Generate a directed graph based on a GraphConfig object.

    Args:
        config: A GraphConfig object with parameters for graph generation

    Returns:
        A NetworkX DiGraph based on the specified configuration
    """
    # Generate base graph based on type
    if config.graph_type == GraphType.RANDOM:
        G = generate_random_directed_graph(config.n_nodes, config.n_edges, config.allow_self_loops)

    elif config.graph_type == GraphType.ERDOS_RENYI:
        G = generate_erdos_renyi_directed_graph(config.n_nodes, config.n_edges)

    elif config.graph_type == GraphType.PREFERENTIAL_ATTACHMENT:
        G = generate_preferential_attachment_directed_graph(config.n_nodes, config.m_per_node)

    else:
        raise ValueError(f"Unknown graph type: {config.graph_type}")

    # Ensure desired number of sources if specified
    if config.n_sources is not None:
        G = ensure_sources(G, config.n_sources)

    # Ensure desired number of sinks if specified
    if config.n_sinks is not None:
        G = ensure_sinks(G, config.n_sinks)

    return G
