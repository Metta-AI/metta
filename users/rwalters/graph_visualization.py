import colorsys
from typing import Any, Dict, List, Optional, Set

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D  # Import Line2D directly


def generate_distinct_colors(n: int) -> List[str]:
    """
    Generate n visually distinct colors by varying hue in HSV color space.

    Args:
        n: Number of distinct colors to generate

    Returns:
        List of hex color codes
    """
    colors = []
    for i in range(n):
        # Evenly space the hues
        h = i / n
        # Fixed saturation and value for vibrant colors
        s = 0.7
        v = 0.9
        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        # Convert to hex
        hex_color = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        colors.append(hex_color)
    return colors


def create_color_map_from_values(values: Set[Any]) -> Dict[Any, str]:
    """
    Create a mapping of unique values to distinct colors.

    Args:
        values: Set of unique values to map to colors

    Returns:
        Dictionary mapping values to hex color codes
    """
    unique_values = list(values)
    colors = generate_distinct_colors(len(unique_values))
    return {value: colors[i] for i, value in enumerate(unique_values)}


def visualize_graph(
    G: nx.DiGraph,
    title: str = "Directed Graph",
    layout: str = "spring",
    color_attribute: Optional[str] = None,
    color_map: Optional[Dict[Any, str]] = None,
    node_labels: Optional[Dict[int, str]] = None,
    show_legend: bool = True,
    node_size_factor: float = 1.0,
) -> None:
    """
    Visualize a directed graph with nodes colored based on an attribute.

    Args:
        G: NetworkX directed graph
        title: Plot title
        layout: Layout algorithm ('spring', 'circular', 'random', 'shell', 'kamada_kawai')
        color_attribute: Node attribute to use for coloring (if None, all nodes get the same color)
        color_map: Optional mapping of attribute values to colors
        node_labels: Optional custom labels for nodes (if None, node IDs are used)
        show_legend: Whether to show a legend for node colors and source/sink nodes
        node_size_factor: Factor to adjust the base node size (default sizes * factor)
    """
    plt.figure(figsize=(12, 9))

    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "random":
        pos = nx.random_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Get node colors
    if color_attribute is not None:
        # Ensure the attribute exists
        if not any(color_attribute in data for _, data in G.nodes(data=True)):
            raise ValueError(f"Graph does not have the attribute '{color_attribute}' on its nodes")

        # Get attribute values
        attr_values = nx.get_node_attributes(G, color_attribute)
        unique_values = set(attr_values.values())

        # Create or use color map
        if color_map is None:
            color_map = create_color_map_from_values(unique_values)

        # Map nodes to colors
        node_colors = {node: color_map[attr_values[node]] for node in G.nodes() if node in attr_values}
        # Default color for nodes without the attribute
        node_color_list = [node_colors.get(node, "lightblue") for node in G.nodes()]

        # Store colors as node attributes for reference
        nx.set_node_attributes(G, {node: {"color": color} for node, color in node_colors.items()})
    else:
        # Default color if no attribute is specified
        node_color_list = ["lightblue" for _ in G.nodes()]

    # Identify source and sink nodes
    sources = [node for node in G.nodes() if G.in_degree(node) == 0]
    sinks = [node for node in G.nodes() if G.out_degree(node) == 0]

    # Create node size list (larger for sources and sinks)
    base_size = 500 * node_size_factor
    large_size = 700 * node_size_factor

    # Node shapes (circle for regular nodes, square for sources, diamond for sinks)
    node_shapes = {
        "regular": "o",
        "source": "s",  # square
        "sink": "d",  # diamond
    }

    # Draw nodes by type (sources, sinks, regular) to use different shapes
    regular_nodes = [node for node in G.nodes() if node not in sources and node not in sinks]

    # For each type of node (regular, source, sink), draw nodes individually

    # Get the order of nodes in the original list for color mapping
    node_list = list(G.nodes())

    # Draw regular nodes (circles)
    for node in regular_nodes:
        # Find this node's index in the original node list
        node_idx = node_list.index(node)
        # Draw this single node with its color
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[node],  # Just this one node
            node_size=int(base_size),  # Cast to int for Pylance
            node_color=node_color_list[node_idx],  # Single color string for this node
            node_shape=node_shapes["regular"],
        )

    # Draw source nodes (squares)
    for node in sources:
        # Find this node's index in the original node list
        node_idx = node_list.index(node)
        # Draw this single node with its color
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[node],  # Just this one node
            node_size=int(large_size),  # Cast to int for Pylance
            node_color=node_color_list[node_idx],  # Single color string for this node
            node_shape=node_shapes["source"],
        )

    # Draw sink nodes (diamonds)
    for node in sinks:
        # Find this node's index in the original node list
        node_idx = node_list.index(node)
        # Draw this single node with its color
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[node],  # Just this one node
            node_size=int(large_size),  # Cast to int for Pylance
            node_color=node_color_list[node_idx],  # Single color string for this node
            node_shape=node_shapes["sink"],
        )

    # Use custom node labels if provided, otherwise use node IDs
    if node_labels is None:
        # Check if the graph has a 'label' attribute
        label_attrs = nx.get_node_attributes(G, "label")
        if label_attrs:
            node_labels = label_attrs
        else:
            node_labels = {node: str(node) for node in G.nodes()}

    nx.draw_networkx_labels(G, pos, labels=node_labels)

    # Draw edges with slight curvature for better visibility
    nx.draw_networkx_edges(
        G, pos, width=1.0, alpha=0.7, arrowsize=15, edge_color="gray", connectionstyle="arc3, rad=0.1"
    )

    # Add a legend if requested
    if show_legend and color_attribute is not None and color_map is not None:
        legend_handles = []

        # Create legend entries for attribute values
        for value, color in color_map.items():
            legend_handles.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10, label=str(value))
            )

        # Add source/sink legend entries with corresponding shapes
        if sources:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=node_shapes["source"],
                    color="w",
                    markerfacecolor="gray",
                    markersize=12,
                    label="Entry Point (Source)",
                )
            )
        if sinks:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=node_shapes["sink"],
                    color="w",
                    markerfacecolor="gray",
                    markersize=12,
                    label="Exit Point (Sink)",
                )
            )

        if legend_handles:
            plt.legend(
                handles=legend_handles,
                loc="upper right",
                title=f"Node Information ({color_attribute})" if color_attribute else "Node Information",
            )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
