import colorsys
from typing import Any, Dict, List, Optional, Set

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_segmented_edges(
    G,
    pos,
    node_color_names,
    light_grey="#cccccc",
    arrow_length=0.03,
    dot_size=30,
    node_size=500,  # Approx base node size used in drawing
):
    # Estimate node radius in data coordinates
    # This factor maps node size (area in pt²) to a visual radius
    node_radius = node_size / 20000.0

    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        # Vector from u to v
        dx, dy = x1 - x0, y1 - y0
        dist = np.hypot(dx, dy)

        if dist == 0:
            continue  # skip degenerate edge

        # Unit direction vector
        ux, uy = dx / dist, dy / dist

        # Adjust start and end to stop at node boundary
        shrink = node_radius
        x0_adj = x0 + ux * shrink
        y0_adj = y0 + uy * shrink
        x1_adj = x1 - ux * shrink
        y1_adj = y1 - uy * shrink

        # Compute segment points (25%, 75%)
        x_mid1 = x0_adj + 0.25 * (x1_adj - x0_adj)
        y_mid1 = y0_adj + 0.25 * (y1_adj - y0_adj)
        x_mid2 = x0_adj + 0.75 * (x1_adj - x0_adj)
        y_mid2 = y0_adj + 0.75 * (y1_adj - y0_adj)

        # Get colors
        src_color = node_color_names.get(v, "lightblue")
        dst_color = node_color_names.get(u, "lightblue")

        # Draw 3 segments
        plt.plot([x0_adj, x_mid1], [y0_adj, y_mid1], color=src_color, linewidth=1.5, alpha=0.7)
        plt.plot([x_mid1, x_mid2], [y_mid1, y_mid2], color=light_grey, linewidth=1.5, alpha=0.7)
        plt.plot([x_mid2, x1_adj], [y_mid2, y1_adj], color=dst_color, linewidth=1.5, alpha=0.7)

        # Start dot at adjusted source
        plt.scatter([x0_adj], [y0_adj], s=dot_size, color=src_color, edgecolors="black", zorder=3)

        # Arrowhead slightly before end (short vector)
        arrow_backoff = 0.05  # smaller value = arrow closer to end
        arrow_start_x = x1_adj - ux * arrow_backoff
        arrow_start_y = y1_adj - uy * arrow_backoff
        arrow_dx = ux * arrow_backoff
        arrow_dy = uy * arrow_backoff
        plt.arrow(
            arrow_start_x,
            arrow_start_y,
            arrow_dx,
            arrow_dy,
            head_width=0.01,
            head_length=0.02,
            fc=dst_color,
            ec=dst_color,
            length_includes_head=True,
            zorder=3,
        )


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
        _node_color_list = [node_colors.get(node, "lightblue") for node in G.nodes()]

        # Store colors as node attributes for reference
        nx.set_node_attributes(G, {node: {"color": color} for node, color in node_colors.items()})
    else:
        # Default color if no attribute is specified
        _node_color_list = ["lightblue" for _ in G.nodes()]

    # Identify source and sink nodes
    sources = [node for node in G.nodes() if G.in_degree(node) == 0]
    sinks = [node for node in G.nodes() if G.out_degree(node) == 0]

    # Create node size list (larger for sources and sinks)
    base_size = 500 * node_size_factor
    large_size = 700 * node_size_factor

    # Node shapes (circle for regular nodes, square for sources, diamond for sinks)
    node_shapes = {
        "regular": "o",
        "source": "d",  # diamond
        "sink": "$\heartsuit$",  # heart
    }

    # Draw nodes by type (sources, sinks, regular) to use different shapes
    regular_nodes = [node for node in G.nodes() if node not in sources and node not in sinks]

    # For each type of node (regular, source, sink), draw nodes individually

    # Get the order of nodes in the original list for color mapping
    _node_list = list(G.nodes())

    # Draw regular nodes (circles)
    for node in regular_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[node],
            node_size=int(base_size),
            node_color="#eeeeee",
            node_shape=node_shapes["regular"],
        )

    # Draw source nodes (squares)
    for node in sources:
        # node_idx = node_list.index(node)
        # node_color = node_color_list[node_idx]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[node],
            node_size=int(large_size),
            node_color="white",  # Fill color
            edgecolors="black",  # Border color
            linewidths=1.0,  # Optional: thicker border
            node_shape=node_shapes["source"],
        )

    # Draw sink nodes (diamonds)
    for node in sinks:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[node],
            node_size=int(large_size),
            node_color="red",
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

    assert color_map is not None
    node_color_names = {
        node: color_map.get(attr_values[node], "lightblue") for node in G.nodes() if node in attr_values
    }
    draw_segmented_edges(G, pos, node_color_names)

    plt.axis("off")
    plt.tight_layout()
    plt.show()
