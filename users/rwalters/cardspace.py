import random
from typing import Any, Dict, Set, Tuple

import networkx as nx

from .graph_analysis import analyze_graph, analyze_nodes
from .graph_generation import GraphConfig, GraphType, generate_graph_from_config
from .graph_visualization import visualize_graph as visualize_graph_enhanced


def get_orb_colors(n: int) -> Dict[str, str]:
    """
    Get a dictionary of orb color names mapped to hex codes.

    Args:
        n: Number of colors to return (1-10)

    Returns:
        Dictionary mapping color names to hex codes
    """
    colors = {
        "Red": "#FF0000",
        "Orange": "#FF7F00",
        "Yellow": "#FFFF00",
        "Green": "#00FF00",
        "Blue": "#0000FF",
        "Indigo": "#4B0082",
        "Violet": "#8B00FF",
        "Pink": "#FF69B4",
        "Cyan": "#00FFFF",
        "Magenta": "#FF00FF",
    }

    if n <= 0 or n > len(colors):
        raise ValueError(f"Number of colors must be between 1 and {len(colors)}")

    # Get first n colors
    return {k: colors[k] for k in list(colors.keys())[:n]}


def set_node_attributes_from_dict(G: nx.DiGraph, attribute_dict: Dict[int, Any], attribute_name: str) -> None:
    """
    Set node attributes from a dictionary.

    Args:
        G: NetworkX directed graph
        attribute_dict: Dictionary mapping node IDs to attribute values
        attribute_name: Name of the attribute to set
    """
    nx.set_node_attributes(G, {node: value for node, value in attribute_dict.items()}, attribute_name)


def get_attribute_values(G: nx.DiGraph, attribute: str) -> Set[Any]:
    """
    Get the set of unique values for a node attribute.

    Args:
        G: NetworkX directed graph
        attribute: The node attribute to query

    Returns:
        Set of unique values for the attribute
    """
    if attribute not in nx.get_node_attributes(G, attribute):
        raise ValueError(f"Graph does not have the attribute '{attribute}' on its nodes")

    attr_values = nx.get_node_attributes(G, attribute)
    return set(attr_values.values())


def assign_random_node_colors(G: nx.DiGraph, num_colors: int) -> Dict[int, Tuple[str, str]]:
    """
    Assign random colors to nodes from a palette of rainbow colors.

    Args:
        G: NetworkX directed graph
        num_colors: Number of colors to use (1-10)

    Returns:
        Dictionary mapping node IDs to (color_name, hex_code) tuples
    """
    color_dict = get_orb_colors(num_colors)
    color_names = list(color_dict.keys())
    node_colors = {}

    # Store both the color name and hex code for each node
    for node in G.nodes():
        color_name = random.choice(color_names)
        hex_code = color_dict[color_name]
        node_colors[node] = (color_name, hex_code)

    # Store the colors as node attributes
    nx.set_node_attributes(G, {node: data[1] for node, data in node_colors.items()}, "color")
    nx.set_node_attributes(G, {node: data[0] for node, data in node_colors.items()}, "color_name")

    return node_colors


# Main function
def main():
    print("\n=== Orb Factory Problem Generator ===\n")

    # Menu for graph parameters
    print("Please enter the graph parameters (press Enter for defaults):")

    # Get node count
    default_nodes = 20
    nodes_input = input(f"Number of factories [{default_nodes}]: ")
    n_nodes = int(nodes_input) if nodes_input.strip() else default_nodes

    # Get edge count
    default_edges = n_nodes * 2
    edges_input = input(f"Number of edges [{default_edges}]: ")
    n_edges = int(edges_input) if edges_input.strip() else default_edges

    # Get number of colors
    default_colors = 5
    colors_input = input(f"Number of orb colors (1-10) [{default_colors}]: ")
    num_colors = int(colors_input) if colors_input.strip() else default_colors
    num_colors = max(1, min(10, num_colors))  # Ensure between 1 and 10

    # Get graph generation method
    print("\nGraph generation methods:")
    print("1. Random Edge Method")
    print("2. Erdős–Rényi Method")
    print("3. Preferential Attachment")

    default_method = 1
    method_input = input(f"Select method (1-3) [{default_method}]: ")
    method = int(method_input) if method_input.strip() else default_method

    # Select graph type based on method choice
    graph_type = None
    if method == 1:
        graph_type = GraphType.RANDOM
        title = "Random Edge Directed Graph"
    elif method == 2:
        graph_type = GraphType.ERDOS_RENYI
        title = "Erdős–Rényi Directed Graph"
    elif method == 3:
        graph_type = GraphType.PREFERENTIAL_ATTACHMENT
        title = "Preferential Attachment Directed Graph"
    else:
        graph_type = GraphType.RANDOM
        title = "Random Edge Directed Graph"

    # Get layout method
    print("\nLayout algorithms:")
    print("1. Spring layout (force-directed)")
    print("2. Circular layout")
    print("3. Random layout")
    print("4. Shell layout")
    print("5. Kamada-Kawai layout")

    default_layout = 1
    layout_input = input(f"Select layout (1-5) [{default_layout}]: ")
    layout_choice = int(layout_input) if layout_input.strip() else default_layout

    # Map layout choice to layout name
    layouts = ["spring", "circular", "random", "shell", "kamada_kawai"]
    layout = layouts[layout_choice - 1] if 1 <= layout_choice <= 5 else layouts[0]

    # Ask about sources and sinks
    default_sources = num_colors
    sources_input = input(f"\nNumber of source nodes (entry points) [{default_sources}]: ")
    n_sources = int(sources_input) if sources_input.strip() else default_sources

    default_sinks = 1
    sinks_input = input(f"Number of sink nodes (exit points) [{default_sinks}]: ")
    n_sinks = int(sinks_input) if sinks_input.strip() else default_sinks

    # Create GraphConfig

    if graph_type == GraphType.PREFERENTIAL_ATTACHMENT:
        # For preferential attachment, calculate appropriate m_per_node
        m_per_node = min(2, n_nodes - 1)  # Default to 2 if possible
        if n_edges > (n_nodes - m_per_node - 1) * m_per_node:
            # If user requested more edges, try to increase m_per_node
            potential_m = n_edges / (n_nodes - m_per_node - 1)
            m_per_node = min(int(potential_m), n_nodes - 1)
    else:
        m_per_node = 0  # ignored

    print(f"Using {m_per_node} edges per new node for preferential attachment.")

    config = GraphConfig(
        graph_type=graph_type,
        n_nodes=n_nodes,
        n_edges=n_edges,
        m_per_node=m_per_node,
        allow_self_loops=False,
        n_sources=n_sources,
        n_sinks=n_sinks,
    )

    # Generate the graph from the config
    print("\nGenerating directed graph...")
    graph = generate_graph_from_config(config)

    # Analyze the graph
    analyze_graph(graph)

    # Assign colors and visualize
    node_colors = assign_random_node_colors(graph, num_colors)

    # Analyze node factories
    analyze_nodes(graph)

    # Use the enhanced visualization from the imported library
    # Instead of the previous visualize_graph function
    node_labels = {node: str(node) for node in graph.nodes()}

    print("\nVisualizing the graph using enhanced visualization...")

    # Create a color_map from the node_colors dictionary
    # Convert from (color_name, hex_code) format to color_name -> hex_code format
    color_attr_values = nx.get_node_attributes(graph, "color_name")
    unique_color_values = set(color_attr_values.values())
    color_map = {}
    for color_name in unique_color_values:
        # Find a node with this color to get the hex code
        for _node, (name, hex_code) in node_colors.items():
            if name == color_name:
                color_map[color_name] = hex_code
                break

    # Use the enhanced visualization function from imported library
    visualize_graph_enhanced(
        graph,
        title=title,
        layout=layout,
        color_attribute="color_name",
        color_map=color_map,
        node_labels=node_labels,
        show_legend=True,
        node_size_factor=1.0,
    )

    print("\nGraph generation complete!")


if __name__ == "__main__":
    main()
