from typing import Dict

import networkx as nx


def analyze_graph(G: nx.DiGraph) -> None:
    """
    Print basic analysis of the graph structure.

    Args:
        G: NetworkX directed graph
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    print("\nGraph Analysis:")
    print(f"  Nodes: {n}")
    print(f"  Edges: {m}")
    print(f"  Density: {nx.density(G):.4f}")

    # Check if the graph is strongly connected
    if nx.is_strongly_connected(G):
        print("  The graph is strongly connected")
    else:
        strongly_connected_components = list(nx.strongly_connected_components(G))
        print(f"  The graph has {len(strongly_connected_components)} strongly connected components")

    # Check if the graph is weakly connected
    if nx.is_weakly_connected(G):
        print("  The graph is weakly connected")
    else:
        weakly_connected_components = list(nx.weakly_connected_components(G))
        print(f"  The graph has {len(weakly_connected_components)} weakly connected components")

    # Check for cycles
    try:
        cycles = list(nx.simple_cycles(G))
        if cycles:
            print(f"  The graph contains {len(cycles)} cycles")
        else:
            print("  The graph is acyclic (DAG)")
    except Exception:
        print("  Cycle detection skipped (may be too complex)")

    # Node degree statistics
    in_degrees = [G.in_degree(n) for n in G.nodes()]
    out_degrees = [G.out_degree(n) for n in G.nodes()]

    print(f"  Average in-degree: {sum(in_degrees) / n:.2f}")
    print(f"  Average out-degree: {sum(out_degrees) / n:.2f}")
    print(f"  Max in-degree: {max(in_degrees)}")
    print(f"  Max out-degree: {max(out_degrees)}")
    print()


def analyze_nodes(G: nx.DiGraph) -> Dict:
    """
    Analyze each node as a 'factory' that transforms input colors to output colors.

    Args:
        G: NetworkX directed graph with color attributes on nodes

    Returns:
        Dictionary of node factories
    """
    # Ensure we have color attributes
    if not nx.get_node_attributes(G, "color"):
        print("Error: Graph nodes don't have color attributes.")
        return {}

    # Get color and color name attributes
    _node_hex_colors = nx.get_node_attributes(G, "color")
    node_color_names = nx.get_node_attributes(G, "color_name")

    node_descriptions = {}

    for node in G.nodes():
        # Get input edges (predecessors)
        predecessors = list(G.predecessors(node))
        input_colors = [node_color_names[pred] for pred in predecessors]

        # Count input colors
        input_color_counts = {}
        for color in input_colors:
            input_color_counts[color] = input_color_counts.get(color, 0) + 1

        # Get output edges (successors)
        successors = list(G.successors(node))
        output_colors = [node_color_names[succ] for succ in successors]

        # Count output colors
        output_color_counts = {}
        for color in output_colors:
            output_color_counts[color] = output_color_counts.get(color, 0) + 1

        # Create node description
        desc = {
            "node": node,
            "node_color": node_color_names[node],
            "input_colors": input_color_counts,
            "output_colors": output_color_counts,
        }

        node_descriptions[node] = desc

    # Collect and print orb generators
    print("=== ORB GENERATORS ===")
    for _node, desc in node_descriptions.items():
        # Check if this is an orb generator (no inputs but has outputs)
        if not desc["input_colors"] and desc["output_colors"]:
            output_color_counts = desc["output_colors"]
            output_str = " + ".join([f"{count} {color}" for color, count in output_color_counts.items()])
            print(f"Orb Generator: {output_str}")

    # Collect and print factories
    print("\n=== FACTORIES ===")
    for _node, desc in node_descriptions.items():
        # Check if this is a factory (has both inputs and outputs)
        if desc["input_colors"] and desc["output_colors"]:
            input_color_counts = desc["input_colors"]
            output_color_counts = desc["output_colors"]

            input_str = " + ".join([f"{count} {color}" for color, count in input_color_counts.items()])
            output_str = " + ".join([f"{count} {color}" for color, count in output_color_counts.items()])

            print(f"Factory: ({input_str}) → ({output_str})")

    # Collect and print heart altars
    print("\n=== HEART ALTARS ===")
    for _node, desc in node_descriptions.items():
        # Check if this is a heart altar (has inputs but no outputs)
        if desc["input_colors"] and not desc["output_colors"]:
            input_color_counts = desc["input_colors"]
            input_str = " + ".join([f"{count} {color}" for color, count in input_color_counts.items()])

            print(f"Heart Altar: {input_str} → Heart")

    return node_descriptions
