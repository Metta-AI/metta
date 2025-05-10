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
    Analyze each node as a 'factory', 'generator', or 'altar' based on graph structure.

    Args:
        G: NetworkX directed graph with color attributes on nodes

    Returns:
        Dictionary mapping node IDs to descriptions including type and input/output colors.
    """
    if not nx.get_node_attributes(G, "color"):
        print("Error: Graph nodes don't have color attributes.")
        return {}

    _node_hex_colors = nx.get_node_attributes(G, "color")
    node_color_names = nx.get_node_attributes(G, "color_name")

    node_descriptions = {}

    for node in G.nodes():
        predecessors = list(G.predecessors(node))
        successors = list(G.successors(node))

        input_colors = [node_color_names[pred] for pred in predecessors]
        output_colors = [node_color_names[succ] for succ in successors]

        input_color_counts = {c: input_colors.count(c) for c in set(input_colors)}
        output_color_counts = {c: output_colors.count(c) for c in set(output_colors)}

        # Determine node type
        if not input_color_counts and output_color_counts:
            node_type = "generator"
        elif input_color_counts and not output_color_counts:
            node_type = "altar"
        else:
            # Allow a more nuanced factory vs. generator distinction later
            node_type = "factory"

        node_descriptions[node] = {
            "node": node,
            "node_color": node_color_names[node],
            "input_colors": input_color_counts,
            "output_colors": output_color_counts,
            "node_type": node_type,
        }

    # Print report
    print("=== ORB GENERATORS ===")
    for node, desc in node_descriptions.items():
        if desc["node_type"] == "generator":
            output_str = " + ".join(f"{v} {k}" for k, v in desc["output_colors"].items())
            print(f"[{node}] Generator: {output_str}")

    print("\n=== FACTORIES ===")
    for node, desc in node_descriptions.items():
        if desc["node_type"] == "factory":
            input_str = " + ".join(f"{v} {k}" for k, v in desc["input_colors"].items())
            output_str = " + ".join(f"{v} {k}" for k, v in desc["output_colors"].items())
            print(f"[{node}] Orb Factory: ({input_str}) → ({output_str})")

    print("\n=== HEART ALTARS ===")
    for node, desc in node_descriptions.items():
        if desc["node_type"] == "altar":
            input_str = " + ".join(f"{v} {k}" for k, v in desc["input_colors"].items())
            print(f"[{node}] Heart Altar: {input_str} → Heart")

    return node_descriptions
