"""
Performance comparison of execution order algorithms for small neural network graphs.
"""

import time
from collections import deque
from typing import List


class GraphPerformanceTest:
    """Test different execution order algorithms."""

    def __init__(self):
        self.modules = {}
        self.connections = {}

    def add_module(self, name: str):
        """Add a module to the graph."""
        self.modules[name] = name
        self.connections[name] = []

    def connect(self, source: str, target: str):
        """Connect source to target."""
        self.connections[target].append(source)

    def create_typical_nn_graph(self):
        """Create a typical neural network graph structure."""
        # Typical RL agent: obs -> embed -> lstm -> [policy, value]
        modules = [
            "obs_norm",
            "embed",
            "encoder",
            "lstm",
            "policy",
            "value",
            "critic_head",
            "actor_head",
            "action_dist",
            "baseline",
        ]

        for module in modules:
            self.add_module(module)

        # Create connections
        connections = [
            ("obs_norm", "embed"),
            ("embed", "encoder"),
            ("encoder", "lstm"),
            ("lstm", "policy"),
            ("lstm", "value"),
            ("policy", "actor_head"),
            ("value", "critic_head"),
            ("actor_head", "action_dist"),
            ("critic_head", "baseline"),
        ]

        for source, target in connections:
            self.connect(source, target)

    def dfs_topological_sort(self) -> List[str]:
        """Current DFS-based approach."""
        visited = set()
        temp = set()
        order = []

        def visit(node):
            if node in temp:
                raise ValueError(f"Cycle detected at {node}")
            if node in visited:
                return

            temp.add(node)
            for source in self.connections[node]:
                visit(source)

            temp.remove(node)
            visited.add(node)
            order.append(node)

        for node in self.modules:
            if node not in visited:
                visit(node)

        return order

    def bfs_from_sources(self) -> List[str]:
        """BFS starting from source nodes (your suggested approach)."""
        # Count incoming edges (dependencies)
        in_degree = {node: len(self.connections[node]) for node in self.modules}

        # Start with nodes that have no dependencies
        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)

            # Find all nodes that depend on this node
            for dependent, sources in self.connections.items():
                if node in sources:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        return order

    def simple_iterative(self) -> List[str]:
        """Simple iterative approach for small graphs."""
        order = []
        remaining = set(self.modules.keys())

        while remaining:
            # Find nodes whose dependencies are already in order
            ready = []
            for node in remaining:
                dependencies_satisfied = all(dep in order for dep in self.connections[node])
                if dependencies_satisfied:
                    ready.append(node)

            if not ready:
                raise ValueError("Cycle detected in graph")

            # Add ready nodes to order (deterministic by sorting)
            ready.sort()
            for node in ready:
                order.append(node)
                remaining.remove(node)

        return order

    def benchmark_algorithms(self, iterations: int = 10000):
        """Benchmark different algorithms."""
        algorithms = {
            "DFS Topological": self.dfs_topological_sort,
            "BFS from Sources": self.bfs_from_sources,
            "Simple Iterative": self.simple_iterative,
        }

        results = {}

        for name, algorithm in algorithms.items():
            # Warmup
            for _ in range(100):
                algorithm()

            # Benchmark
            start_time = time.perf_counter()
            for _ in range(iterations):
                order = algorithm()
            end_time = time.perf_counter()

            avg_time_us = (end_time - start_time) / iterations * 1_000_000
            results[name] = {
                "avg_time_us": avg_time_us,
                "order_length": len(order),
                "first_node": order[0],
                "last_node": order[-1],
            }

        return results

    def verify_correctness(self):
        """Verify all algorithms produce valid execution orders."""
        algorithms = {
            "DFS": self.dfs_topological_sort,
            "BFS": self.bfs_from_sources,
            "Iterative": self.simple_iterative,
        }

        orders = {}
        for name, algorithm in algorithms.items():
            orders[name] = algorithm()

        # Check that all orders are valid
        for name, order in orders.items():
            print(f"{name}: {order}")

            # Verify topological property
            position = {node: i for i, node in enumerate(order)}
            valid = True

            for node in order:
                for dependency in self.connections[node]:
                    if position[dependency] >= position[node]:
                        print(f"  ❌ Invalid: {dependency} comes after {node}")
                        valid = False

            if valid:
                print("  ✅ Valid topological order")
            print()


def main():
    """Run performance comparison."""
    print("=== Neural Network Graph Execution Order Performance ===\n")

    # Create test graph
    graph = GraphPerformanceTest()
    graph.create_typical_nn_graph()

    print(f"Graph size: {len(graph.modules)} modules")
    print(f"Connections: {sum(len(sources) for sources in graph.connections.values())} edges\n")

    # Verify correctness
    print("=== Correctness Verification ===")
    graph.verify_correctness()

    # Benchmark performance
    print("=== Performance Benchmark (10,000 iterations) ===")
    results = graph.benchmark_algorithms(10000)

    for algorithm, metrics in results.items():
        print(f"{algorithm}:")
        print(f"  Average time: {metrics['avg_time_us']:.2f} μs")
        print(f"  Order length: {metrics['order_length']}")
        print()

    # Find fastest
    fastest = min(results.items(), key=lambda x: x[1]["avg_time_us"])
    print(f"🏆 Fastest: {fastest[0]} ({fastest[1]['avg_time_us']:.2f} μs)")

    # Show relative performance
    print("\n=== Relative Performance ===")
    baseline_time = results["DFS Topological"]["avg_time_us"]
    for algorithm, metrics in results.items():
        speedup = baseline_time / metrics["avg_time_us"]
        print(f"{algorithm}: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")


if __name__ == "__main__":
    main()
