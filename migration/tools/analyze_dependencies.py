#!/usr/bin/env uv run
"""Analyze and visualize dependencies between modules in the Metta codebase."""

import ast
import json
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict, deque
import argparse

class DependencyAnalyzer:
    def __init__(self):
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.module_to_file: Dict[str, Path] = {}
        self.file_to_module: Dict[Path, str] = {}
        self.external_deps: Set[str] = set()
        self.internal_packages = {'metta', 'agent', 'common', 'mettagrid', 'mettascope', 'app_backend'}
        
    def analyze_file(self, file_path: Path) -> None:
        """Analyze dependencies in a single Python file."""
        try:
            with open(file_path) as f:
                tree = ast.parse(f.read())
                
            module = self._path_to_module(file_path)
            self.module_to_file[module] = file_path
            self.file_to_module[file_path] = module
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self._process_import(module, node.module)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self._process_import(module, alias.name)
                        
        except (SyntaxError, FileNotFoundError):
            pass
    
    def _process_import(self, from_module: str, imported: str) -> None:
        """Process a single import statement."""
        # Check if it's an internal import
        top_level = imported.split('.')[0]
        
        if top_level in self.internal_packages:
            self.dependencies[from_module].add(imported)
            self.reverse_dependencies[imported].add(from_module)
        else:
            self.external_deps.add(top_level)
    
    def _path_to_module(self, path: Path) -> str:
        """Convert file path to module notation."""
        # Handle different directory structures
        parts = path.parts
        
        # Find where the package starts
        for i, part in enumerate(parts):
            if part in self.internal_packages:
                # Build module path from this point
                module_parts = list(parts[i:])
                if module_parts[-1].endswith('.py'):
                    module_parts[-1] = module_parts[-1][:-3]
                    
                # Handle src directory
                if 'src' in module_parts:
                    src_idx = module_parts.index('src')
                    if src_idx + 1 < len(module_parts):
                        module_parts = module_parts[:src_idx] + module_parts[src_idx+2:]
                        
                return '.'.join(module_parts)
        
        # Fallback: remove .py and use path as is
        return str(path).replace('/', '.').replace('.py', '')
    
    def analyze_directory(self, directory: Path) -> None:
        """Analyze all Python files in a directory."""
        python_files = [
            f for f in directory.rglob("*.py")
            if '.venv' not in str(f) 
            and '__pycache__' not in str(f)
            and '.git/' not in str(f)
            and 'migration/' not in str(f)
            and 'wandb/' not in str(f)
        ]
        
        print(f"Analyzing {len(python_files)} Python files...")
        
        for py_file in python_files:
            self.analyze_file(py_file)
    
    def find_strongly_connected_components(self) -> List[Set[str]]:
        """Find strongly connected components (circular dependencies)."""
        # Tarjan's algorithm for SCC
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = {}
        sccs = []
        
        def strongconnect(node):
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            on_stack[node] = True
            stack.append(node)
            
            if node in self.dependencies:
                for successor in self.dependencies[node]:
                    if successor not in index:
                        strongconnect(successor)
                        lowlinks[node] = min(lowlinks[node], lowlinks[successor])
                    elif on_stack.get(successor, False):
                        lowlinks[node] = min(lowlinks[node], index[successor])
            
            if lowlinks[node] == index[node]:
                component = set()
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.add(w)
                    if w == node:
                        break
                if len(component) > 1:  # Only keep non-trivial SCCs
                    sccs.append(component)
        
        for node in self.dependencies:
            if node not in index:
                strongconnect(node)
        
        return sccs
    
    def find_dependency_layers(self) -> List[Set[str]]:
        """Organize modules into dependency layers (topological sort)."""
        # Calculate in-degrees
        in_degree = defaultdict(int)
        all_modules = set(self.dependencies.keys()) | set(self.reverse_dependencies.keys())
        
        for module in all_modules:
            for dep in self.dependencies.get(module, set()):
                in_degree[dep] += 1
        
        # Find modules with no dependencies
        queue = deque([m for m in all_modules if in_degree[m] == 0])
        layers = []
        
        while queue:
            current_layer = set()
            next_queue = deque()
            
            while queue:
                module = queue.popleft()
                current_layer.add(module)
                
                for dependent in self.reverse_dependencies.get(module, set()):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_queue.append(dependent)
            
            if current_layer:
                layers.append(current_layer)
            queue = next_queue
        
        return layers
    
    def get_module_metrics(self, module: str) -> Dict:
        """Get metrics for a specific module."""
        return {
            'dependencies': len(self.dependencies.get(module, set())),
            'dependents': len(self.reverse_dependencies.get(module, set())),
            'total_connections': (
                len(self.dependencies.get(module, set())) + 
                len(self.reverse_dependencies.get(module, set()))
            )
        }
    
    def find_high_risk_modules(self, threshold: int = 10) -> List[Tuple[str, Dict]]:
        """Find modules with high dependency counts that are risky to migrate."""
        high_risk = []
        
        for module in set(self.dependencies.keys()) | set(self.reverse_dependencies.keys()):
            metrics = self.get_module_metrics(module)
            if metrics['total_connections'] >= threshold:
                high_risk.append((module, metrics))
        
        return sorted(high_risk, key=lambda x: x[1]['total_connections'], reverse=True)
    
    def generate_migration_order(self) -> List[str]:
        """Generate suggested migration order based on dependencies."""
        layers = self.find_dependency_layers()
        order = []
        
        for layer in layers:
            # Within each layer, sort by number of dependents (migrate less depended-on first)
            sorted_layer = sorted(
                layer, 
                key=lambda m: len(self.reverse_dependencies.get(m, set()))
            )
            order.extend(sorted_layer)
        
        return order
    
    def report(self) -> None:
        """Generate analysis report."""
        print("\n" + "="*60)
        print("DEPENDENCY ANALYSIS REPORT")
        print("="*60)
        
        # Basic statistics
        total_modules = len(set(self.dependencies.keys()) | set(self.reverse_dependencies.keys()))
        total_deps = sum(len(deps) for deps in self.dependencies.values())
        
        print(f"\nTotal modules analyzed: {total_modules}")
        print(f"Total internal dependencies: {total_deps}")
        print(f"External packages used: {len(self.external_deps)}")
        
        # Package-level statistics
        package_stats = defaultdict(lambda: {'modules': 0, 'deps_out': 0, 'deps_in': 0})
        
        for module in set(self.dependencies.keys()) | set(self.reverse_dependencies.keys()):
            package = module.split('.')[0]
            if package in self.internal_packages:
                package_stats[package]['modules'] += 1
                package_stats[package]['deps_out'] += len(self.dependencies.get(module, set()))
                package_stats[package]['deps_in'] += len(self.reverse_dependencies.get(module, set()))
        
        print("\nPackage-level statistics:")
        for package, stats in sorted(package_stats.items()):
            print(f"  {package}:")
            print(f"    Modules: {stats['modules']}")
            print(f"    Outgoing deps: {stats['deps_out']}")
            print(f"    Incoming deps: {stats['deps_in']}")
        
        # Find circular dependencies
        sccs = self.find_strongly_connected_components()
        if sccs:
            print(f"\n⚠ Found {len(sccs)} circular dependency groups:")
            for i, scc in enumerate(sccs[:5], 1):
                print(f"  Group {i}: {len(scc)} modules")
                for module in list(scc)[:3]:
                    print(f"    - {module}")
                if len(scc) > 3:
                    print(f"    ... and {len(scc) - 3} more")
        
        # High-risk modules
        high_risk = self.find_high_risk_modules()
        if high_risk:
            print(f"\n⚠ High-risk modules (>10 connections):")
            for module, metrics in high_risk[:10]:
                print(f"  {module}: {metrics['total_connections']} connections")
                print(f"    ({metrics['dependencies']} deps, {metrics['dependents']} dependents)")
        
        # External dependencies
        print(f"\nTop 10 external dependencies:")
        for dep in sorted(self.external_deps)[:10]:
            print(f"  - {dep}")
    
    def save_report(self, output_path: Path) -> None:
        """Save detailed analysis to JSON."""
        # Prepare data for JSON serialization
        report = {
            'statistics': {
                'total_modules': len(set(self.dependencies.keys()) | set(self.reverse_dependencies.keys())),
                'total_dependencies': sum(len(deps) for deps in self.dependencies.values()),
                'external_packages': sorted(list(self.external_deps))
            },
            'dependencies': {k: sorted(list(v)) for k, v in self.dependencies.items()},
            'reverse_dependencies': {k: sorted(list(v)) for k, v in self.reverse_dependencies.items()},
            'circular_dependencies': [sorted(list(scc)) for scc in self.find_strongly_connected_components()],
            'high_risk_modules': [
                {'module': m, 'metrics': metrics} 
                for m, metrics in self.find_high_risk_modules()
            ],
            'dependency_layers': [sorted(list(layer)) for layer in self.find_dependency_layers()],
            'suggested_migration_order': self.generate_migration_order()[:50]  # First 50 modules
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        print(f"\nDetailed report saved to: {output_path}")
    
    def generate_graphviz(self, output_path: Path, package_level: bool = True) -> None:
        """Generate Graphviz DOT file for visualization."""
        lines = ['digraph dependencies {', '  rankdir=LR;', '  node [shape=box];']
        
        if package_level:
            # Package-level view
            package_deps = defaultdict(set)
            
            for from_module, deps in self.dependencies.items():
                from_pkg = from_module.split('.')[0]
                for dep in deps:
                    to_pkg = dep.split('.')[0]
                    if from_pkg != to_pkg and from_pkg in self.internal_packages and to_pkg in self.internal_packages:
                        package_deps[from_pkg].add(to_pkg)
            
            for from_pkg, to_pkgs in package_deps.items():
                for to_pkg in to_pkgs:
                    lines.append(f'  "{from_pkg}" -> "{to_pkg}";')
        else:
            # Module-level view (limited to avoid clutter)
            high_risk = self.find_high_risk_modules(threshold=5)
            important_modules = {m for m, _ in high_risk[:20]}
            
            for from_module in important_modules:
                for to_module in self.dependencies.get(from_module, set()):
                    if to_module in important_modules:
                        lines.append(f'  "{from_module}" -> "{to_module}";')
        
        lines.append('}')
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('\n'.join(lines))
        print(f"Graphviz file saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze dependencies in the Metta codebase')
    parser.add_argument('--path', default='.', help='Path to analyze (default: current directory)')
    parser.add_argument('--output', default='migration/baselines/dependency-analysis.json',
                      help='Output path for JSON report')
    parser.add_argument('--graph', action='store_true', help='Generate Graphviz visualization')
    
    args = parser.parse_args()
    
    analyzer = DependencyAnalyzer()
    analyzer.analyze_directory(Path(args.path))
    analyzer.report()
    analyzer.save_report(Path(args.output))
    
    if args.graph:
        # Generate package-level graph
        analyzer.generate_graphviz(
            Path('migration/baselines/dependencies-packages.dot'),
            package_level=True
        )
        print("\nTo visualize: dot -Tpng migration/baselines/dependencies-packages.dot -o dependencies.png")

if __name__ == "__main__":
    main()