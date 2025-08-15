#!/usr/bin/env uv run
"""Validate all Python imports in the repository."""

import ast
import sys
from pathlib import Path
from typing import Set, List, Tuple, Dict
from collections import defaultdict

class ImportValidator:
    def __init__(self):
        self.valid_imports: Set[str] = set()
        self.invalid_imports: List[Tuple[Path, str, str]] = []
        self.import_graph: Dict[str, Set[str]] = defaultdict(set)
        self.file_imports: Dict[Path, List[str]] = defaultdict(list)
        
    def validate_file(self, file_path: Path) -> bool:
        """Validate all imports in a Python file."""
        try:
            with open(file_path) as f:
                content = f.read()
                tree = ast.parse(content)
                
            # Extract module path for graph building
            module_path = self._path_to_module(file_path)
                
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.file_imports[file_path].append(node.module)
                    self.import_graph[module_path].add(node.module)
                    
                    # Check if it's a local import
                    if node.module.startswith(('metta.', 'agent.', 'common.', 'mettagrid.')):
                        try:
                            # Try to import it
                            __import__(node.module)
                            self.valid_imports.add(node.module)
                        except ImportError as e:
                            self.invalid_imports.append((file_path, node.module, str(e)))
                            
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.file_imports[file_path].append(alias.name)
                        self.import_graph[module_path].add(alias.name)
                        
                        if alias.name.startswith(('metta', 'agent', 'common', 'mettagrid')):
                            try:
                                __import__(alias.name)
                                self.valid_imports.add(alias.name)
                            except ImportError as e:
                                self.invalid_imports.append((file_path, alias.name, str(e)))
            return True
        except SyntaxError as e:
            print(f"SYNTAX ERROR in {file_path}: {e}")
            return False
        except Exception as e:
            print(f"ERROR parsing {file_path}: {e}")
            return False
    
    def _path_to_module(self, path: Path) -> str:
        """Convert file path to module path."""
        # Remove .py extension and convert to module notation
        module_path = str(path).replace('.py', '').replace('/', '.')
        
        # Handle src directory structure
        if '.src.metta.' in module_path:
            module_path = module_path.replace('.src.', '.')
        
        return module_path
    
    def validate_directory(self, directory: Path):
        """Recursively validate all Python files."""
        python_files = list(directory.rglob("*.py"))
        
        # Filter out virtual environments, cache, and .git
        python_files = [
            f for f in python_files 
            if '.venv' not in str(f) 
            and '__pycache__' not in str(f)
            and 'migration/' not in str(f)
            and '.git/' not in str(f)
        ]
        
        print(f"Validating {len(python_files)} Python files...")
        
        for py_file in python_files:
            self.validate_file(py_file)
    
    def analyze_import_patterns(self):
        """Analyze import patterns in the codebase."""
        import_counts = defaultdict(int)
        
        for imports in self.file_imports.values():
            for imp in imports:
                if imp.startswith(('metta', 'agent', 'common', 'mettagrid')):
                    import_counts[imp.split('.')[0]] += 1
        
        return dict(import_counts)
    
    def find_circular_dependencies(self) -> List[Tuple[str, str]]:
        """Find potential circular import dependencies."""
        cycles = []
        
        for module, imports in self.import_graph.items():
            for imported in imports:
                # Check if the imported module also imports this module
                if imported in self.import_graph and module in self.import_graph[imported]:
                    cycle = tuple(sorted([module, imported]))
                    if cycle not in cycles:
                        cycles.append(cycle)
        
        return cycles
    
    def report(self):
        """Generate validation report."""
        print("\n" + "="*60)
        print("IMPORT VALIDATION REPORT")
        print("="*60)
        
        print(f"\n✓ Valid imports: {len(self.valid_imports)}")
        print(f"✗ Invalid imports: {len(self.invalid_imports)}")
        
        if self.invalid_imports:
            print("\nInvalid imports (first 10):")
            for path, module, error in self.invalid_imports[:10]:
                try:
                    rel_path = path.relative_to(Path.cwd())
                except ValueError:
                    # If path is not relative to cwd, just use it as is
                    rel_path = path
                print(f"  {rel_path}: {module}")
                print(f"    Error: {error}")
        
        # Analyze import patterns
        patterns = self.analyze_import_patterns()
        print("\nImport patterns by top-level package:")
        for package, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
            print(f"  {package}: {count} imports")
        
        # Check for circular dependencies
        cycles = self.find_circular_dependencies()
        if cycles:
            print(f"\n⚠ Found {len(cycles)} potential circular dependencies:")
            for a, b in cycles[:5]:
                print(f"  {a} <-> {b}")
        
        print("\nTotal files analyzed:", len(self.file_imports))
        print("Total unique local imports:", len(self.valid_imports))
        
    def save_report(self, output_path: Path):
        """Save detailed report to JSON."""
        import json
        
        report = {
            "valid_imports": sorted(list(self.valid_imports)),
            "invalid_imports": [
                {
                    "file": str(path),
                    "module": module,
                    "error": error
                }
                for path, module, error in self.invalid_imports
            ],
            "import_patterns": self.analyze_import_patterns(),
            "circular_dependencies": [
                {"module1": a, "module2": b}
                for a, b in self.find_circular_dependencies()
            ],
            "statistics": {
                "total_files": len(self.file_imports),
                "total_valid_imports": len(self.valid_imports),
                "total_invalid_imports": len(self.invalid_imports),
            }
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        print(f"\nDetailed report saved to: {output_path}")

if __name__ == "__main__":
    validator = ImportValidator()
    
    # Validate the entire repository
    validator.validate_directory(Path("."))
    
    # Generate reports
    validator.report()
    validator.save_report(Path("migration/baselines/import-validation-report.json"))
    
    # Exit with error code if there are invalid imports
    sys.exit(1 if validator.invalid_imports else 0)