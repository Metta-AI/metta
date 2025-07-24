#!/usr/bin/env python3
"""CLI tool to create a new research notebook.

Usage:
    ./experiments/new_research_notebook.py my_experiment
    ./experiments/new_research_notebook.py rl_ablation --description "Testing different RL algorithms"
    ./experiments/new_research_notebook.py quick_test --sections setup,launch,monitor,metrics
"""

import argparse

from experiments.notebooks.generation import generate_notebook, AVAILABLE_SECTIONS, DEFAULT_SECTIONS


def main():
    """Command-line interface for generating research notebooks."""
    parser = argparse.ArgumentParser(
        description="Generate a research notebook with customizable sections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available sections:
{chr(10).join(f'  {k}: {v}' for k, v in AVAILABLE_SECTIONS.items())}

Examples:
  # Create with all sections (default)
  ./experiments/new_research_notebook.py my_experiment
  
  # Create with specific sections
  ./experiments/new_research_notebook.py quick_test --sections setup,launch,monitor,metrics
  
  # Create minimal notebook for debugging
  ./experiments/new_research_notebook.py debug --sections setup,state,monitor
        """
    )
    parser.add_argument("name", help="Name for the research notebook (use underscores for spaces)")
    parser.add_argument("-d", "--description", help="Description of research focus")
    parser.add_argument("-s", "--sections", help="Comma-separated list of sections to include")
    parser.add_argument("-o", "--output-dir", default="experiments/notebooks/research",
                       help="Output directory (default: experiments/notebooks/research)")
    
    args = parser.parse_args()
    
    # Parse sections if provided
    sections = None
    if args.sections:
        sections = [s.strip() for s in args.sections.split(",")]
        # Validate sections
        invalid = [s for s in sections if s not in AVAILABLE_SECTIONS]
        if invalid:
            print(f"Error: Invalid sections: {', '.join(invalid)}")
            print(f"Available sections: {', '.join(AVAILABLE_SECTIONS.keys())}")
            return 1
    
    # Generate notebook
    filepath = generate_notebook(
        name=args.name,
        description=args.description or "",
        sections=sections,
        output_dir=args.output_dir
    )
    
    print(f"\nTo open your notebook:")
    print(f"  jupyter notebook {filepath}")
    print(f"\nOr in VS Code:")
    print(f"  code {filepath}")
    
    return 0


if __name__ == "__main__":
    exit(main())