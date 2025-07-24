"""HTML export utilities for notebooks."""

import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional


def export_to_html(notebook_name: str, output_dir: Optional[str] = None) -> Path:
    """Export a notebook to HTML format.
    
    Args:
        notebook_name: Name of the notebook file (e.g., "my_notebook.ipynb")
        output_dir: Directory to save HTML. If None, uses ../log/
        
    Returns:
        Path to the exported HTML file
        
    Raises:
        RuntimeError: If export fails
    """
    # Generate output filename with timestamp
    base_name = notebook_name.replace('.ipynb', '')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_name = f"{base_name}_export_{timestamp}.html"
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path.cwd().parent / "log"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_name
    
    # Export using nbconvert
    try:
        result = subprocess.run([
            "jupyter", "nbconvert", 
            "--to", "html",
            "--no-input",  # Hide code cells
            "--output", str(output_path),
            notebook_name
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Exported to: {output_path}")
            print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
            return output_path
        else:
            raise RuntimeError(f"Export failed: {result.stderr}")
            
    except FileNotFoundError:
        raise RuntimeError("jupyter nbconvert not found. Install with: pip install nbconvert")
    except Exception as e:
        raise RuntimeError(f"Export error: {e}")