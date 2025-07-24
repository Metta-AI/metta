"""Tests for notebook generation - focusing on outcomes."""

import pytest
import json
import os
import tempfile
from experiments.notebooks.generation import generate_notebook, generate_notebook_from_template


class TestNotebookGeneration:
    """Test notebook generation produces expected outcomes."""
    
    def test_generates_valid_jupyter_notebook(self):
        """Test that generated file is a valid Jupyter notebook."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = generate_notebook(
                name="test_notebook",
                output_dir=tmpdir
            )
            
            # File should exist
            assert os.path.exists(filepath)
            assert filepath.endswith(".ipynb")
            
            # Should be valid JSON
            with open(filepath) as f:
                notebook = json.load(f)
            
            # Should have notebook structure
            assert "cells" in notebook
            assert "metadata" in notebook
            assert "nbformat" in notebook
            assert notebook["nbformat"] == 4
    
    def test_notebook_contains_requested_sections(self):
        """Test that notebook contains only requested sections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate with specific sections
            filepath = generate_notebook(
                name="selective_notebook",
                sections=["setup", "launch", "monitor"],
                output_dir=tmpdir
            )
            
            with open(filepath) as f:
                notebook = json.load(f)
            
            # Extract section headers
            section_headers = []
            for cell in notebook["cells"]:
                if cell["cell_type"] == "markdown":
                    content = "".join(cell["source"])
                    if content.strip().startswith("## "):
                        section_headers.append(content.strip())
            
            # Should have requested sections
            assert "## Setup" in section_headers
            assert "## Launch Training" in section_headers
            assert "## Monitor Training" in section_headers
            
            # Should NOT have other sections
            assert "## Metrics Analysis" not in section_headers
            assert "## Visualizations" not in section_headers
            assert "## Experiment Log" not in section_headers
    
    def test_research_notebook_starts_empty(self):
        """Test that research notebooks start with empty state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = generate_notebook(
                name="research_notebook",
                description="For exploratory research",
                output_dir=tmpdir
            )
            
            with open(filepath) as f:
                notebook = json.load(f)
            
            # Find state management cell
            state_cell_content = None
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code" and "wandb_run_names = " in "".join(cell["source"]):
                    state_cell_content = "".join(cell["source"])
                    break
            
            assert state_cell_content is not None
            # Should initialize with empty lists
            assert "wandb_run_names = []" in state_cell_content
            assert "skypilot_job_ids = []" in state_cell_content
    
    def test_experiment_notebook_has_prefilled_data(self):
        """Test that experiment notebooks have pre-filled run data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wandb_names = ["user.exp.run1", "user.exp.run2", "user.exp.run3"]
            sky_ids = ["sky-123", "sky-456", "sky-789"]
            
            filepath = generate_notebook(
                name="experiment_analysis",
                wandb_run_names=wandb_names,
                skypilot_job_ids=sky_ids,
                additional_metadata={"experiment_type": "ablation"},
                output_dir=tmpdir
            )
            
            with open(filepath) as f:
                notebook = json.load(f)
            
            # Find state cell
            state_content = None
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code" and "wandb_run_names = " in "".join(cell["source"]):
                    state_content = "".join(cell["source"])
                    break
            
            assert state_content is not None
            # Should have pre-filled data
            assert "wandb_run_names = ['user.exp.run1', 'user.exp.run2', 'user.exp.run3']" in state_content
            assert "skypilot_job_ids = ['sky-123', 'sky-456', 'sky-789']" in state_content
            assert '"experiment_type": "ablation"' in state_content
    
    def test_notebook_includes_working_imports(self):
        """Test that generated notebooks have correct imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = generate_notebook(
                name="import_test",
                sections=["setup", "metrics"],
                output_dir=tmpdir
            )
            
            with open(filepath) as f:
                notebook = json.load(f)
            
            # Find setup cell
            setup_cell = None
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code" and "import" in "".join(cell["source"]):
                    setup_cell = "".join(cell["source"])
                    break
            
            assert setup_cell is not None
            # Should import from correct locations
            assert "from experiments.wandb_utils import" in setup_cell
            assert "from experiments.notebooks.analysis import" in setup_cell
            assert "from experiments.notebooks.monitoring import" in setup_cell
            # Should NOT import from old structure
            assert "from experiments.notebooks.utils" not in setup_cell
    
    def test_template_wrapper_maintains_compatibility(self):
        """Test that the template wrapper function works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use the backwards-compatibility wrapper
            filepath = generate_notebook_from_template(
                experiment_name="compat_test",
                run_names=["run1", "run2"],
                sky_job_ids=["sky1", "sky2"],
                output_dir=tmpdir
            )
            
            assert os.path.exists(filepath)
            assert "compat_test" in filepath  # May include timestamp
            
            with open(filepath) as f:
                notebook = json.load(f)
            
            # Should have experiment name in title
            title_cell = notebook["cells"][0]
            assert "compat_test" in "".join(title_cell["source"])
            
            # Should have description somewhere in early cells
            found_description = False
            for i in range(min(5, len(notebook["cells"]))):
                cell_content = "".join(notebook["cells"][i]["source"])
                if "Analysis notebook for compat_test experiment" in cell_content:
                    found_description = True
                    break
            assert found_description
    
    def test_notebook_sections_are_self_contained(self):
        """Test that each section can work independently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate notebook with only monitor section
            filepath = generate_notebook(
                name="monitor_only",
                sections=["setup", "state", "monitor"],
                wandb_run_names=["test.run"],
                output_dir=tmpdir
            )
            
            with open(filepath) as f:
                notebook = json.load(f)
            
            # Should still be able to monitor without launch/metrics sections
            code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
            
            # Should have monitoring code that references wandb_run_names
            monitor_code = "".join(["".join(cell["source"]) for cell in code_cells])
            assert "monitor_training_statuses" in monitor_code
            assert "wandb_run_names" in monitor_code
    
    def test_invalid_sections_are_ignored(self):
        """Test that invalid section names are ignored gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Try to include non-existent sections
            filepath = generate_notebook(
                name="invalid_sections",
                sections=["setup", "invalid_section", "another_bad_one", "monitor"],
                output_dir=tmpdir
            )
            
            # Should still generate successfully
            assert os.path.exists(filepath)
            
            with open(filepath) as f:
                notebook = json.load(f)
            
            # Should only have valid sections
            section_headers = []
            for cell in notebook["cells"]:
                if cell["cell_type"] == "markdown" and "## " in "".join(cell["source"]):
                    section_headers.append("".join(cell["source"]).strip())
            
            assert "## Setup" in section_headers
            assert "## Monitor Training" in section_headers
            assert "## invalid_section" not in section_headers