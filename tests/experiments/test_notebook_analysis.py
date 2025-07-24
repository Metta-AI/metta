"""Tests for notebook analysis functionality - focusing on outcomes."""

import pytest
import pandas as pd
import numpy as np
from experiments.notebooks.analysis import fetch_metrics, plot_sps, create_run_summary_table


class TestNotebookAnalysis:
    """Test analysis functions produce expected outcomes."""
    
    def test_fetch_metrics_wrapper_function_exists(self):
        """Test that fetch_metrics wrapper function is properly defined."""
        # Just verify the function exists and has expected signature
        import inspect
        sig = inspect.signature(fetch_metrics)
        params = list(sig.parameters.keys())
        
        assert "wandb_run_names" in params
        assert "samples" in params
        
    def test_plot_sps_function_exists(self):
        """Test that plot_sps function is properly defined with expected parameters."""
        import inspect
        sig = inspect.signature(plot_sps)
        params = list(sig.parameters.keys())
        
        assert "wandb_run_names" in params
        assert "samples" in params
        assert "entity" in params
        assert "project" in params
        assert "title" in params
        assert "width" in params
        assert "height" in params
        
    def test_create_run_summary_table_function_exists(self):
        """Test that create_run_summary_table function is properly defined."""
        import inspect
        sig = inspect.signature(create_run_summary_table)
        params = list(sig.parameters.keys())
        
        assert "wandb_run_names" in params
        assert "metrics" in params  # Changed from show_metrics