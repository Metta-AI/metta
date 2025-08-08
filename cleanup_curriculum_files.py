#!/usr/bin/env python3
"""
Cleanup script for curriculum analysis files.

This script removes unnecessary files and consolidates the curriculum analysis
codebase to focus on the three core goals:
1. Use main branch curricula (learning progress, random, prioritized regression)
2. Compare against oracle baseline
3. Keep learning progress sweep code
"""

import os
import shutil


def cleanup_curriculum_files():
    """Remove unnecessary curriculum analysis files."""

    # Files to remove (these are replaced by consolidated_curriculum_analysis.py)
    files_to_remove = [
        "metta/examples/curriculum_analysis_advanced_demo.py",
        "metta/examples/enhanced_oracle_demo.py",
        "metta/test_curriculum_analysis_integration.py",
        "metta/advanced_curriculum_analysis.png",
        "metta/learning_progress_grid_search.png",
        "metta/oracle_comparison_demo.png",
        "metta/oracle_performance_demo.png",
        "metta/learning_curves_demo.png",
    ]

    # Directories to clean up (remove generated files)
    dirs_to_clean = [
        "metta/integration_test",
        "metta/outputs",
    ]

    print("Cleaning up curriculum analysis files...")

    # Remove files
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Removed: {file_path}")
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

    # Clean directories
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"Removed directory: {dir_path}")
            except Exception as e:
                print(f"Failed to remove directory {dir_path}: {e}")
        else:
            print(f"Directory not found: {dir_path}")

    print("\nCleanup completed!")
    print("\nRemaining files:")
    print("- metta/examples/consolidated_curriculum_analysis.py (main analysis script)")
    print("- metta/examples/learning_progress_grid_search_demo.py (sweep code - kept as requested)")
    print("- metta/mettagrid/src/metta/mettagrid/curriculum/learning_progress.py (main branch curriculum)")
    print("- metta/mettagrid/src/metta/mettagrid/curriculum/random.py (main branch curriculum)")
    print("- metta/mettagrid/src/metta/mettagrid/curriculum/prioritize_regressed.py (main branch curriculum)")
    print("- metta/metta/rl/enhanced_oracle.py (oracle baseline)")
    print("- metta/metta/rl/curriculum_analysis.py (analysis framework)")
    print("- metta/metta/eval/curriculum_analysis.py (evaluation framework)")


if __name__ == "__main__":
    cleanup_curriculum_files()
