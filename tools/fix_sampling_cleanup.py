#!/usr/bin/env python3
"""
Script to clean up malformed sampling replacements and ensure all sampling syntax is removed.
"""

import re
import os
from pathlib import Path


def clean_and_fix_file(file_path: Path) -> tuple[int, int]:
    """Clean up malformed replacements and fix any remaining sampling syntax."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    fixes = 0
    cleanups = 0
    
    for line in lines:
        original_line = line
        
        # First, clean up any malformed nested comments
        # Pattern: "value  # Fixed value (was value  # Fixed value (was value  # Fixed value (was ${sampling:...})))"
        nested_pattern = r'(\d+(?:\.\d+)?)\s*#\s*Fixed value \(was \1\s*#\s*Fixed value.*?\)'
        if re.search(nested_pattern, line):
            # Extract the actual value and the original sampling expression
            match = re.search(r'(\d+(?:\.\d+)?)\s*#.*?\$\{sampling:([^}]+)\}', line)
            if match:
                value = match.group(1)
                sampling_expr = match.group(2)
                # Replace the entire mess with a clean version
                line = re.sub(r'(\d+(?:\.\d+)?)\s*#.*', f'{value}  # Fixed value (was ${{sampling:{sampling_expr}}})', line)
                cleanups += 1
        
        # Fix incorrect comments that say "(was ${X,Y,Z})" instead of "(was ${sampling:X,Y,Z})"
        bad_comment_pattern = r'#\s*Fixed value \(was \$\{(\d+(?:\.\d+)?),(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\}\)'
        match = re.search(bad_comment_pattern, line)
        if match:
            line = re.sub(bad_comment_pattern, f'# Fixed value (was ${{sampling:{match.group(1)},{match.group(2)},{match.group(3)}}})', line)
            cleanups += 1
        
        # Now look for any remaining ${sampling:...} patterns
        sampling_pattern = r'\$\{sampling:([^}]+)\}'
        matches = list(re.finditer(sampling_pattern, line))
        
        for match in matches:
            try:
                # Extract the parameters
                params = match.group(1).split(',')
                if len(params) == 3:
                    center = params[2].strip()
                    # Replace with center value and proper comment
                    line = line.replace(match.group(0), f"{center}  # Fixed value (was {match.group(0)})")
                    fixes += 1
            except Exception as e:
                print(f"Error processing {match.group(0)} in {file_path}: {e}")
        
        cleaned_lines.append(line)
    
    # Only write if we made changes
    if fixes > 0 or cleanups > 0:
        with open(file_path, 'w') as f:
            f.writelines(cleaned_lines)
    
    return fixes, cleanups


def main():
    """Main function to process all YAML files."""
    workspace_root = Path("/workspace")
    
    total_fixes = 0
    total_cleanups = 0
    
    # Process all YAML files
    yaml_files = []
    for yaml_file in workspace_root.rglob("*.yaml"):
        if ".git" not in str(yaml_file):
            yaml_files.append(yaml_file)
    
    print(f"Processing {len(yaml_files)} YAML files...")
    
    for yaml_file in yaml_files:
        try:
            fixes, cleanups = clean_and_fix_file(yaml_file)
            if fixes > 0 or cleanups > 0:
                print(f"Fixed {yaml_file.relative_to(workspace_root)}: {fixes} replacements, {cleanups} cleanups")
                total_fixes += fixes
                total_cleanups += cleanups
        except Exception as e:
            print(f"Error processing {yaml_file}: {e}")
    
    print(f"\nTotal: {total_fixes} replacements, {total_cleanups} cleanups")
    
    # Final check for any remaining sampling syntax
    print("\nChecking for any remaining sampling syntax...")
    remaining = []
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r') as f:
                content = f.read()
                if "${sampling:" in content:
                    remaining.append(str(yaml_file.relative_to(workspace_root)))
        except Exception:
            pass
    
    if remaining:
        print(f"Warning: Found {len(remaining)} files still containing sampling syntax:")
        for f in remaining[:10]:  # Show first 10
            print(f"  - {f}")
        if len(remaining) > 10:
            print(f"  ... and {len(remaining) - 10} more")
    else:
        print("✓ No remaining files with sampling syntax found.")


if __name__ == "__main__":
    main()