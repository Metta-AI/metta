#!/usr/bin/env python3
"""Script to remove sampling resolver usage from configuration files."""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Regex pattern to match ${sampling:lower,upper,center}
SAMPLING_PATTERN = re.compile(r'\$\{sampling:([^,]+),([^,]+),([^}]+)\}')

def find_sampling_usages(root_dir: str = '.') -> Dict[str, List[Tuple[int, str, str]]]:
    """Find all uses of sampling resolver in YAML files.
    
    Returns:
        Dict mapping file paths to list of (line_number, line, replacement) tuples
    """
    results = {}
    
    for path in Path(root_dir).rglob('*.yaml'):
        if any(part.startswith('.') for part in path.parts):
            continue  # Skip hidden directories
            
        with open(path, 'r') as f:
            lines = f.readlines()
            
        file_results = []
        for i, line in enumerate(lines):
            matches = list(SAMPLING_PATTERN.finditer(line))
            if matches:
                # Extract the center value (third parameter)
                for match in matches:
                    lower, upper, center = match.groups()
                    # Create replacement with just the center value
                    replacement = line.replace(match.group(0), center.strip())
                    file_results.append((i + 1, line.rstrip(), replacement.rstrip()))
                    
        if file_results:
            results[str(path)] = file_results
            
    return results

def find_sampling_params(root_dir: str = '.') -> Dict[str, List[Tuple[int, str]]]:
    """Find all 'sampling: X' parameter declarations in YAML files.
    
    Returns:
        Dict mapping file paths to list of (line_number, line) tuples
    """
    results = {}
    
    for path in Path(root_dir).rglob('*.yaml'):
        if any(part.startswith('.') for part in path.parts):
            continue  # Skip hidden directories
            
        with open(path, 'r') as f:
            lines = f.readlines()
            
        file_results = []
        for i, line in enumerate(lines):
            if re.match(r'^\s*sampling:\s*[0-9]', line):
                file_results.append((i + 1, line.rstrip()))
                    
        if file_results:
            results[str(path)] = file_results
            
    return results

def main():
    print("=== Sampling Resolver Usage Report ===\n")
    
    # Find all sampling resolver usages
    sampling_usages = find_sampling_usages()
    
    if sampling_usages:
        print(f"Found {len(sampling_usages)} files with sampling resolver usage:\n")
        
        total_usages = 0
        for filepath, usages in sorted(sampling_usages.items()):
            print(f"\n{filepath}:")
            for line_num, original, replacement in usages:
                print(f"  Line {line_num}: {original}")
                print(f"    Replace with: {replacement}")
                total_usages += 1
                
        print(f"\nTotal sampling resolver usages: {total_usages}")
    else:
        print("No sampling resolver usage found!")
        
    # Find all sampling parameter declarations
    print("\n\n=== Sampling Parameter Declarations ===\n")
    
    sampling_params = find_sampling_params()
    
    if sampling_params:
        print(f"Found {len(sampling_params)} files with sampling parameter:\n")
        
        for filepath, params in sorted(sampling_params.items()):
            print(f"\n{filepath}:")
            for line_num, line in params:
                print(f"  Line {line_num}: {line}")
    else:
        print("No sampling parameter declarations found!")
        
    # Summary
    print("\n\n=== Summary ===")
    print(f"Files with sampling resolver usage: {len(sampling_usages)}")
    print(f"Files with sampling parameter: {len(sampling_params)}")
    print("\nTo remove sampling resolver:")
    print("1. Replace all ${sampling:X,Y,Z} with Z (the center value)")
    print("2. Remove all 'sampling: X' parameter declarations")
    print("3. Update code that references sampling parameter")

if __name__ == "__main__":
    main()