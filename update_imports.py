#!/usr/bin/env python3
import os
import re
import sys

def update_imports_in_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace imports
    updated_content = re.sub(r'from metta\.util', r'from common.src.metta.util', content)
    updated_content = re.sub(r'import metta\.util', r'import common.src.metta.util', updated_content)
    
    if content != updated_content:
        with open(file_path, 'w') as f:
            f.write(updated_content)
        print(f"Updated imports in {file_path}")

def main():
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                update_imports_in_file(file_path)

if __name__ == "__main__":
    main()