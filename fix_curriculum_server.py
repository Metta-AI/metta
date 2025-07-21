#!/usr/bin/env python3
"""Fix CurriculumServer calls to remove host parameter."""

import re
import glob

# Files to fix
files = [
    "tests/rl/curriculum/test_trainer_integration.py",
    "tests/rl/curriculum/test_server_client.py"
]

# Pattern to match
pattern = r'CurriculumServer\((.*?),\s*host="127\.0\.0\.1",\s*port=(.*?)\)'
replacement = r'CurriculumServer(\1, port=\2)'

for file_path in files:
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace all occurrences
    new_content = re.sub(pattern, replacement, content)
    
    # Write back
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"Fixed {file_path}")

print("Done!")