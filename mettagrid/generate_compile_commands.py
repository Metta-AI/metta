#!/usr/bin/env python3
# Generate compile_commands.json from bazel for clang-tidy using bazel's aquery

import json
import subprocess
import os
import sys
import re
from pathlib import Path

def run_bazel_aquery():
    """Run bazel aquery to get compilation commands."""
    # Build first to ensure all actions are available
    print("Building targets...")
    build_cmd = ["bazel", "build", "--config=dbg", "//:mettagrid_c", "//:mettagrid_lib"]
    try:
        subprocess.run(build_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error building targets: {e}")
        return None
    
    # Get compilation actions using aquery
    print("Extracting compilation commands from Bazel...")
    aquery_cmd = [
        "bazel", "aquery",
        "--config=dbg",
        "--output=textproto",
        'mnemonic("CppCompile", deps(//:mettagrid_c) + deps(//:mettagrid_lib))'
    ]
    
    try:
        result = subprocess.run(aquery_cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running bazel aquery: {e}")
        print(f"stderr: {e.stderr}")
        return None

def parse_aquery_output(aquery_output):
    """Parse aquery textproto output to extract compilation commands."""
    commands = []
    
    # Split into action blocks
    action_blocks = re.split(r'\n(?=actions {)', aquery_output)
    
    for block in action_blocks:
        if 'mnemonic: "CppCompile"' not in block:
            continue
        
        # Extract arguments
        arguments_match = re.search(r'arguments: \[(.*?)\]', block, re.DOTALL)
        if not arguments_match:
            continue
        
        # Parse arguments - they're quoted strings separated by commas
        arguments_str = arguments_match.group(1)
        # Extract quoted strings
        arguments = re.findall(r'"([^"]*)"', arguments_str)
        
        # Find the source file in arguments
        source_file = None
        for arg in arguments:
            if arg.endswith(('.cpp', '.cc', '.cxx', '.c')) and not arg.startswith('-'):
                # Convert bazel path to real path
                if arg.startswith('src/'):
                    source_file = arg
                elif arg.startswith('bazel-out/'):
                    # Skip generated files
                    continue
                else:
                    # Try to find the file
                    possible_paths = [
                        arg,
                        f"src/metta/mettagrid/{arg}",
                        f"tests/{arg}",
                        f"benchmarks/{arg}"
                    ]
                    for path in possible_paths:
                        if os.path.exists(path):
                            source_file = path
                            break
                
                if source_file:
                    break
        
        if source_file and os.path.exists(source_file):
            # Clean up the command - remove bazel-specific paths and replace with actual paths
            clean_args = []
            for arg in arguments:
                # Skip bazel internal flags
                if arg.startswith('bazel-out/') or arg.startswith('external/'):
                    continue
                # Replace includes
                if arg.startswith('-I'):
                    include_path = arg[2:]
                    if 'bazel-out' in include_path or 'external/' in include_path:
                        # Try to map common external includes
                        if 'python' in include_path.lower():
                            clean_args.append('-I/usr/include/python3.11')
                        elif 'pybind11' in include_path.lower():
                            # Find pybind11 in site-packages
                            clean_args.append('-I/usr/local/include')
                    else:
                        clean_args.append(arg)
                else:
                    clean_args.append(arg)
            
            # Build the compile command
            command = {
                "directory": os.getcwd(),
                "file": os.path.abspath(source_file),
                "command": " ".join(clean_args) if clean_args else f"clang++ -std=c++20 -Isrc/metta/mettagrid -Isrc/metta -c {source_file}"
            }
            commands.append(command)
    
    return commands

def generate_fallback_commands():
    """Generate fallback compile commands for all C++ files."""
    print("Generating fallback compile commands...")
    
    commands = []
    workspace_root = Path.cwd()
    
    # Find all C++ source files
    cpp_patterns = ['**/*.cpp', '**/*.cc', '**/*.cxx', '**/*.c']
    hpp_patterns = ['**/*.hpp', '**/*.h', '**/*.hxx']
    
    cpp_files = []
    for pattern in cpp_patterns + hpp_patterns:
        cpp_files.extend(workspace_root.glob(f"src/{pattern}"))
        cpp_files.extend(workspace_root.glob(f"tests/{pattern}"))
        cpp_files.extend(workspace_root.glob(f"benchmarks/{pattern}"))
    
    # Filter out bazel directories
    cpp_files = [f for f in cpp_files if 'bazel-' not in str(f)]
    
    # Check if we're in a virtual environment or use the project's .venv
    venv_path = Path.cwd().parent / '.venv'
    if venv_path.exists():
        # Use the project's virtual environment python
        python_cmd = str(venv_path / 'bin' / 'python')
    else:
        python_cmd = "python3"
    
    # Get Python include path
    python_include = subprocess.run(
        [python_cmd, "-c", "import sysconfig; print(sysconfig.get_path('include'))"],
        capture_output=True, text=True
    ).stdout.strip()
    
    # Get pybind11 include path
    pybind11_include = subprocess.run(
        [python_cmd, "-c", "import pybind11; print(pybind11.get_include())"],
        capture_output=True, text=True
    ).stdout.strip()
    
    print(f"Using Python include: {python_include}")
    print(f"Using pybind11 include: {pybind11_include}")
    
    for cpp_file in cpp_files:
        command = {
            "directory": str(workspace_root),
            "file": str(cpp_file.absolute()),
            "command": (
                f"clang++ -std=c++20 "
                f"-Isrc/metta/mettagrid -Isrc/metta -Isrc "
                f"-I{python_include} "
                f"-I{pybind11_include} "
                f"-I/usr/local/include "
                f"-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION "
                f"-O3 -DNDEBUG "
                f"-c {cpp_file}"
            )
        }
        commands.append(command)
    
    return commands

def main():
    """Generate compile_commands.json for clang-tidy."""
    # Try to get commands from bazel aquery
    aquery_output = run_bazel_aquery()
    
    if aquery_output:
        commands = parse_aquery_output(aquery_output)
    else:
        commands = []
    
    # If we didn't get enough commands, use fallback
    if len(commands) < 5:
        print(f"Only found {len(commands)} commands from Bazel, using fallback...")
        commands = generate_fallback_commands()
    
    # Ensure build-debug directory exists
    build_dir = Path("build-debug")
    build_dir.mkdir(exist_ok=True)
    
    # Write compile_commands.json
    output_file = build_dir / "compile_commands.json"
    with open(output_file, "w") as f:
        json.dump(commands, f, indent=2)
    
    print(f"✅ Generated {output_file} with {len(commands)} entries")
    
    # Verify the file is valid JSON
    try:
        with open(output_file, "r") as f:
            json.load(f)
        print("✅ compile_commands.json is valid")
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())