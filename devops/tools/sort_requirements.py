import os
import sys


def find_repo_root(start_path=None):
    """Find the repository root by looking for common repo indicators."""
    if start_path is None:
        start_path = os.getcwd()

    current_path = os.path.abspath(start_path)

    # Look for common repository indicators
    repo_indicators = [".git", ".hg", ".svn"]

    while current_path != os.path.dirname(current_path):  # Stop at filesystem root
        for indicator in repo_indicators:
            if os.path.exists(os.path.join(current_path, indicator)):
                return current_path
        current_path = os.path.dirname(current_path)

    # If we reach here, we couldn't find a repo root
    return None


def sort_requirements(repo_root):
    """Sort the requirements.txt file and rewrite it."""
    requirements_path = os.path.join(repo_root, "requirements.txt")

    # Check if the file exists
    if not os.path.isfile(requirements_path):
        print(f"Error: {requirements_path} not found.")
        sys.exit(1)

    # Open the file for reading
    with open(requirements_path, "r") as file:
        # Read all lines and remove trailing whitespace
        lines = [line.strip() for line in file.readlines() if line.strip()]

    # Sort the lines alphabetically (case-insensitive)
    sorted_lines = sorted(lines, key=str.lower)

    # Write the sorted lines back to the file
    with open(requirements_path, "w") as file:
        for line in sorted_lines:
            file.write(line + "\n")

    print(f"Requirements file at {requirements_path} has been sorted successfully.")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Find the repository root
    repo_root = find_repo_root(script_dir)

    if repo_root:
        print(f"Repository root found at: {repo_root}")
        sort_requirements(repo_root)
    else:
        print("Error: Could not find repository root.")
        sys.exit(1)


if __name__ == "__main__":
    main()
