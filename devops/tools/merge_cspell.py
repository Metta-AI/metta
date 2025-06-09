#!/usr/bin/env -S uv run
import json
import os
import re
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


def process_workspace_file(file_path):
    """
    Extract words from repository code.workspace file and clear the list.

    vscode json rules are lax so it's easier to just use regex.

    """

    assert "code-workspace" in file_path

    try:
        # Read the file as text
        with open(file_path, "r") as file:
            content = file.read()

        # Improved regex to find cSpell.words array
        pattern = r'"cSpell\.words"\s*:\s*\[([\s\S]*?)\]'
        match = re.search(pattern, content)

        if not match:
            print("Could not find cSpell.words section in workspace file.")
            return [], content

        words_section = match.group(1)

        # Extract words with a more forgiving pattern that handles different formats
        words = []
        for word_match in re.finditer(r'"([^"]+)"', words_section):
            words.append(word_match.group(1))

        print(f"Found {len(words)} words in workspace file.")

        # Replace the words array with an empty array
        new_content = re.sub(pattern, '"cSpell.words": []', content)

        return words, new_content

    except Exception as e:
        print(f"Error processing workspace file: {e}")
        return [], None


def merge_spelling_words(workspace_file_path, cspell_file_path):
    """Merge spelling words from workspace file into cspell json file and resort."""
    # Check if both files exist
    if not os.path.isfile(workspace_file_path):
        print(f"Error: Workspace file {workspace_file_path} not found.")
        return False

    if not os.path.isfile(cspell_file_path):
        print(f"Error: CSpell file {cspell_file_path} not found.")
        return False

    print(f"Processing workspace file: {workspace_file_path}")
    print(f"Processing cspell file: {cspell_file_path}")

    # Extract words from workspace file and get modified content
    workspace_words, new_workspace_content = process_workspace_file(workspace_file_path)

    if not workspace_words:
        print("Warning: No words found in workspace file.")

    if new_workspace_content is None:
        print("Error: Failed to process workspace file.")
        return False

    # Read the cspell file
    try:
        with open(cspell_file_path, "r") as file:
            cspell_data = json.load(file)

        # Extract existing words
        cspell_words = cspell_data.get("words", [])
        print(f"Found {len(cspell_words)} words in cspell file.")
    except Exception as e:
        print(f"Error reading cspell file: {e}")
        return False

    # Merge the word lists
    merged_words = list(set(cspell_words + workspace_words))

    # Sort alphabetically (case-insensitive)
    merged_words.sort(key=str.lower)

    # Update the cspell data
    cspell_data["words"] = merged_words

    # Write the updated files
    try:
        # Update cspell file
        with open(cspell_file_path, "w") as file:
            json.dump(cspell_data, file, indent=2)
            file.write("\n")

        print(f"Updated {cspell_file_path} with {len(merged_words)} sorted words.")

        # Update workspace file only if we found words to transfer
        if workspace_words:
            with open(workspace_file_path, "w") as file:
                file.write(new_workspace_content)
            print(f"Cleared words from {workspace_file_path}.")

        return True
    except Exception as e:
        print(f"Error writing files: {e}")
        return False


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Find the repository root
    repo_root = find_repo_root(script_dir)

    if repo_root:
        print(f"Repository root found at: {repo_root}")

        # Define paths to files
        workspace_file = os.path.join(repo_root, "metta.code-workspace")
        cspell_file = os.path.join(repo_root, ".cspell.json")

        # Merge and sort the spelling words
        if merge_spelling_words(workspace_file, cspell_file):
            print("Successfully merged and sorted spelling words.")
        else:
            print("Failed to merge spelling words.")
            sys.exit(1)
    else:
        print("Error: Could not find repository root.")
        sys.exit(1)


if __name__ == "__main__":
    main()
