#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess


def get_git_repo_files(repo_path: str) -> list[str]:
    """
    Lists all files in a git repository using 'git ls-files'.
    Returns a list of paths relative to the repository root.
    """
    try:
        # Ensure repo_path is an absolute path
        abs_repo_path = os.path.abspath(repo_path)
        if not os.path.isdir(os.path.join(abs_repo_path, ".git")):
            print(
                f"Error: '{abs_repo_path}' does not appear to be a git repository root."
            )
            sys.exit(1)

        result = subprocess.run(
            ["git", "-C", abs_repo_path, "ls-files"],
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",  # Git should output UTF-8 paths
        )
        return result.stdout.splitlines()
    except subprocess.CalledProcessError as e:
        print(f"Error listing files in git repository '{repo_path}':")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"Stderr: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print(
            "Error: 'git' command not found. Please ensure git is installed and in your PATH."
        )
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while listing git files: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate specified text files from a git repository into a single text file for LLM input."
    )
    parser.add_argument("repo_path", help="Path to the local git repository.")
    parser.add_argument(
        "--max-lines",
        type=int,
        default=8000,
        help="Maximum number of lines for a file to be included (default: 8000)",
    )
    args = parser.parse_args()

    repo_path = os.path.abspath(args.repo_path)

    if not os.path.isdir(repo_path):
        print(f"Error: Repository path '{repo_path}' not found or is not a directory.")
        sys.exit(1)

    # Determine output file path
    try:
        # Directory where the script itself is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # __file__ is not defined (e.g., interactive interpreter)
        # Fallback to current working directory for script_dir.
        script_dir = os.getcwd()
        print(
            f"Warning: Could not determine script directory, using CWD '{script_dir}' for output."
        )

    # Name the output file after the target repository directory
    target_dir_name = os.path.basename(repo_path)
    output_filename = os.path.join(script_dir, f"{target_dir_name}_llm_context.txt")

    print(f"Processing repository: {repo_path}")
    print(f"Output will be saved to: {output_filename}")

    relative_file_paths = get_git_repo_files(repo_path)
    skipped_files = []
    included_files_count = 0

    with open(output_filename, "w", encoding="utf-8") as outfile:
        outfile.write(
            f"--- CONTEXT FOR REPOSITORY: {os.path.basename(repo_path)} ---\n"
        )
        outfile.write(f"--- BASE PATH: {repo_path} ---\n\n")

        for rel_path in relative_file_paths:
            full_file_path = os.path.join(repo_path, rel_path)

            try:
                # 1. Try to read as UTF-8. If this fails, it's likely binary or an incompatible encoding.
                with open(
                    full_file_path, "r", encoding="utf-8", errors="strict"
                ) as infile:
                    file_content_str = infile.read()

                # 2. Check for NULL bytes in the decoded string.
                #    Presence of NULL bytes is a strong indicator of a binary file.
                if "\x00" in file_content_str:
                    skipped_files.append(
                        (rel_path, "Contains NULL bytes (likely binary)")
                    )
                    continue

                # 3. Filter by line count
                #    Count lines based on the read string content.
                #    Using splitlines() is generally robust.
                lines = file_content_str.splitlines()
                num_lines = len(lines)

                #                if num_lines == 0: # Skip empty files
                #                    skipped_files.append((rel_path, "Empty file"))
                #                    continue

                if num_lines > args.max_lines:
                    skipped_files.append(
                        (
                            rel_path,
                            f"Exceeds max lines ({num_lines} > {args.max_lines})",
                        )
                    )
                    continue

                # If all checks pass, write to output
                outfile.write(f"--- File: {rel_path} ---\n")
                outfile.write(file_content_str)  # Write the already read content
                outfile.write("\n\n")  # Add a couple of newlines for separation
                included_files_count += 1

            except UnicodeDecodeError:
                skipped_files.append(
                    (
                        rel_path,
                        "Cannot decode as UTF-8 (likely binary or non-UTF-8 text)",
                    )
                )
                continue
            except FileNotFoundError:
                skipped_files.append(
                    (rel_path, "File not found (possibly removed after ls-files)")
                )
                continue
            except IsADirectoryError:
                skipped_files.append(
                    (
                        rel_path,
                        "Is a directory (git ls-files might list submodule dirs)",
                    )
                )
                continue
            except IOError as e:
                skipped_files.append((rel_path, f"IOError reading file: {e}"))
                continue
            except Exception as e:
                skipped_files.append(
                    (rel_path, f"Unexpected error processing file: {e}")
                )
                continue

    print("\n--- Processing Complete ---")
    print(
        f"Successfully concatenated {included_files_count} files into '{output_filename}'."
    )

    if skipped_files:
        print("\n--- Skipped Files ---")
        for file_path, reason in skipped_files:
            print(f"- {file_path}: {reason}")
    else:
        print("No files were skipped.")


if __name__ == "__main__":
    main()
