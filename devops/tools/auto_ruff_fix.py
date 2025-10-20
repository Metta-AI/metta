#!/usr/bin/env -S uv run
"""
Automatically fix Ruff errors using Claude to generate patches.

This script:
1. Runs Ruff to identify issues
2. For each issue, extracts relevant file content
3. Sends the error and file content to Claude API
4. Applies the suggested diff to fix the file
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    from dotenv import load_dotenv

    # Load environment variables from .env file if it exists
    load_dotenv()
except ImportError:
    # python-dotenv is not installed, continue without it
    pass

import anthropic


@dataclass
class RuffError:
    file_path: str
    line_number: int
    column: int
    error_code: str
    message: str
    context_lines: List[str]


class AutoRuffFix:
    def __init__(self, claude_api_key: str, model: str = "claude-3-7-sonnet-20250219", context_lines: int = 5):
        """Initialize the AutoRuffFix tool.

        Args:
            claude_api_key: Anthropic API key for Claude
            model: Claude model to use for generating fixes
            context_lines: Number of lines to include before and after the error line (default: 5)
        """
        self.client = anthropic.Anthropic(api_key=claude_api_key)
        self.model = model
        self.verbose = False
        self.context_lines = context_lines

    def set_verbose(self, verbose: bool):
        """Set verbosity level."""
        self.verbose = verbose

    def run_ruff(self, paths: List[str], config: Optional[str] = None) -> List[RuffError]:
        """Run Ruff and parse the output to extract errors.

        Args:
            paths: List of file paths or directories to check
            config: Optional path to Ruff config file

        Returns:
            List of RuffError objects
        """
        cmd = ["ruff", "check", "--output-format=json"]
        if config:
            cmd.extend(["--config", config])
        cmd.extend(paths)

        if self.verbose:
            print(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode == 0:
                print("No Ruff errors found!")
                return []

            try:
                errors_json = json.loads(result.stdout)
            except json.JSONDecodeError:
                print(f"Failed to parse Ruff output as JSON: {result.stdout}")
                return []

            ruff_errors = []
            for error in errors_json:
                file_path = error.get("filename", "")
                line = error.get("location", {}).get("row", 0)
                column = error.get("location", {}).get("column", 0)
                code = error.get("code", "")
                message = error.get("message", "")

                # Get context lines from the file using the configured context_lines
                context_lines = self._get_context_lines(file_path, line)

                ruff_error = RuffError(
                    file_path=file_path,
                    line_number=line,
                    column=column,
                    error_code=code,
                    message=message,
                    context_lines=context_lines,
                )
                ruff_errors.append(ruff_error)

            return ruff_errors

        except subprocess.CalledProcessError as e:
            print(f"Error running Ruff: {e}")
            return []

    def _get_context_lines(self, file_path: str, line_number: int, context_size: Optional[int] = None) -> List[str]:
        """Get context lines around the specified line in the file.

        Args:
            file_path: Path to the file
            line_number: The line number where the error was found (1-based)
            context_size: Number of lines to get before and after the error line
                         (if None, uses self.context_lines)

        Returns:
            List of context lines with line numbers
        """
        if context_size is None:
            context_size = self.context_lines

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            start = max(0, line_number - context_size - 1)
            end = min(len(lines), line_number + context_size)

            context_lines = []
            for i in range(start, end):
                line_content = lines[i].rstrip()
                line_num = i + 1
                marker = "|" if line_num == line_number else " "
                context_lines.append(f"{line_num:4d} {marker} {line_content}")

            return context_lines

        except Exception as e:
            print(f"Error getting context lines from {file_path}: {e}")
            return []

    def get_file_content(self, file_path: str) -> str:
        """Read and return the content of a file.

        Args:
            file_path: Path to the file

        Returns:
            File content as a string
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return ""

    def generate_fix(self, error: RuffError, file_content: str) -> Optional[str]:
        """Use Claude to generate a diff that fixes the given error.

        Args:
            error: The RuffError to fix
            file_content: The content of the file containing the error

        Returns:
            A diff string that fixes the error, or None if generation failed
        """
        context_str = "\n".join(error.context_lines)

        # Calculate line count of the file to check if we should include selective context or whole file
        line_count = len(file_content.splitlines())
        file_too_large = line_count > 300  # Set a reasonable threshold (about 15KB of text)

        # If we have a specific large context size but the file isn't huge, we might want to
        # include the entire file content as context instead of just the error surroundings
        include_file_snippet = self.context_lines >= 20 and not file_too_large

        # Prepare the prompt with appropriate context
        if include_file_snippet:
            context_section = f"""Here's the full file content:
```python
{file_content}
```"""
        else:
            context_section = f"""Context of the error:
```
{context_str}
```

Here's the full file content:
```python
{file_content}
```"""

        prompt = f"""I need to fix a Ruff linting error in my Python code. Please help me generate a diff that only
contains the changes needed to fix this specific issue.

File: {error.file_path}
Error: {error.error_code} at line {error.line_number}, column {error.column}
Message: {error.message}

{context_section}

Please generate a unified diff that contains ONLY the minimal changes needed to fix this specific {error.error_code}
error. The diff should use the standard unified diff format with @@ line markers.

Important rules:
1. The diff should change as few lines as possible - ideally just the problematic line
2. Make sure any changed Python code is syntactically valid
3. Do not use wildcard characters like * in the Python code
4. For line length issues (E501), prefer simple formatting fixes like line breaks
5. Include ONLY the diff and enclose it between <diff> and </diff> tags

Example of the format I need:
<diff>
@@ -21,7 +21,7 @@
         return HTMLResponse(content=html_content)
     except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Client HTML file not found") from None

 @app.websocket("/ws")
</diff>
"""

        if self.verbose:
            print(f"Sending request to Claude for {error.file_path}:{error.line_number} ({error.error_code})")
            if include_file_snippet:
                print(f"Including entire file content in prompt (lines: {line_count})")

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.0,
                system="You are an expert Python developer who specializes in fixing code style and linting issues. "
                "You provide precise, minimal diffs to fix specific issues.",
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract the diff from the response
            content = response.content[0].text
            diff_match = re.search(r"<diff>(.*?)</diff>", content, re.DOTALL)

            if diff_match:
                diff_content = diff_match.group(1).strip()

                # Validate the diff - make sure it doesn't have any wildcard characters
                # that would make invalid Python syntax
                if "*" in diff_content and re.search(r"[+].*\*.*\*", diff_content):
                    print(f"Invalid diff generated for {error.file_path}:{error.line_number} ({error.error_code})")
                    print("Diff contains wildcard characters that would create invalid Python")

                    # Try to fix the diff by replacing wildcard patterns
                    # This is a simple fix for patterns like "*, X, *" which are likely SVD unpacking
                    diff_content = re.sub(r"([+].*)\*, (.*?), \*(.*)", r"\1_, \2, _\3", diff_content)

                    if self.verbose:
                        print(f"Attempted to fix diff:\n{diff_content}")

                return diff_content
            else:
                print(f"Failed to extract diff for {error.file_path}:{error.line_number} ({error.error_code})")
                if self.verbose:
                    print(f"Claude response: {content}")
                return None

        except Exception as e:
            print(f"Error generating fix with Claude: {e}")
            return None

    def apply_diff(self, file_path: str, diff_content: str) -> bool:
        """Apply the diff using a robust string replacement approach.

        This bypasses the patch command entirely and just looks for the specific lines to replace.

        Args:
            file_path: Path to the file to be patched
            diff_content: The diff content to apply

        Returns:
            True if replacement was applied, False otherwise
        """
        try:
            # Write the diff to a temporary file (for reference)
            diff_file = f"{file_path}.patch"
            with open(diff_file, "w", encoding="utf-8") as f:
                f.write(diff_content)

            # Read the original file
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
                file_lines = file_content.split("\n")

            # Parse the diff to extract changes - handle line additions, removals, and replacements
            changes = []
            current_hunk = None

            for line in diff_content.split("\n"):
                if line.startswith("@@"):
                    # Parse the @@ -start,count +start,count @@ line to get position info
                    parts = line.split()
                    if len(parts) >= 2:
                        # Extract line numbers from the format "@@ -start,count +start,count @@"
                        minus_part = parts[1]  # e.g., "-5,6"
                        plus_part = parts[2]  # e.g., "+5,7"

                        # Extract the starting line numbers
                        try:
                            minus_start = int(minus_part.split(",")[0][1:])  # Remove the '-' and get the number
                            plus_start = int(plus_part.split(",")[0][1:])  # Remove the '+' and get the number

                            current_hunk = {
                                "minus_start": minus_start,
                                "plus_start": plus_start,
                                "minus_lines": [],
                                "plus_lines": [],
                            }
                            changes.append(current_hunk)
                        except (ValueError, IndexError):
                            print(f"Failed to parse hunk header: {line}")
                            current_hunk = None
                elif current_hunk is not None:
                    if line.startswith("-"):
                        current_hunk["minus_lines"].append(line[1:])
                    elif line.startswith("+"):
                        current_hunk["plus_lines"].append(line[1:])
                    elif line.startswith(" "):
                        # Context lines appear in both old and new versions
                        current_hunk["minus_lines"].append(line[1:])
                        current_hunk["plus_lines"].append(line[1:])

            if self.verbose:
                print(f"Parsed {len(changes)} change hunks from diff")

            # Apply the changes in reverse order (to maintain line numbers)
            changes.sort(key=lambda x: x["minus_start"], reverse=True)
            modified = False

            for hunk in changes:
                minus_start = hunk["minus_start"] - 1  # Convert to 0-based index
                minus_lines = hunk["minus_lines"]
                plus_lines = hunk["plus_lines"]

                # Special case: handle pure additions (no lines removed)
                if not minus_lines and plus_lines:
                    if self.verbose:
                        print(f"Adding {len(plus_lines)} new lines at line {minus_start + 1}")
                    file_lines[minus_start:minus_start] = plus_lines
                    modified = True
                    continue

                # Special case: handle pure deletions (no lines added)
                if minus_lines and not plus_lines:
                    if self.verbose:
                        print(f"Removing {len(minus_lines)} lines at line {minus_start + 1}")
                    del file_lines[minus_start : minus_start + len(minus_lines)]
                    modified = True
                    continue

                # Regular case: replacing lines
                # Check if the lines to be removed match what's in the file
                file_section = file_lines[minus_start : minus_start + len(minus_lines)]

                if file_section == minus_lines:
                    # Direct match, replace the lines
                    if self.verbose:
                        print(f"Replacing {len(minus_lines)} lines at line {minus_start + 1}")
                    file_lines[minus_start : minus_start + len(minus_lines)] = plus_lines
                    modified = True
                else:
                    # Try fuzzy matching if exact match fails
                    if self._text_similar("\n".join(file_section), "\n".join(minus_lines)):
                        if self.verbose:
                            print(f"Fuzzy replacing {len(minus_lines)} lines at line {minus_start + 1}")
                        file_lines[minus_start : minus_start + len(minus_lines)] = plus_lines
                        modified = True
                    else:
                        print(f"Failed to match lines at {minus_start + 1}, skipping this hunk")
                        if self.verbose:
                            print(f"Expected:\n{minus_lines}\nFound:\n{file_section}")

            if modified:
                # Write the modified content back to the file
                modified_content = "\n".join(file_lines)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(modified_content)

                print(f"Successfully applied fix to {file_path}")
                os.unlink(diff_file)  # Remove the patch file on success
                return True
            else:
                print(f"No changes applied to {file_path}")
                print(f"Diff saved to {diff_file}")
                return False

        except Exception as e:
            print(f"Error applying diff to {file_path}: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _text_similar(self, text1, text2, threshold=0.8):
        """Check if two text blocks are similar enough."""
        import difflib

        return difflib.SequenceMatcher(None, text1, text2).ratio() >= threshold

    def fix_errors(
        self, errors: List[RuffError], config: Optional[str] = None, max_errors: Optional[int] = None
    ) -> Tuple[int, int]:
        """Fix the given Ruff errors.

        Args:
            errors: List of RuffError objects to fix
            config: Optional path to Ruff config file
            max_errors: Maximum number of errors to fix (None for all)

        Returns:
            Tuple of (number of errors fixed, number of errors attempted)
        """
        fixed_count = 0
        attempted_count = 0

        # Group errors by file
        errors_by_file = {}
        for error in errors:
            if error.file_path not in errors_by_file:
                errors_by_file[error.file_path] = []
            errors_by_file[error.file_path].append(error)

        # Process one file at a time
        total_files = len(errors_by_file)
        file_counter = 0

        for file_path, file_errors in errors_by_file.items():
            file_counter += 1
            print(f"\nProcessing file {file_counter}/{total_files}: {file_path} ({len(file_errors)} errors)")

            # Sort errors by line number in descending order to avoid line number shifting
            # (fixing errors from bottom to top of the file)
            file_errors.sort(key=lambda e: e.line_number, reverse=True)

            # Process one error at a time for this file
            for error_index, error in enumerate(file_errors):
                if max_errors is not None and attempted_count >= max_errors:
                    break

                attempted_count += 1
                print(
                    f"  Fixing error {error_index + 1}/{len(file_errors)}: Line {error.line_number} - "
                    f"{error.error_code} {error.message}"
                )

                # Get the latest file content
                file_content = self.get_file_content(error.file_path)
                if not file_content:
                    continue

                # Generate and apply the fix
                diff = self.generate_fix(error, file_content)
                if not diff:
                    continue

                if self.apply_diff(error.file_path, diff):
                    fixed_count += 1

                    # Re-run Ruff on this file to get updated errors
                    if error_index < len(file_errors) - 1:  # If there are more errors to fix in this file
                        print("  Re-running Ruff to get updated errors...")
                        # Just run Ruff on the current file to get updated errors
                        updated_errors = self.run_ruff([file_path], config)

                        # Update the remaining errors for this file
                        updated_file_errors = [e for e in updated_errors if e.file_path == file_path]
                        if updated_file_errors:
                            # Sort by line number in descending order
                            updated_file_errors.sort(key=lambda e: e.line_number, reverse=True)
                            # Replace the remaining errors with the updated ones
                            file_errors[error_index + 1 :] = updated_file_errors
                            print(f"  Found {len(updated_file_errors)} remaining errors after fix")
                        else:
                            # No more errors in this file, skip the rest
                            print("  All errors in this file fixed!")
                            break

        return fixed_count, attempted_count


def main():
    parser = argparse.ArgumentParser(description="Automatically fix Ruff errors using Claude")
    parser.add_argument("paths", nargs="+", help="File paths or directories to check")
    parser.add_argument("--config", help="Path to Ruff config file")
    parser.add_argument("--max-errors", type=int, help="Maximum number of errors to fix")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--model",
        default="claude-3-7-sonnet-20250219",
        help="Claude model to use (default: claude-3-7-sonnet-20250219)",
    )
    parser.add_argument("--api-key", help="Anthropic API key (can also use ANTHROPIC_API_KEY env var)")
    parser.add_argument(
        "--context-lines",
        type=int,
        default=10,
        help="Number of context lines to include before and after the error (default: 5)",
    )

    args = parser.parse_args()

    # Get API key from command line argument or environment variable
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set and --api-key not provided")
        sys.exit(1)

    auto_fix = AutoRuffFix(claude_api_key=api_key, model=args.model, context_lines=args.context_lines)
    auto_fix.set_verbose(args.verbose)

    errors = auto_fix.run_ruff(args.paths, args.config)
    if not errors:
        return

    print(f"Found {len(errors)} Ruff errors")
    # Pass just the config and max_errors to fix_errors
    fixed, attempted = auto_fix.fix_errors(errors, args.config, args.max_errors)

    print(f"\nFixed {fixed}/{attempted} errors")

    # Run Ruff again to check if all errors were fixed
    remaining_errors = auto_fix.run_ruff(args.paths, args.config)
    if remaining_errors:
        print(f"Still {len(remaining_errors)} errors remaining")
    else:
        print("All errors fixed!")


if __name__ == "__main__":
    main()
