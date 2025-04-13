#!/usr/bin/env python3
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
    def __init__(self, claude_api_key: str, model: str = "claude-3-7-sonnet-20250219"):
        """Initialize the AutoRuffFix tool.

        Args:
            claude_api_key: Anthropic API key for Claude
            model: Claude model to use for generating fixes
        """
        self.client = anthropic.Anthropic(api_key=claude_api_key)
        self.model = model
        self.verbose = False

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

                # Get context lines from the file
                context_lines = self._get_context_lines(file_path, line, 5)

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

    def _get_context_lines(self, file_path: str, line_number: int, context_size: int) -> List[str]:
        """Get context lines around the specified line in the file.

        Args:
            file_path: Path to the file
            line_number: The line number where the error was found (1-based)
            context_size: Number of lines to get before and after the error line

        Returns:
            List of context lines with line numbers
        """
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

        prompt = f"""I need to fix a Ruff linting error in my Python code. Please help me generate a diff that only contains the changes needed to fix this specific issue.

File: {error.file_path}
Error: {error.error_code} at line {error.line_number}, column {error.column}
Message: {error.message}

Context of the error:
```
{context_str}
```

Here's the full file content:
```python
{file_content}
```

Please generate a unified diff that contains ONLY the minimal changes needed to fix this specific {error.error_code} error. The diff should use the standard unified diff format with @@ line markers. 

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
-        raise HTTPException(status_code=404, detail="Client HTML file not found")
+        raise HTTPException(status_code=404, detail="Client HTML file not found") from None

 @app.websocket("/ws")
</diff>
"""

        if self.verbose:
            print(f"Sending request to Claude for {error.file_path}:{error.line_number} ({error.error_code})")

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.0,
                system="You are an expert Python developer who specializes in fixing code style and linting issues. You provide precise, minimal diffs to fix specific issues.",
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

            # Parse the diff to extract what needs to be changed
            old_text = ""
            new_text = ""

            in_content = False
            for line in diff_content.split("\n"):
                if line.startswith("@@"):
                    in_content = True
                    old_text = ""
                    new_text = ""
                elif in_content:
                    if line.startswith("-"):
                        if old_text:
                            old_text += "\n"
                        old_text += line[1:]
                    elif line.startswith("+"):
                        if new_text:
                            new_text += "\n"
                        new_text += line[1:]

            if not old_text or not new_text:
                print(f"Could not extract changes from diff for {file_path}")
                return False

            if self.verbose:
                print(f"Looking to replace:\n{old_text}\n\nWith:\n{new_text}")

            # Find and replace the old text with the new text
            if old_text in file_content:
                modified_content = file_content.replace(old_text, new_text)

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(modified_content)

                print(f"Successfully applied fix to {file_path}")
                os.unlink(diff_file)  # Remove the patch file on success
                return True
            else:
                # Try with normalized whitespace
                import re

                # First, try a more relaxed match with flexible whitespace
                escaped_text = re.escape(old_text.strip()).replace("\\ ", "\\s+")
                pattern = re.compile(escaped_text, re.MULTILINE)
                if pattern.search(file_content):
                    modified_content = pattern.sub(new_text, file_content)

                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(modified_content)

                    print(f"Successfully applied fix to {file_path} (relaxed match)")
                    os.unlink(diff_file)
                    return True

                # If still not found, try a line-by-line approach
                # for long files, this can be time-consuming, so we'll use it as a last resort
                file_lines = file_content.split("\n")
                old_lines = old_text.split("\n")

                if len(old_lines) > 1:  # Only try this for multi-line replacements
                    from difflib import get_close_matches

                    # Find the best matching line for the first line of old_text
                    first_line = old_lines[0].strip()
                    matches = []

                    # Look for potential matches
                    for i, line in enumerate(file_lines):
                        if first_line in line:
                            matches.append((i, line))

                    # If no direct substring matches, try fuzzy matching
                    if not matches:
                        matches = [
                            (file_lines.index(match), match)
                            for match in get_close_matches(first_line, file_lines, n=5, cutoff=0.7)
                        ]

                    if matches:
                        for match_idx, match in matches:
                            # Check if the next lines also match
                            if match_idx + len(old_lines) > len(file_lines):
                                continue  # Not enough lines left

                            # Look for a sequence match
                            matched_section = "\n".join(file_lines[match_idx : match_idx + len(old_lines)])
                            if self._text_similar(matched_section, old_text):
                                # Replace the matching lines with the new lines
                                new_lines = new_text.split("\n")

                                # Update file_lines with the new content
                                file_lines[match_idx : match_idx + len(old_lines)] = new_lines

                                # Write the modified content back
                                modified_content = "\n".join(file_lines)
                                with open(file_path, "w", encoding="utf-8") as f:
                                    f.write(modified_content)

                                print(f"Successfully applied fix to {file_path} (fuzzy line match)")
                                os.unlink(diff_file)
                                return True

                # If everything fails, keep the patch file for manual inspection
                print(f"Could not find the text to replace in {file_path}")
                print(f"Diff saved to {diff_file}")

                # Also save specific instructions for manual fixing
                with open(f"{file_path}.manual_fix", "w", encoding="utf-8") as f:
                    f.write(f"Original text to find:\n{old_text}\n\nNew text to replace with:\n{new_text}")

                print(f"Manual fix instructions saved to {file_path}.manual_fix")
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

    def fix_errors(self, errors: List[RuffError], max_errors: Optional[int] = None) -> Tuple[int, int]:
        """Fix the given Ruff errors.

        Args:
            errors: List of RuffError objects to fix
            max_errors: Maximum number of errors to fix (None for all)

        Returns:
            Tuple of (number of errors fixed, number of errors attempted)
        """
        fixed_count = 0
        attempted_count = 0

        if max_errors is not None:
            errors = errors[:max_errors]

        for error in errors:
            attempted_count += 1
            print(
                f"Fixing error {attempted_count}/{len(errors)}: {error.file_path}:{error.line_number} - {error.error_code} {error.message}"
            )

            file_content = self.get_file_content(error.file_path)
            if not file_content:
                continue

            diff = self.generate_fix(error, file_content)
            if not diff:
                continue

            if self.apply_diff(error.file_path, diff):
                fixed_count += 1

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

    args = parser.parse_args()

    # Get API key from command line argument or environment variable
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set and --api-key not provided")
        sys.exit(1)

    auto_fix = AutoRuffFix(claude_api_key=api_key, model=args.model)
    auto_fix.set_verbose(args.verbose)

    errors = auto_fix.run_ruff(args.paths, args.config)
    if not errors:
        return

    print(f"Found {len(errors)} Ruff errors")
    fixed, attempted = auto_fix.fix_errors(errors, args.max_errors)

    print(f"Fixed {fixed}/{attempted} errors")

    # Run Ruff again to check if all errors were fixed
    remaining_errors = auto_fix.run_ruff(args.paths, args.config)
    if remaining_errors:
        print(f"Still {len(remaining_errors)} errors remaining")
    else:
        print("All errors fixed!")


if __name__ == "__main__":
    main()
