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
from pathlib import Path
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

        prompt = f"""
I need to fix a Ruff linting error in my Python code. Please help me generate a diff that only 
contains the changes needed to fix this specific issue.

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

Please generate a unified diff that contains ONLY the changes needed to fix this specific 
{error.error_code} error. The diff should use the standard unified diff format with @@ line
markers. Include ONLY the diff and enclose it between <diff> and </diff> tags.

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
            print(
                f"Sending request to Claude for "
                f"{error.file_path}:{error.line_number} ({error.error_code})"
            )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.0,
                system="You are an expert Python developer who specializes in fixing code style and"
                "linting issues. You provide precise, minimal diffs to fix specific issues.",
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract the diff from the response
            content = response.content[0].text
            diff_match = re.search(r"<diff>(.*?)</diff>", content, re.DOTALL)

            if diff_match:
                return diff_match.group(1).strip()
            else:
                print(
                    f"Failed to extract diff for "
                    f"{error.file_path}:{error.line_number} ({error.error_code})"
                )
                if self.verbose:
                    print(f"Claude response: {content}")
                return None

        except Exception as e:
            print(f"Error generating fix with Claude: {e}")
            return None

    def apply_diff(self, file_path: str, diff_content: str) -> bool:
        """Apply the diff to the file using the patch command.

        Args:
            file_path: Path to the file to be patched
            diff_content: The diff content to apply

        Returns:
            True if patch was successfully applied, False otherwise
        """
        # Write the diff to a temporary file
        diff_file = Path(f"{file_path}.patch")
        try:
            with open(diff_file, "w", encoding="utf-8") as f:
                f.write(diff_content)

            # Apply the patch
            cmd = ["patch", file_path, diff_file]
            if self.verbose:
                print(f"Running: {' '.join(cmd)}")

            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"Successfully applied fix to {file_path}")
                os.unlink(diff_file)
                return True
            else:
                print(f"Failed to apply patch to {file_path}")
                print(f"Patch output: {result.stdout}")
                print(f"Patch error: {result.stderr}")
                print(f"Diff saved to {diff_file}")
                return False

        except Exception as e:
            print(f"Error applying diff to {file_path}: {e}")
            return False

    def fix_errors(
        self, errors: List[RuffError], max_errors: Optional[int] = None
    ) -> Tuple[int, int]:
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
                f"Fixing error {attempted_count}/{len(errors)}: "
                f"{error.file_path}:{error.line_number} - "
                f"{error.error_code} {error.message}"
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
    parser.add_argument(
        "--api-key", help="Anthropic API key (can also use ANTHROPIC_API_KEY env var)"
    )

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
