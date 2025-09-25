#!/usr/bin/env python3
"""Upload filesets to Google Drive using codeclip."""

import argparse
import sys
from pathlib import Path

# Add paths for imports
sys.path.extend(
    [
        str(Path(__file__).parent.parent / "manybot" / "codebot" / "codeclip"),
        str(Path(__file__).parent.parent / "mettagrid" / "src"),
    ]
)

from codeclip.file import get_context

from metta.utils.file import write_data

# Simple fileset definitions
FILESETS = {
    "readme": ["README.md"],
    "docs": ["README.md", "AGENTS.md", "CLAUDE.md", "roadmap.md", "LICENSE"],
    "rl": ["metta/rl/**/*"],
    "mettagrid": ["packages/mettagrid/**/*"],
    "mettascope": ["mettascope/**/*"],
    "agent": ["agent/**/*"],
    "common": ["common/**/*"],
    "utilities": ["common/**", "library/**", "tools/**", "scripts/**"],
}


def main():
    parser = argparse.ArgumentParser(description="Upload filesets to Google Drive")
    parser.add_argument("filesets", nargs="*", help="Filesets to upload (default: all)")
    parser.add_argument("--folder", required=True, help="Google Drive folder ID")
    args = parser.parse_args()

    # Determine which filesets to process
    targets = args.filesets if args.filesets else list(FILESETS.keys())

    for name in targets:
        if name not in FILESETS:
            print(f"Unknown fileset: {name}")
            continue

        # Expand patterns to files
        files = []
        for pattern in FILESETS[name]:
            files.extend(str(p) for p in Path.cwd().glob(pattern) if p.is_file())

        if not files:
            print(f"No files found for {name}")
            continue

        # Generate context and upload
        context, token_info = get_context(paths=files, raw=False)

        # Print token information to stderr
        total_tokens = token_info.get("total_tokens", 0)
        total_files = token_info.get("total_files", 0)
        print(f"Processing {name}: {total_files} files, {total_tokens:,} tokens", file=sys.stderr)

        gdrive_path = f"gdrive://folder/{args.folder}/{name}.txt"
        write_data(gdrive_path, context, content_type="text/plain")
        print(f"Uploaded {name}: {len(files)} files")


if __name__ == "__main__":
    main()
