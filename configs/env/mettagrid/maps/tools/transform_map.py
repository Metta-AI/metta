#!/usr/bin/env -S uv run --active
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
CLI tool to transform ASCII map files with rotation and mirroring operations.
Supports .map files used in the metta mettagrid environment.
"""

import argparse
import sys
from pathlib import Path

from metta.map.utils.ascii_grid import (
    load_map_file,
    mirror_lines_horizontal,
    mirror_lines_vertical,
    rotate_lines_90,
    rotate_lines_180,
    rotate_lines_270,
    validate_map_file,
)


def generate_output_filename(input_path: Path, rotate: int, mirror: str | None) -> Path:
    """
    Generate output filename based on transformations applied.

    Args:
        input_path: Path to input file
        rotate: Rotation angle in degrees
        mirror: Mirror direction or None

    Returns:
        Path object for output file
    """
    parts = [input_path.stem]  # filename without extension

    # Add rotation if not 0
    if rotate != 0:
        parts.append(f"rotate_{rotate}")

    # Add mirror if specified
    if mirror:
        parts.append(f"mirror_{mirror}")

    # Reconstruct with original extension
    output_name = "_".join(parts) + input_path.suffix
    return input_path.parent / output_name


def transform_map(input_file: str, output_file: str | None = None, rotate: int = 0, mirror: str | None = None) -> None:
    """
    Transform a map file with rotation and/or mirroring.

    Args:
        input_file: Path to input .map file
        output_file: Path to output file (optional, auto-generated if None)
        rotate: Rotation angle (0, 90, 180, 270)
        mirror: Mirror direction ('horizontal', 'vertical', or None)
    """
    # Check if any transformation is requested
    if rotate == 0 and mirror is None:
        print("Warning: No transformation requested (rotate=0, no mirror)", file=sys.stderr)
        print("Use -r/--rotate or -m/--mirror to specify a transformation", file=sys.stderr)
        sys.exit(0)

    # Validate input file
    input_path = Path(input_file)

    if not input_path.suffix == ".map":
        print(f"Error: Input file '{input_file}' must have .map extension", file=sys.stderr)
        sys.exit(1)

    if not input_path.exists():
        print(f"Error: File '{input_file}' not found", file=sys.stderr)
        sys.exit(1)

    # Validate map file content
    is_valid, error_msg = validate_map_file(input_file)
    if not is_valid:
        print(f"Error: Invalid map file - {error_msg}", file=sys.stderr)
        sys.exit(1)

    try:
        # Load the map file
        lines = load_map_file(input_file)

        original_rows = len(lines)
        original_cols = max(len(line) for line in lines) if lines else 0

        # Apply rotation
        if rotate == 90:
            lines = rotate_lines_90(lines)
        elif rotate == 180:
            lines = rotate_lines_180(lines)
        elif rotate == 270:
            lines = rotate_lines_270(lines)

        # Apply mirroring
        if mirror == "horizontal":
            lines = mirror_lines_horizontal(lines)
        elif mirror == "vertical":
            lines = mirror_lines_vertical(lines)

        # Generate output filename if not specified
        if output_file is None:
            output_path = generate_output_filename(input_path, rotate, mirror)
        else:
            output_path = Path(output_file)

        # Write transformed map
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
            if lines and not lines[-1].endswith("\n"):
                f.write("\n")  # Ensure file ends with newline

        # Print success message and info
        print(f"✓ Transformed map written to: {output_path}")

        final_rows = len(lines)
        final_cols = max(len(line) for line in lines) if lines else 0

        print(f"\nDimensions: {original_rows}×{original_cols} → {final_rows}×{final_cols}")

        transforms = []
        if rotate != 0:
            transforms.append(f"rotated {rotate}°")
        if mirror:
            transforms.append(f"mirrored {mirror}ly")

        if transforms:
            print(f"Transformations: {', '.join(transforms)}")

    except Exception as e:
        print(f"Error during transformation: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Transform ASCII map files with rotation and mirroring",
        epilog="Examples:\n"
        "  %(prog)s map.map -r 90                    # Rotate 90° clockwise\n"
        "  %(prog)s map.map -m horizontal            # Mirror left-right\n"
        "  %(prog)s map.map -r 180 -m vertical       # Rotate 180° and flip\n"
        "  %(prog)s map.map -r 90 -o custom.map      # Rotate with custom output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input_file", help="Input map file (must have .map extension)")

    parser.add_argument("-o", "--output", help="Output file path (default: auto-generated based on transformations)")

    parser.add_argument(
        "-r",
        "--rotate",
        type=int,
        choices=[0, 90, 180, 270],
        default=0,
        help="Rotation angle in degrees clockwise (default: 0)",
    )

    parser.add_argument(
        "-m",
        "--mirror",
        choices=["horizontal", "vertical"],
        help="Mirror direction (horizontal=left-right, vertical=top-bottom)",
    )

    parser.add_argument("-v", "--version", action="version", version="%(prog)s 1.0.0")

    args = parser.parse_args()

    # Run the transformation
    transform_map(input_file=args.input_file, output_file=args.output, rotate=args.rotate, mirror=args.mirror)


if __name__ == "__main__":
    main()
