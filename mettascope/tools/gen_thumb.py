#!/usr/bin/env -S uv run
"""
Generate a thumbnail image for a given replay file and step number.

This CLI tool uses the thumbnail generation library from mettagrid.mapgen.utils.thumbnail
to provide a command-line interface for generating thumbnails from replay files
or ASCII map files.
"""

import argparse
import json
import sys
import traceback
import zlib

from mettagrid.mapgen.utils.thumbnail import (
    generate_thumbnail_from_ascii,
    generate_thumbnail_from_replay,
)


def main():
    # Let the user specify the input and output parameters.
    parser = argparse.ArgumentParser(
        description="Generate a thumbnail image for a given replay file and step number.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--debug", "-d", action="store_true", help="Print debug information (default: False)")
    parser.add_argument("--file", "-f", type=str, required=True, help="Path to replay or ascii map file (required)")
    parser.add_argument(
        "--output", "-o", type=str, default="thumb.png", help="Path to output file (default: thumb.png)"
    )
    parser.add_argument("--step", "-s", type=int, default=0, help="Step number to process (default: 0)")
    parser.add_argument("--size", "-S", type=int, default=4, help="Size in pixels of a cell (default: 4)")
    parser.add_argument(
        "--width", "-W", type=int, default=800, help="Width in pixels of the output image (default: 800)"
    )
    parser.add_argument(
        "--height", "-H", type=int, default=600, help="Height in pixels of the output image (default: 600)"
    )
    args = parser.parse_args()

    # Early validations unrelated to the replay file.
    if args.step < 0:
        print("Step number must be non-negative", file=sys.stderr)
        sys.exit(1)
    if args.size < 1:
        print("Size must be at least 1 pixel", file=sys.stderr)
        sys.exit(1)

    # Generate thumbnail using the library
    try:
        if args.file.endswith(".map"):
            # Handle ASCII map files
            with open(args.file, "rb") as f:
                ascii_data = f.read()

            thumbnail_data = generate_thumbnail_from_ascii(
                ascii_data, width=args.width, height=args.height, cell_size=args.size
            )
        else:
            # Handle replay files
            with open(args.file, "rb") as f:
                input_raw = f.read()

            input_json = zlib.decompress(input_raw)
            replay_data = json.loads(input_json)

            if args.debug:
                print("Keys:", replay_data.keys())
                if "grid_objects" in replay_data and replay_data["grid_objects"]:
                    print("Vals:", replay_data["grid_objects"][0].keys())
                elif "objects" in replay_data and replay_data["objects"]:
                    print("Vals:", replay_data["objects"][0].keys())

            thumbnail_data = generate_thumbnail_from_replay(
                replay_data, width=args.width, height=args.height, cell_size=args.size, step=args.step
            )

        # Write the thumbnail to file
        with open(args.output, "wb") as f:
            f.write(thumbnail_data)
        print(f"Generated {args.output} of size {args.width}x{args.height} from {args.file} at step {args.step}")

    except Exception as e:
        print(f"Error generating thumbnail: {e}", file=sys.stderr)
        if args.debug:
            print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
