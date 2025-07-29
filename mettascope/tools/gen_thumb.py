#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pixie-python>=4.3.0",
# ]
# ///
"""
Generate a thumbnail image for a given replay file and step number.
"""

import argparse
import json
import math
import sys
import zlib

import pixie

colors = {
    "agent": None,
    "wall": pixie.Color(0.380, 0.341, 0.294, 1),
    "mine": pixie.Color(0.8, 0, 0, 1),
    "generator": pixie.Color(0, 0, 0.8, 1),
    "altar": pixie.Color(0, 0.8, 0, 1),
    "armory": pixie.Color(0, 0.8, 0.8, 1),
    "lasery": pixie.Color(0.8, 0, 0.8, 1),
    "lab": pixie.Color(0.2, 0.4, 0.8, 1),
    "factory": pixie.Color(0.3, 0.3, 0.3, 1),
    "temple": pixie.Color(0.3, 0.6, 0.9, 1),
    "converter": pixie.Color(0.9, 0.6, 0.3, 1),
    "$ground": pixie.Color(0.906, 0.831, 0.718, 1),
}


def path_agent(path, size):
    path.circle(size / 2, size / 2, size / 2)


def path_wall(path, size):
    path.rect(0, 0, size, size)


def path_mine(path, size):
    path.circle(size / 2, size / 2, size / 4)


def path_generic(path, size):
    s = size / 2
    path.rect(s, s, s, s)


def get_position_component(object, step, component):
    x = object[component]
    if not isinstance(x, list):
        return x
    result = 0
    for [frame, value] in x:
        if frame > step:
            break
        result = value
    return result


def gen_thumb(input, step, size, output):
    if input["version"] != 1:
        raise ValueError("Unsupported replay version")
    if input["max_steps"] <= step:
        raise ValueError("Step is out of range")

    # Setup phase: map object types to drawing functions.
    shape = []
    fills = []
    for object_type in input["object_types"]:
        fills.append(colors[object_type])
        match object_type:
            case "agent":
                shape.append(path_agent)
            case "wall":
                shape.append(path_wall)
            case "mine":
                shape.append(path_mine)
            case _:
                shape.append(path_generic)

    # Broad phase: bucket objects by layer, then sort within each bucket by type.
    objects = input["grid_objects"]
    objects.sort(key=lambda x: (x["layer"], x["type"]))

    # Also draw the background, and setup the drawing state.
    path = pixie.Path()
    path.rect(0, 0, output.width, output.height)
    path.close_path()
    paint = pixie.Paint(pixie.SOLID_PAINT)
    paint.color = colors["$ground"]
    transform = pixie.Matrix3()
    output.fill_path(path, paint, transform)

    # Narrow phase: draw objects.
    last_type = -1
    last_fill = None
    for object in objects:
        type = object["type"]

        # Change draw states with object type.
        if type != last_type:
            last_type = type
            path = pixie.Path()
            shape[type](path, size)
            path.close_path()
            last_fill = fills[type]
            if last_fill is not None:
                paint.color = last_fill

        # Position and filter visible objects.
        x = get_position_component(object, step, "c")
        y = get_position_component(object, step, "r")
        if x < 0 or y < 0:
            continue  # Not displayed for this frame.

        # Each agent has a unique color.
        if last_fill is None:
            n = object["agent_id"] + math.pi + math.e + math.sqrt(2)
            paint.color = pixie.Color((n * math.pi) % 1.0, (n * math.e) % 1.0, (n * math.sqrt(2)) % 1.0, 1)

        # Transform and draw the scene object.
        transform.values[6] = x * size
        transform.values[7] = y * size
        output.fill_path(path, paint, transform)


def main():
    # Let the user specify the input and output parameters.
    parser = argparse.ArgumentParser(
        description="Generate a thumbnail image for a given replay file and step number.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--replay", "-r", type=str, required=True, help="Path to replay file (required)")
    parser.add_argument(
        "--output", "-o", type=str, default="thumb.png", help="Path to output file (default: thumb.png)"
    )
    parser.add_argument("--step", "-s", type=int, default=0, help="Step number to process (default: 0)")
    parser.add_argument("--size", "-S", type=int, default=8, help="Size in pixels of a cell (default: 8)")
    args = parser.parse_args()

    # Early validations unrelated to the replay file.
    if args.step < 0:
        print("Step number must be non-negative", file=sys.stderr)
        sys.exit(1)
    if args.size < 1:
        print("Size must be at least 1 pixel", file=sys.stderr)
        sys.exit(1)

    # Input the replay file into a data object.
    try:
        with open(args.replay, "rb") as file:
            input_raw = file.read()
        input_json = zlib.decompress(input_raw)
        input_data = json.loads(input_json)
    except Exception as e:
        print(f"Error reading replay file: {e}", file=sys.stderr)
        sys.exit(1)

    # Transform the replay data into a thumbnail image.
    try:
        bounds = input_data["map_size"]
        width = bounds[0] * args.size
        height = bounds[1] * args.size
        output = pixie.Image(width, height)
        gen_thumb(input_data, args.step, args.size, output)
    except Exception as e:
        print(f"Error creating thumbnail image: {e}", file=sys.stderr)
        sys.exit(1)

    # Output the thumbnail image to another file.
    try:
        output.write_file(args.output)
        print(f"Generated image: {args.output} of size {width}x{height} from {args.replay} at step {args.step}")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
