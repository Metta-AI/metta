#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26.4",
#   "pixie-python>=4.3.0",
# ]
# ///
"""
Generate a thumbnail image for a given replay file and step number.
"""

import argparse
import collections
import json
import math
import sys
import zlib

import numpy as np
import pixie

obs_radius = 5

colors = {
    "agent": None,
    "wall": pixie.Color(0.380, 0.341, 0.294, 1),
    "$object": pixie.Color(1, 1, 1, 1),
    "$ground": pixie.Color(0.906, 0.831, 0.718, 1),
    "$frame": pixie.Color(0.102, 0.102, 0.102, 1),
    "$shadow": pixie.Color(0, 0, 0, 0.25),
}


def path_agent(path, size):
    path.circle(size / 2, size / 2, size * 0.75)


def path_wall(path, size):
    path.rect(0, 0, size, size)


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
    agent_type_id = -1
    shape = []
    fills = []
    for type_id, object_type in enumerate(input["object_types"]):
        if object_type == "agent":
            agent_type_id = type_id
        if object_type in colors:
            fills.append(colors[object_type])
        else:
            fills.append(colors["$object"])

        match object_type:
            case "agent":
                shape.append(path_agent)

            case _:
                shape.append(path_wall)

    # Broad phase: bucket objects by layer, then sort within each bucket by type.
    groups = collections.defaultdict(list)
    for object in input["grid_objects"]:
        groups[object["layer"]].append(object)
    for layer in groups:
        groups[layer].sort(key=lambda x: -x["type"])  # Draw agents on top of everything.

    # Also draw the background, and setup the drawing state.
    path = pixie.Path()
    path.rect(0, 0, output.width, output.height)
    path.close_path()
    paint = pixie.Paint(pixie.SOLID_PAINT)
    paint.color = colors["$ground"]
    transform = pixie.Matrix3()
    output.fill_path(path, paint, transform)

    stroke = pixie.Paint(pixie.SOLID_PAINT)
    stroke.color = pixie.Color(0, 0, 0, 1)

    # Narrow phase: draw objects.
    last_type = -1
    last_fill = None
    for layer in groups:
        for object in groups[layer]:
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

            # Each agent has a thick border.
            if last_fill is None:
                output.stroke_path(path, stroke, transform, 3)

    # Post-process phase: draw visibility overlay.
    grid_width = input["map_size"][0]
    grid_height = input["map_size"][1]
    visibility_map = np.zeros(grid_width * grid_height, dtype=np.uint8)
    for group in groups:
        for object in groups[group]:
            if object["type"] != agent_type_id:
                continue

            x = get_position_component(object, step, "c") - obs_radius
            y1 = get_position_component(object, step, "r") - obs_radius
            y2 = y1 + obs_radius * 2 + 1
            while y1 < y2:
                y = y1 * grid_width
                x1 = x
                x2 = x1 + obs_radius * 2 + 1
                while x1 < x2:
                    visibility_map[y + x1] = 1
                    x1 += 1
                y1 += 1

        path = pixie.Path()
        for y in range(grid_height):
            offset = y * grid_width
            for x in range(grid_width):
                if visibility_map[offset + x] == 0:
                    path.rect(x * size, y * size, size, size)
        path.close_path()
        paint.color = colors["$shadow"]
        output.fill_path(path, paint)


def gen_frame(image, output):
    path = pixie.Path()
    path.rect(0, 0, output.width, output.height)
    path.close_path()
    paint = pixie.Paint(pixie.SOLID_PAINT)
    paint.color = colors["$frame"]
    output.fill_path(path, paint)

    s = 1
    if image.width > output.width or image.height > output.height:
        if (image.width - output.width) > (image.height - output.height):
            s = output.width / image.width
        else:
            s = output.height / image.height

    transform = pixie.Matrix3()
    transform.values[0] = s
    transform.values[4] = s
    transform.values[6] = (output.width - image.width * s) / 2
    transform.values[7] = (output.height - image.height * s) / 2
    output.draw(image, transform)


def main():
    # Let the user specify the input and output parameters.
    parser = argparse.ArgumentParser(
        description="Generate a thumbnail image for a given replay file and step number.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--debug", "-d", action="store_true", help="Print debug information (default: False)")
    parser.add_argument("--replay", "-r", type=str, required=True, help="Path to replay file (required)")
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

    # Input the replay file into a data object.
    try:
        with open(args.replay, "rb") as file:
            input_raw = file.read()
        input_json = zlib.decompress(input_raw)
        input_data = json.loads(input_json)
    except Exception as e:
        print(f"Error reading replay file: {e}", file=sys.stderr)
        sys.exit(1)

    if args.debug:
        print("Keys:", input_data.keys())
        print("Vals:", input_data["grid_objects"][0].keys())

    # Transform the replay data into a thumbnail image.
    try:
        bounds = input_data["map_size"]
        image = pixie.Image(bounds[0] * args.size, bounds[1] * args.size)
        output = pixie.Image(args.width, args.height)
        gen_thumb(input_data, args.step, args.size, image)
        gen_frame(image, output)
    except Exception as e:
        print(f"Error creating thumbnail image: {e}", file=sys.stderr)
        import traceback

        print(traceback.format_exc())
        sys.exit(1)

    # Output the thumbnail image to another file.
    try:
        output.write_file(args.output)
        print(f"Generated {args.output} of size {args.width}x{args.height} from {args.replay} at step {args.step}")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
