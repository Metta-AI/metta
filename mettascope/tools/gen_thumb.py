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
import json
import math
import sys
import traceback
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


def read_replay_map(input, step):
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

    objects = input["grid_objects"]
    nodes = [0] * len(objects)
    for i, object in enumerate(objects):
        x = get_position_component(object, step, "c")
        y = get_position_component(object, step, "r")
        nodes[i] = y | (x << 16) | (object["type"] << 32) | (object.get("agent_id", 0) << 48)

    size = input["map_size"]
    return [size[0], size[1], nodes, shape, fills, agent_type_id]


def read_ascii_map(input):
    width = input.find(b"\n")
    if width <= 1:
        raise ValueError("Failed to detect the ascii map width.")

    input_len = len(input)
    newline_width = 2 if chr(input[width - 1]) == "\r" else 1
    trailing_newline = 0 if chr(input[input_len - 1]) == "\n" else newline_width

    width1 = width + newline_width
    height_f = (input_len + trailing_newline) / width1
    height = int(height_f)
    if height != height_f:  # All rows are complete when height_f is *.0
        raise ValueError("Failed to detect the ascii map height.")

    nodes = [0] * (width * height)
    num_nodes = 0
    num_agents = 0
    for y in range(height):
        offset = y * width1
        for x in range(width):
            type_id = 0
            agent_id = 0
            match chr(input[offset + x]):
                case "@" | "A" | "1" | "2" | "3" | "4" | "p" | "P":
                    type_id = 0
                    agent_id = num_agents
                    num_agents += 1
                case "#" | "W" | "s":  # TODO unsure about s
                    type_id = 1
                case "m" | "R" | "G" | "B":  # TODO split colors?
                    type_id = 2
                case "_" | "a":
                    type_id = 2
                case "o":
                    type_id = 2
                case "S":
                    type_id = 2
                case "L":
                    type_id = 2
                case "F":
                    type_id = 2
                case "T":
                    type_id = 2
                case "c":
                    type_id = 2
                case "." | " ":
                    continue
                case c:
                    print("Unknown tile code:", c)
                    continue
            nodes[num_nodes] = y | (x << 16) | (type_id << 32) | (agent_id << 48)
            num_nodes += 1

    return [
        width,
        height,
        nodes[:num_nodes],
        [path_agent, path_wall, path_wall],
        [colors["agent"], colors["wall"], colors["$object"]],
        0,
    ]


def gen_thumb(scene, size, output):
    # Broad phase: sort objects by type.
    [grid_width, grid_height, nodes, shape, fills, agent_type_id] = scene
    nodes.sort(key=lambda x: -(x & (0xFFFF << 32)))  # Draw agents on top of everything.

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
    for node in nodes:
        type = (node >> 32) & 0xFFFF

        # Change draw states with object type.
        if type != last_type:
            last_type = type
            path = pixie.Path()
            shape[type](path, size)
            path.close_path()
            last_fill = fills[type]
            if last_fill is not None:
                paint.color = last_fill

        # Each agent has a unique color.
        if last_fill is None:
            n = (node >> 48) + math.pi + math.e + math.sqrt(2)
            paint.color = pixie.Color((n * math.pi) % 1.0, (n * math.e) % 1.0, (n * math.sqrt(2)) % 1.0, 1)

        # Transform and draw the scene object.
        transform.values[6] = size * (0xFFFF & (node >> 16))
        transform.values[7] = size * (0xFFFF & node)
        output.fill_path(path, paint, transform)

        # Each agent has a thick border.
        if last_fill is None:
            output.stroke_path(path, stroke, transform, 3)

    # Post-process phase: draw visibility overlay.
    visibility_map = np.zeros(grid_width * grid_height, dtype=np.uint8)
    for node in nodes:
        type = (node >> 32) & 0xFFFF
        if type != agent_type_id:
            continue

        x = (0xFFFF & (node >> 16)) - obs_radius
        y1 = (0xFFFF & node) - obs_radius
        y2 = y1 + obs_radius * 2 + 1
        while y1 < y2:
            if y1 > 0 and y1 < grid_height:
                y = y1 * grid_width
                x1 = x
                x2 = x1 + obs_radius * 2 + 1
                while x1 < x2:
                    if x1 >= 0 and x1 < grid_width:
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

    # Input the replay file into a data object.
    try:
        with open(args.file, "rb") as file:
            input_raw = file.read()
        if args.file.endswith(".map"):
            input = read_ascii_map(input_raw)
        else:
            input_json = zlib.decompress(input_raw)
            input_data = json.loads(input_json)
            input = read_replay_map(input_data, args.step)
    except Exception as e:
        print(f"Error reading replay file: {e}", file=sys.stderr)
        print(traceback.format_exc())
        sys.exit(1)

    if args.debug:
        print("Keys:", input_data.keys())
        print("Vals:", input_data["grid_objects"][0].keys())

    # Transform the replay data into a thumbnail image.
    try:
        image = pixie.Image(input[0] * args.size, input[1] * args.size)
        output = pixie.Image(args.width, args.height)
        gen_thumb(input, args.size, image)
        gen_frame(image, output)
    except Exception as e:
        print(f"Error creating thumbnail image: {e}", file=sys.stderr)
        print(traceback.format_exc())
        sys.exit(1)

    # Output the thumbnail image to another file.
    try:
        output.write_file(args.output)
        print(f"Generated {args.output} of size {args.width}x{args.height} from {args.file} at step {args.step}")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
