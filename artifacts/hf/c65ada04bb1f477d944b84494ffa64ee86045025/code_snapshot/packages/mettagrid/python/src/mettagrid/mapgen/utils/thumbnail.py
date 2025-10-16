"""
Thumbnail generation utilities for MettaGrid environments.

This module contains the core thumbnail generation logic extracted from gen_thumb.py,
making it reusable for both CLI tools and automatic generation during simulations.
All functions are faithful extractions from the original gen_thumb.py with minimal modifications.
"""

import math
import os
import tempfile

import numpy as np
import pixie

from mettagrid.map_builder.ascii import AsciiMapBuilder

# Faithful extraction from gen_thumb.py
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
    """Faithfully extracted from gen_thumb.py."""
    path.circle(size / 2, size / 2, size * 0.75)


def path_wall(path, size):
    """Faithfully extracted from gen_thumb.py."""
    path.rect(0, 0, size, size)


def get_position_component(object, step, component):
    """Faithfully extracted from gen_thumb.py."""
    x = object[component]
    if not isinstance(x, list):
        return x
    result = 0
    for [frame, value] in x:
        if frame > step:
            break
        result = value
    return result


def get_position_component_v2(object, step, component):
    """Get position component for version 2 replay format (uses location array)."""
    location = object["location"]

    # Check if it's animated (list of [frame, value] pairs) or simple location array
    if isinstance(location, list) and len(location) > 0 and isinstance(location[0], list):
        # Animated location - find value at step
        result = [0, 0, 0]  # [c, r, layer]
        for [frame, value] in location:
            if frame > step:
                break
            result = value
        return result[0] if component == "c" else result[1]
    else:
        # Simple location array [c, r, layer]
        return location[0] if component == "c" else location[1]


def read_replay_map(input, step):
    """
    Faithfully extracted from gen_thumb.py with version 2 support added.
    Supports both version 1 and version 2 replay formats.
    """
    version = input.get("version", 1)
    if version not in [1, 2]:
        raise ValueError(f"Unsupported replay version: {version}")

    if input["max_steps"] <= step:
        raise ValueError("Step is out of range")

    # Handle version differences for object types and objects list
    if version == 1:
        object_types = input["object_types"]
        objects_list = input["grid_objects"]
        type_key = "type"
    else:  # version == 2
        object_types = input["type_names"]
        objects_list = input["objects"]
        type_key = "type_id"

    # Setup phase: map object types to drawing functions.
    agent_type_id = -1
    shape = []
    fills = []
    for type_id, object_type in enumerate(object_types):
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

    # Process objects with version-specific position handling
    nodes = [0] * len(objects_list)
    for i, object in enumerate(objects_list):
        if version == 1:
            x = get_position_component(object, step, "c")
            y = get_position_component(object, step, "r")
        else:  # version == 2
            x = get_position_component_v2(object, step, "c")
            y = get_position_component_v2(object, step, "r")

        nodes[i] = y | (x << 16) | (object[type_key] << 32) | (object.get("agent_id", 0) << 48)

    size = input["map_size"]
    return [size[0], size[1], nodes, shape, fills, agent_type_id]


def read_ascii_map(input):
    """Parse YAML/legacy ASCII maps for thumbnail generation."""

    text = input.decode("utf-8") if isinstance(input, (bytes, bytearray)) else str(input)
    config = AsciiMapBuilder.Config.from_str(text)

    height = len(config.map_data)
    width = len(config.map_data[0]) if height else 0

    nodes = [0] * (width * height)
    num_nodes = 0
    num_agents = 0
    for y, row in enumerate(config.map_data):
        for x, char in enumerate(row):
            type_id = 0
            agent_id = 0
            match char:
                case "@" | "A" | "1" | "2" | "3" | "4" | "p" | "P":
                    type_id = 0
                    agent_id = num_agents
                    num_agents += 1
                case "#" | "W" | "s":
                    type_id = 1
                case "m" | "R" | "G" | "B":
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
                case other:
                    print("Unknown tile code:", other)
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
    """Faithfully extracted from gen_thumb.py."""
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
    """
    Scale and center the map image within the output frame, removing excessive margins.
    """
    path = pixie.Path()
    path.rect(0, 0, output.width, output.height)
    path.close_path()
    paint = pixie.Paint(pixie.SOLID_PAINT)
    paint.color = colors["$frame"]
    output.fill_path(path, paint)

    # Calculate scale to fit image within output bounds
    scale = min(output.width / image.width, output.height / image.height)

    transform = pixie.Matrix3()
    transform.values[0] = scale
    transform.values[4] = scale
    transform.values[6] = (output.width - image.width * scale) / 2
    transform.values[7] = (output.height - image.height * scale) / 2
    output.draw(image, transform)


# Minimal wrapper functions needed for CLI and automation
# These are the only new additions - everything above is faithful extraction


def generate_thumbnail_from_replay(replay_data, width=800, height=600, cell_size=4, step=0):
    """
    Generate thumbnail PNG data from replay data.

    This wrapper function is needed because the CLI tool and simulation automation
    both need to generate PNG bytes rather than writing files directly.
    The core logic uses the faithfully extracted functions above.
    """
    # Convert replay data to scene format using faithful extraction
    scene = read_replay_map(replay_data, step=step)

    # Create images using the same logic as original gen_thumb.py
    map_width, map_height = scene[0], scene[1]
    image = pixie.Image(map_width * cell_size, map_height * cell_size)
    output = pixie.Image(width, height)

    # Generate thumbnail using faithful extraction
    gen_thumb(scene, cell_size, image)
    gen_frame(image, output)

    # Convert to PNG bytes (this is the only new part - original wrote to file)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        output.write_file(temp_file.name)

        with open(temp_file.name, "rb") as f:
            png_data = f.read()

        os.unlink(temp_file.name)  # Clean up temp file

    return png_data


def generate_thumbnail_from_ascii(ascii_data, width=800, height=600, cell_size=4):
    """
    Generate thumbnail PNG data from ASCII map data.

    This wrapper function is needed because the CLI tool needs to generate PNG bytes
    rather than writing files directly. The core logic uses the faithfully extracted functions above.
    """
    # Convert ASCII data to scene format using faithful extraction
    scene = read_ascii_map(ascii_data)

    # Create images using the same logic as original gen_thumb.py
    map_width, map_height = scene[0], scene[1]
    image = pixie.Image(map_width * cell_size, map_height * cell_size)
    output = pixie.Image(width, height)

    # Generate thumbnail using faithful extraction
    gen_thumb(scene, cell_size, image)
    gen_frame(image, output)

    # Convert to PNG bytes (this is the only new part - original wrote to file)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        output.write_file(temp_file.name)

        with open(temp_file.name, "rb") as f:
            png_data = f.read()

        os.unlink(temp_file.name)  # Clean up temp file

    return png_data
