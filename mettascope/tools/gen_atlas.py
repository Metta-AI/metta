#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pixie-python>=4.3.0",
# ]
# ///
"""
Generate texture atlas for Mettascope by packing images using Skyline bin packing algorithm.
"""

import hashlib
import json
import math
import os
import sys

import pixie

# Pack all of the images into a single atlas.
# We are using Skyline bin packing algorithm, its simple to implement, fast,
# and works well small number of images. No fancy packing required!
ATLAS_SIZE = 1024 * 8
atlas_image = pixie.Image(ATLAS_SIZE, ATLAS_SIZE)
images = {}
heights = [0] * atlas_image.width
padding = 64
FONTS = [
    {
        "fontName": "plexSans",
        "fontPath": "data/fonts/IBMPlexSans-Regular.ttf",
        "fontSize": 64,
        "fontCharset": "".join(chr(cp) for cp in range(32, 127)),
        "glyphInnerPadding": 2,
    }
]


def compute_build_hash() -> str:
    """Hash this generator and the font files themselves to detect changes."""
    sha = hashlib.sha256()
    with open(os.path.abspath(__file__), "rb") as f:
        sha.update(f.read())
    for font in FONTS:
        with open(font["fontPath"], "rb") as f:
            sha.update(f.read())
    return sha.hexdigest()


def needs_rebuild(atlas_dir):
    """Check if the atlas needs to be rebuilt based on file timestamps."""
    # Check if output files exist
    atlas_png = "dist/atlas.png"
    atlas_json = "dist/atlas.json"

    if not os.path.exists(atlas_png) or not os.path.exists(atlas_json):
        return True, "Output files missing"

    # Get output file timestamps
    output_time = min(os.path.getmtime(atlas_png), os.path.getmtime(atlas_json))

    # Check if any input file is newer than output
    for root, _dirs, files in os.walk(atlas_dir):
        for file in files:
            if file.endswith(".png"):
                img_path = os.path.join(root, file)
                if os.path.getmtime(img_path) > output_time:
                    relative_path = os.path.relpath(img_path, atlas_dir)
                    return True, f"Input file newer: {relative_path}"

    # Check if generator or font files have changed since last build using a build hash.
    with open(atlas_json, "r") as f:
        data = json.load(f)
    saved_hash = data.get("buildHash")
    current_hash = compute_build_hash()
    if saved_hash != current_hash:
        return True, "Build hash changed"

    return False, "No changes detected"


def put_image(img, name):
    """
    Place an image in the atlas at the specified coordinates or find the best position.
    Args:
        img: The pixie Image to place in the atlas
        name: The name/key to use in the images dictionary
    Returns:
        Tuple of (x, y, width, height) indicating the image position
    """
    global heights, atlas_image, images

    # Create a new image with padding
    padded_img = pixie.Image(img.width + 2 * padding, img.height + 2 * padding)
    padded_img.fill(pixie.Color(0, 0, 0, 0))
    padded_img.draw(img, pixie.translate(padding, padding))
    # Duplicate the edges padding to the edges of the image.
    top_line = img.sub_image(0, 0, img.width, 1)
    bottom_line = img.sub_image(0, img.height - 1, img.width, 1)
    left_line = img.sub_image(0, 0, 1, img.height)
    right_line = img.sub_image(img.width - 1, 0, 1, img.height)
    for p in range(padding):
        h = padded_img.height - p - 1
        w = padded_img.width - p - 1
        padded_img.draw(top_line, pixie.translate(padding, p))
        padded_img.draw(bottom_line, pixie.translate(padding, h))
        padded_img.draw(left_line, pixie.translate(p, padding))
        padded_img.draw(right_line, pixie.translate(w, padding))
    # Now duplicate each of the corner pixels to the edges of the image.
    top_left = img.get_color(0, 0)
    top_right = img.get_color(img.width - 1, 0)
    bottom_left = img.get_color(0, img.height - 1)
    bottom_right = img.get_color(img.width - 1, img.height - 1)
    for x in range(padding):
        for y in range(padding):
            padded_img.set_color(x, y, top_left)
            padded_img.set_color(padded_img.width - 1 - x, y, top_right)
            padded_img.set_color(x, padded_img.height - 1 - y, bottom_left)
            padded_img.set_color(padded_img.width - 1 - x, padded_img.height - 1 - y, bottom_right)

    # Find the lowest value in the heights array
    min_height = atlas_image.height
    min_x = -1
    for i in range(len(heights)):
        if heights[i] < min_height:
            this_height = heights[i]
            this_x = i

            # Check if there's enough space for the image
            for j in range(1, padded_img.width):
                if this_x + j >= len(heights) or heights[this_x + j] > this_height:
                    break
            else:
                min_height = this_height
                min_x = this_x

    if min_x == -1:
        print(f"Failed to find a place for: {name}", file=sys.stderr)
        sys.exit(1)

    # Draw the image at the position
    atlas_image.draw(padded_img, pixie.translate(min_x, min_height))
    images[name] = (min_x + padding, min_height + padding, img.width, img.height)

    # Update the heights array
    for i in range(padded_img.width):
        if min_x + i < len(heights):
            heights[min_x + i] = min_height + padded_img.height

    return images[name]


def char_label(cp: int) -> str:
    """Convert a character code to a safe to use label string."""
    return f"U+{cp:04X}"


def generate_font_glyphs(font_cfg: dict):
    """Render glyphs for the font, pack them into the atlas, and return font metadata."""
    if not os.path.exists(font_cfg["fontPath"]):
        print(f"Error: Font file not found: {font_cfg['fontPath']}", file=sys.stderr)
        sys.exit(1)

    typeface = pixie.read_typeface(font_cfg["fontPath"])

    font_name = font_cfg["fontName"]
    fonts_meta = {font_name: {}}
    cps = [ord(ch) for ch in font_cfg["fontCharset"]]

    font = typeface.new_font()
    font.size = float(font_cfg["fontSize"])
    white = pixie.Paint(pixie.SOLID_PAINT)
    white.color = pixie.Color(1.0, 1.0, 1.0, 1.0)
    black = pixie.Paint(pixie.SOLID_PAINT)
    black.color = pixie.Color(0.0, 0.0, 0.0, 1.0)
    font.paint = white
    scale = font.scale()
    ascent_px = typeface.ascent() * scale
    descent_px = typeface.descent() * scale
    line_height_px = font.default_line_height()

    glyphs = {}
    added_count = 0
    # Render and pack glyphs using font typesetting to ensure correct pixel sizes.
    for cp in cps:
        if not typeface.has_glyph(cp):
            raise ValueError(f"Font {font_name} does not have glyph for {char_label(cp)}")
        ch = chr(cp)
        arrangement = font.typeset(ch)
        bounds = arrangement.compute_bounds()
        w = int(math.ceil(bounds.w))
        h = int(math.ceil(bounds.h))
        bearing_x = float(bounds.x)
        bearing_y = float(bounds.y)

        rect = None
        if w > 0 and h > 0:
            # put_image will stretch edges through padding (for walls, etc).
            # we do not want this for glyphs, so we add an inner padding.
            gw = w + 2 * font_cfg["glyphInnerPadding"]
            gh = h + 2 * font_cfg["glyphInnerPadding"]
            img = pixie.Image(gw, gh)
            img.fill(pixie.Color(0, 0, 0, 0))
            # Draw the arrangement translated to inside the padding.
            # First, bake a simple 1px black outline by drawing 8 offsets.
            font.paint = black
            outline_px = 1
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
                img.arrangement_fill_text(
                    arrangement,
                    pixie.translate(
                        -bounds.x + font_cfg["glyphInnerPadding"] + dx * outline_px,
                        -bounds.y + font_cfg["glyphInnerPadding"] + dy * outline_px,
                    ),
                )
            # Then, draw the main glyph in white.
            font.paint = white
            img.arrangement_fill_text(
                arrangement,
                pixie.translate(-bounds.x + font_cfg["glyphInnerPadding"], -bounds.y + font_cfg["glyphInnerPadding"]),
            )
            name = f"fonts/{font_name}/{char_label(cp)}"
            x, y, rw, rh = put_image(img, name)
            rect = [x, y, rw, rh]
            print(f"Added {name} to atlas")
            added_count += 1

        advance = float(typeface.get_advance(cp)) * scale
        glyphs[char_label(cp)] = {
            "rect": rect,
            "advance": advance,
            "bearingX": bearing_x,
            "bearingY": bearing_y,
        }

    # Kerning table: nested map with only non-zero pairs.
    kerning = {}
    present_cps = [cp for cp in cps if char_label(cp) in glyphs]
    for left in present_cps:
        left_label = char_label(left)
        row = None
        for right in present_cps:
            adjust = float(typeface.get_kerning_adjustment(left, right)) * scale
            if adjust != 0.0:
                if row is None:
                    row = {}
                row[char_label(right)] = adjust
        if row:
            kerning[left_label] = row

    fonts_meta[font_name] = {
        "ascent": float(ascent_px),
        "descent": float(descent_px),
        "lineHeight": float(line_height_px),
        "glyphs": glyphs,
        "kerning": kerning,
    }
    print(f"Packed {added_count} glyphs for {font_name} size {font_cfg['fontSize']}")

    return fonts_meta


def main():
    """Main entry point for atlas generation."""
    # Ensure output directory exists
    os.makedirs("dist", exist_ok=True)

    # Walk the data dir:
    atlas_dir = "data/atlas"
    if not os.path.exists(atlas_dir):
        print(f"Error: Atlas directory '{atlas_dir}' not found", file=sys.stderr)
        sys.exit(1)

    # Check if rebuild is needed
    should_rebuild, reason = needs_rebuild(atlas_dir)

    if not should_rebuild:
        print("Atlas is up to date, skipping generation")
        return

    print(f"Rebuilding atlas: {reason}")

    fonts_meta = {}
    for font in FONTS:
        fm = generate_font_glyphs(font)
        fonts_meta.update(fm)

    image_count = 0
    for root, _dirs, files in os.walk(atlas_dir):
        for file in files:
            if file.endswith(".png"):
                img_path = os.path.join(root, file)
                try:
                    img = pixie.read_image(img_path)
                    relative_path = os.path.relpath(img_path, atlas_dir)
                    put_image(img, relative_path)
                    image_count += 1
                    print(f"Added {relative_path} to atlas")
                except Exception as e:
                    print(f"Error processing {img_path}: {e}", file=sys.stderr)
                    sys.exit(1)

    if image_count == 0:
        print("Warning: No images found to pack into atlas", file=sys.stderr)
    else:
        print(f"Successfully packed {image_count} images into atlas")

    # Write the atlas image and the atlas json file.
    for font in FONTS:
        fid = font["fontName"]
        if fid in fonts_meta:
            fonts_meta[fid].update(font)
    atlas_out = {
        "images": dict(images),
        "fonts": fonts_meta,
        "buildHash": compute_build_hash(),
    }
    with open("dist/atlas.json", "w") as f:
        json.dump(atlas_out, f, indent=2)
    atlas_image.write_file("dist/atlas.png")
    print("Atlas generation complete: dist/atlas.png and dist/atlas.json")


if __name__ == "__main__":
    main()
