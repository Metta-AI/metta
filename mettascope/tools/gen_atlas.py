#!/usr/bin/env -S uv run
import json
import os

import pixie

# Pack all of the images into a single atlas.
# We are using Skyline bin packing algorithm, its simple to implement, fast,
# and works well small number of images. No fancy packing required!
ATLAS_SIZE = 1024 * 8
atlas_image = pixie.Image(ATLAS_SIZE, ATLAS_SIZE)
images = {}
heights = [0] * atlas_image.width
padding = 64


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
        quit("failed to find a place for: " + name)

    # Draw the image at the position
    atlas_image.draw(padded_img, pixie.translate(min_x, min_height))
    images[name] = (min_x + padding, min_height + padding, img.width, img.height)

    # Update the heights array
    for i in range(padded_img.width):
        if min_x + i < len(heights):
            heights[min_x + i] = min_height + padded_img.height

    return images[name]


# Walk the data dir:
for root, _dirs, files in os.walk("data"):
    for file in files:
        if file.endswith(".png"):
            img = pixie.read_image(root + "/" + file)
            put_image(img, (root + "/" + file).replace("data/", ""))

# Write the atlas image and the atlas json file.
with open("dist/atlas.json", "w") as f:
    json.dump(images, f)
atlas_image.write_file("dist/atlas.png")
