from mettagrid.config.room.fractal_cylinder import FractalCylinder


def visualize_map(width: int, height: int, recursion_depth: int, min_cylinder_length: int):
    """Generate and visualize a fractal cylinder map"""
    # Create the map generator
    generator = FractalCylinder(
        width=width, height=height, recursion_depth=recursion_depth, min_cylinder_length=min_cylinder_length
    )

    # Generate the map
    grid = generator._build()

    # Convert to ASCII representation
    ascii_map = []
    for row in grid:
        ascii_row = []
        for cell in row:
            if cell == "wall":
                ascii_row.append("W")
            elif cell == "empty":
                ascii_row.append(" ")
            elif cell.startswith("agent"):
                ascii_row.append("A")
            elif cell == "altar":
                ascii_row.append("a")
            else:
                ascii_row.append("?")
        ascii_map.append("".join(ascii_row))

    # Add border
    border = "W" * (width + 2)
    ascii_map = [border] + ["W" + row + "W" for row in ascii_map] + [border]

    # Print the map
    print("\n".join(ascii_map))


if __name__ == "__main__":
    # Test different configurations
    print("Small map (20x20), depth 2:")
    visualize_map(20, 20, 2, 5)

    print("\nMedium map (40x40), depth 3:")
    visualize_map(40, 40, 3, 5)

    print("\nLarge map (60x60), depth 4:")
    visualize_map(60, 60, 4, 5)
