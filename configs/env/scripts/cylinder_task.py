import random
import os

def create_cylinder(width, height, cylinder_params, num_agents=1, seed=None):
    """
    Generate a cylinder task environment with parallel walls and multiple agents.
    Supports both horizontal and vertical orientations.
    """
    if seed is not None:
        random.seed(seed)
        
    # Validate inputs
    assert 3 <= cylinder_params['length'], "Cylinder length must be at least 3"
    assert cylinder_params['orientation'] in ['horizontal', 'vertical'], "Invalid orientation"
    
    # Initialize grid
    grid = {}
    for x in range(width):
        for y in range(height):
            grid[(x, y)] = ' '
            
    def place_cylinder():
        """Place parallel walls and generator"""
        cylinder_positions = set()
        
        if cylinder_params['orientation'] == 'horizontal':
            center_y = height // 2
            wall_length = cylinder_params['length']
            start_x = (width - wall_length) // 2
            
            # Create parallel walls with openings
            for x in range(start_x, start_x + wall_length):
                # Skip wall placement at the ends if both_ends is True
                if not cylinder_params['both_ends'] or (x != start_x and x != start_x + wall_length - 1):
                    # Top wall
                    grid[(x, center_y - 1)] = 'W'
                    cylinder_positions.add((x, center_y - 1))
                    # Bottom wall
                    grid[(x, center_y + 1)] = 'W'
                    cylinder_positions.add((x, center_y + 1))
            
            # Place generator in middle
            generator_x = start_x + (wall_length // 2)
            grid[(generator_x, center_y)] = 'g'
            cylinder_positions.add((generator_x, center_y))
            
            # Place agents in a line above the cylinder
            agent_start_x = start_x + (wall_length - num_agents) // 2
            for i in range(num_agents):
                grid[(agent_start_x + i, center_y - 2)] = 'A'
                cylinder_positions.add((agent_start_x + i, center_y - 2))
                
        else:  # vertical orientation
            center_x = width // 2
            wall_length = cylinder_params['length']
            start_y = (height - wall_length) // 2
            
            # Create parallel walls with openings
            for y in range(start_y, start_y + wall_length):
                # Skip wall placement at the ends if both_ends is True
                if not cylinder_params['both_ends'] or (y != start_y and y != start_y + wall_length - 1):
                    # Left wall
                    grid[(center_x - 1, y)] = 'W'
                    cylinder_positions.add((center_x - 1, y))
                    # Right wall
                    grid[(center_x + 1, y)] = 'W'
                    cylinder_positions.add((center_x + 1, y))
            
            # Place generator in middle
            generator_y = start_y + (wall_length // 2)
            grid[(center_x, generator_y)] = 'g'
            cylinder_positions.add((center_x, generator_y))
            
            # Place agents in a line to the left of the cylinder
            agent_start_y = start_y + (wall_length - num_agents) // 2
            for i in range(num_agents):
                grid[(center_x - 2, agent_start_y + i)] = 'A'
                cylinder_positions.add((center_x - 2, agent_start_y + i))
        
        # Get valid positions for other elements
        valid_positions = set()
        for x in range(1, width-1):
            for y in range(1, height-1):
                if (x, y) not in cylinder_positions:
                    valid_positions.add((x, y))
                    
        return valid_positions
    
    def place_elements(valid_positions):
        """Place heart altar and converter"""
        if cylinder_params['orientation'] == 'horizontal':
            # Place them on opposite sides of the top half
            top_positions = [(x, y) for x, y in valid_positions if y < height//2]
            left_positions = [(x, y) for x, y in top_positions if x < width//2]
            right_positions = [(x, y) for x, y in top_positions if x >= width//2]
        else:  # vertical orientation
            # Place them on opposite sides
            left_positions = [(x, y) for x, y in valid_positions if x < width//2]
            right_positions = [(x, y) for x, y in valid_positions if x >= width//2]
        
        if left_positions and right_positions:
            altar_pos = random.choice(left_positions)
            converter_pos = random.choice(right_positions)
            
            grid[altar_pos] = 'a'  # heart altar
            grid[converter_pos] = 'c'  # converter
    
    # Add border walls
    for x in range(width):
        grid[(x, 0)] = 'W'
        grid[(x, height-1)] = 'W'
    for y in range(height):
        grid[(0, y)] = 'W'
        grid[(width-1, y)] = 'W'
        
    # Place cylinder and get valid positions
    valid_positions = place_cylinder()
    
    # Place other elements
    place_elements(valid_positions)
    
    return grid

def print_grid(grid, width, height):
    """Print the grid"""
    for y in range(height):
        for x in range(width):
            print(grid[(x, y)], end='')
        print()

def save_grid(grid, width, height, filename):
    """Save grid to file"""
    with open(filename, 'w') as f:
        for y in range(height):
            for x in range(width):
                f.write(grid[(x, y)])
            f.write('\n')

if __name__ == "__main__":
    # Example usage

    import argparse

    parser = argparse.ArgumentParser(description='Generate a cylinder task map with optional parameters')
    parser.add_argument('--width', type=int, help='Width of map')
    parser.add_argument('--height', type=int, help='Height of map')
    parser.add_argument('--num_agents', type=int, help='Number of agents')
    parser.add_argument('--cylinder_length', type=int, help='Length of cylinder')
    parser.add_argument('--cylinder_orientation', choices=['horizontal', 'vertical'], help='Orientation of cylinder')
    parser.add_argument('--cylinder_both_ends', type=bool, help='Whether cylinder has both ends')
    args = parser.parse_args()

    # Set width and height from args or defaults
    WIDTH = args.width if args.width else random.randint(15, 25)
    HEIGHT = args.height if args.height else random.randint(12, 20)
    num_agents = args.num_agents if args.num_agents else 1

    # Set cylinder parameters from args or defaults
    cylinder_params = {
        'length': args.cylinder_length if args.cylinder_length else random.randint(10, max(min(WIDTH, HEIGHT)-5, 12)),
        'orientation': args.cylinder_orientation if args.cylinder_orientation else random.choice(['horizontal', 'vertical']),
        'both_ends': args.cylinder_both_ends if args.cylinder_both_ends is not None else random.choice([True, False])
    }

    print(f"Grid size: {WIDTH}x{HEIGHT}")
    print(f"Cylinder params: {cylinder_params}")
    
    grid = create_cylinder(WIDTH, HEIGHT, cylinder_params, num_agents=num_agents, seed=42)
    
    print("\nGenerated Cylinder Task:")
    print_grid(grid, WIDTH, HEIGHT)
    
    # Save to file
    os.makedirs("configs/env/mettagrid/maps/cylinders", exist_ok=True)
    filename = f"configs/env/mettagrid/maps/cylinders/cylinder_{WIDTH}x{HEIGHT}.map"
    save_grid(grid, WIDTH, HEIGHT, filename)