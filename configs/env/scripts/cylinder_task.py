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

def create_multiroom_cylinder(num_rooms_horizontal=2, num_rooms_vertical=2, 
                            min_room_size=15, max_room_size=25,
                            num_agents_per_room=1, seed=None):
    """
    Generate a multi-room environment with cylinders in each room.
    Rooms are completely separated by walls.
    """
    if seed is not None:
        random.seed(seed)
        
    # Generate room sizes
    room_sizes = []
    for y in range(num_rooms_vertical):
        row_sizes = []
        for x in range(num_rooms_horizontal):
            width = random.randrange(min_room_size, max_room_size + 1)
            height = random.randrange(min_room_size, max_room_size + 1)
            row_sizes.append((width, height))
        room_sizes.append(row_sizes)
    
    # Calculate total grid size
    row_heights = [max(room[1] for room in row) for row in room_sizes]
    total_height = sum(row_heights) + num_rooms_vertical + 1
    
    column_widths = []
    for col in range(num_rooms_horizontal):
        max_width = max(row[col][0] for row in room_sizes)
        column_widths.append(max_width)
    total_width = sum(column_widths) + num_rooms_horizontal + 1
    
    # Initialize grid with all walls
    grid = {}
    for x in range(total_width):
        for y in range(total_height):
            grid[(x, y)] = 'W'
            
    def create_cylinder_room(width, height, room_x, room_y):
        """Create a cylinder in a single room"""
        # Calculate cylinder parameters based on room size
        cylinder_length = min(width - 4, height - 4)  # Leave space for walls
        orientation = random.choice(['horizontal', 'vertical'])
        cylinder_params = {
            'length': cylinder_length,
            'orientation': orientation,
            'both_ends': random.choice([True, False])
        }
        
        # Create room grid
        room_grid = {}
        for x in range(width):
            for y in range(height):
                room_grid[(x, y)] = ' '
                
        # Add cylinder to room
        center_x = width // 2
        center_y = height // 2
        
        if orientation == 'horizontal':
            # Create parallel walls
            for x in range(center_x - cylinder_length//2, center_x + cylinder_length//2):
                room_grid[(x, center_y - 1)] = 'W'
                room_grid[(x, center_y + 1)] = 'W'
                
            # Place generator
            room_grid[(center_x, center_y)] = 'g'
            
            # Place agents
            for i in range(num_agents_per_room):
                agent_x = center_x - num_agents_per_room//2 + i
                room_grid[(agent_x, center_y - 2)] = 'A'
                
        else:  # vertical
            # Create parallel walls
            for y in range(center_y - cylinder_length//2, center_y + cylinder_length//2):
                room_grid[(center_x - 1, y)] = 'W'
                room_grid[(center_x + 1, y)] = 'W'
                
            # Place generator
            room_grid[(center_x, center_y)] = 'g'
            
            # Place agents
            for i in range(num_agents_per_room):
                agent_y = center_y - num_agents_per_room//2 + i
                room_grid[(center_x - 2, agent_y)] = 'A'
        
        # Place heart altar and converter
        valid_positions = [(x, y) for x in range(2, width-2) 
                         for y in range(2, height-2) 
                         if room_grid[(x, y)] == ' ']
        
        if valid_positions:
            left_positions = [(x, y) for x, y in valid_positions if x < width//2]
            right_positions = [(x, y) for x, y in valid_positions if x >= width//2]
            
            if left_positions and right_positions:
                altar_pos = random.choice(left_positions)
                converter_pos = random.choice(right_positions)
                room_grid[altar_pos] = 'a'
                room_grid[converter_pos] = 'c'
        
        # Add room to main grid
        for x in range(width):
            for y in range(height):
                grid[(room_x + x, room_y + y)] = room_grid[(x, y)]
    
    # Generate and place each room
    current_y = 1
    total_agents = 0
    
    for row_idx, row in enumerate(room_sizes):
        current_x = 1
        max_height_in_row = row_heights[row_idx]
        
        for col_idx, (width, height) in enumerate(row):
            # Create cylinder room
            create_cylinder_room(width, height, current_x, current_y)
            total_agents += num_agents_per_room
            current_x += column_widths[col_idx] + 1
        
        current_y += max_height_in_row + 1
    
    return grid, total_width, total_height, total_agents

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate a multi-room cylinder map')
    parser.add_argument('--num_rooms_h', type=int, help='Number of rooms horizontally')
    parser.add_argument('--num_rooms_v', type=int, help='Number of rooms vertically')
    parser.add_argument('--min_room_size', type=int, help='Minimum room size')
    parser.add_argument('--max_room_size', type=int, help='Maximum room size')
    parser.add_argument('--agents_per_room', type=int, help='Number of agents per room')
    parser.add_argument('--seed', type=int, help='Random seed')

    args = parser.parse_args()

    # Set parameters
    NUM_ROOMS_H = args.num_rooms_h if args.num_rooms_h else random.randint(1, 3)
    NUM_ROOMS_V = args.num_rooms_v if args.num_rooms_v else random.randint(1, 3)
    MIN_ROOM_SIZE = args.min_room_size if args.min_room_size else 15
    MAX_ROOM_SIZE = args.max_room_size if args.max_room_size else 25
    AGENTS_PER_ROOM = args.agents_per_room if args.agents_per_room else 1

    # Generate the multi-room cylinder environment
    grid, width, height, num_agents = create_multiroom_cylinder(
        num_rooms_horizontal=NUM_ROOMS_H,
        num_rooms_vertical=NUM_ROOMS_V,
        min_room_size=MIN_ROOM_SIZE,
        max_room_size=MAX_ROOM_SIZE,
        num_agents_per_room=AGENTS_PER_ROOM,
        seed=args.seed
    )

    # Print info
    print(f"Generated {width}x{height} multi-room cylinder environment")
    print(f"Number of rooms: {NUM_ROOMS_H}x{NUM_ROOMS_V}")
    print(f"Total number of agents: {num_agents}")

    # Print the grid
    for y in range(height):
        for x in range(width):
            print(grid[(x, y)], end='')
        print()

    # Save to file
    os.makedirs("configs/env/mettagrid/maps/cylinders", exist_ok=True)
    filename = f"configs/env/mettagrid/maps/cylinders/multiroom_cylinder_{width}x{height}_agents_{num_agents}.map"
    save_grid(grid, width, height, filename)