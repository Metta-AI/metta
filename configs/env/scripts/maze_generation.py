import random
import os

def create_maze(width, height, start_pos=(1,1), end_pos=None, seed=None, branching=0.5):
    """
    Generate a maze with given dimensions, start/end positions, and branching.
    
    Args:
        width (int): Width of maze (must be odd)
        height (int): Height of maze (must be odd)
        start_pos (tuple): Starting position as (x,y) tuple (must be odd coordinates)
        end_pos (tuple): Ending position as (x,y) tuple (must be odd coordinates)
        seed (int): Random seed for reproducible mazes
        branching (float): Value between 0 and 1 controlling maze branching
                          0 = least branching, 1 = most branching
    """
    # Validate inputs
    assert 0 <= branching <= 1, "Branching parameter must be between 0 and 1"
    assert width % 2 == 1 and width >= 3, "Width must be odd and >= 3"
    assert height % 2 == 1 and height >= 3, "Height must be odd and >= 3"
    assert start_pos[0] % 2 == 1 and start_pos[1] % 2 == 1, "Start position must have odd coordinates"
    if end_pos:
        assert end_pos[0] % 2 == 1 and end_pos[1] % 2 == 1, "End position must have odd coordinates"
        assert 0 < end_pos[0] < width and 0 < end_pos[1] < height, "End position must be within maze bounds"
    assert 0 < start_pos[0] < width and 0 < start_pos[1] < height, "Start position must be within maze bounds"

    if seed is not None:
        random.seed(seed)

    EMPTY, WALL = ' ', 'W'
    START, END = 'A', 'a'
    NORTH, SOUTH, EAST, WEST = 'n', 's', 'e', 'w'

    maze = {}
    for x in range(width):
        for y in range(height):
            maze[(x, y)] = WALL

    def should_branch():
        """Determine if we should create a new branch based on complexity"""
        # Higher complexity = more likely to branch
        return random.random() < branching

    def get_preferred_direction(x, y, target_x, target_y):
        """Get the direction that leads closer to the target"""
        if abs(target_x - x) > abs(target_y - y):
            return EAST if target_x > x else WEST
        return SOUTH if target_y > y else NORTH

    def visit(x, y, has_visited, target_x=None, target_y=None):
        """Recursive function to generate maze paths"""
        maze[(x, y)] = EMPTY

        while True:
            unvisited_neighbors = []
            if y > 1 and (x, y - 2) not in has_visited:
                unvisited_neighbors.append(NORTH)
            if y < height - 2 and (x, y + 2) not in has_visited:
                unvisited_neighbors.append(SOUTH)
            if x > 1 and (x - 2, y) not in has_visited:
                unvisited_neighbors.append(WEST)
            if x < width - 2 and (x + 2, y) not in has_visited:
                unvisited_neighbors.append(EAST)

            if not unvisited_neighbors:
                return

            # Complexity affects path choice:
            if target_x and target_y and not should_branch():
                # With low complexity, prefer paths toward the target
                preferred = get_preferred_direction(x, y, target_x, target_y)
                if preferred in unvisited_neighbors:
                    next_direction = preferred
                else:
                    next_direction = random.choice(unvisited_neighbors)
            else:
                # With high complexity, choose random directions more often
                next_direction = random.choice(unvisited_neighbors)

            if next_direction == NORTH:
                next_x, next_y = x, y - 2
                maze[(x, y - 1)] = EMPTY
            elif next_direction == SOUTH:
                next_x, next_y = x, y + 2
                maze[(x, y + 1)] = EMPTY
            elif next_direction == WEST:
                next_x, next_y = x - 2, y
                maze[(x - 1, y)] = EMPTY
            elif next_direction == EAST:
                next_x, next_y = x + 2, y
                maze[(x + 1, y)] = EMPTY

            # Add dead ends based on complexity
            if should_branch() and len(unvisited_neighbors) > 1:
                # Create a dead end branch
                alt_direction = random.choice([d for d in unvisited_neighbors if d != next_direction])
                if alt_direction == NORTH:
                    maze[(x, y - 1)] = EMPTY
                elif alt_direction == SOUTH:
                    maze[(x, y + 1)] = EMPTY
                elif alt_direction == WEST:
                    maze[(x - 1, y)] = EMPTY
                elif alt_direction == EAST:
                    maze[(x + 1, y)] = EMPTY

            has_visited.append((next_x, next_y))
            visit(next_x, next_y, has_visited, 
                  target_x=end_pos[0] if end_pos else None,
                  target_y=end_pos[1] if end_pos else None)

    has_visited = [start_pos]
    visit(start_pos[0], start_pos[1], has_visited,
          target_x=end_pos[0] if end_pos else None,
          target_y=end_pos[1] if end_pos else None)

    maze[start_pos] = START
    if end_pos:
        maze[end_pos] = END

    return maze

def create_multiroom_maze(num_rooms_horizontal=2, num_rooms_vertical=2, min_room_size=11, max_room_size=19, branching=0.5, seed=None):
    """
    Generate a multi-room maze where each room is a separate maze connected by doorways.
    """
    if seed is not None:
        random.seed(seed)
        
    assert min_room_size % 2 == 1, "Minimum room size must be odd"
    assert max_room_size % 2 == 1, "Maximum room size must be odd"
    
    # Generate room sizes
    room_sizes = []
    for y in range(num_rooms_vertical):
        row_sizes = []
        for x in range(num_rooms_horizontal):
            width = random.randrange(min_room_size, max_room_size + 1, 2)
            height = random.randrange(min_room_size, max_room_size + 1, 2)
            row_sizes.append((width, height))
        room_sizes.append(row_sizes)
    
    # Calculate total grid size
    # First calculate maximum height for each row
    row_heights = [max(room[1] for room in row) for row in room_sizes]
    total_height = sum(row_heights) + num_rooms_vertical + 1
    
    # Calculate maximum width for each column
    column_widths = []
    for col in range(num_rooms_horizontal):
        max_width = max(row[col][0] for row in room_sizes)
        column_widths.append(max_width)
    total_width = sum(column_widths) + num_rooms_horizontal + 1
    
    # Initialize the complete grid
    grid = {}
    for x in range(total_width):
        for y in range(total_height):
            grid[(x, y)] = 'W'  # Start with all walls
            
    def add_room_to_grid(room_maze, start_x, start_y, width, height):
        """Add a room's maze to the main grid at the specified position"""
        for x in range(width):
            for y in range(height):
                grid[(start_x + x, start_y + y)] = room_maze[(x, y)]
    
    def add_door(x, y, direction):
        """Add a door between rooms"""
        grid[(x, y)] = ' '
    
    # Generate and place each room
    current_y = 1
    num_agents = 0

    for row_idx, row in enumerate(room_sizes):
        current_x = 1
        max_height_in_row = row_heights[row_idx]
        
        for col_idx, (width, height) in enumerate(row):
            # Generate maze for this room
            start_pos = (1, 1)
            end_pos = (width-2, height-2)
            room_maze = create_maze(width, height, start_pos, end_pos, branching=branching)
            
            # Add room to grid
            add_room_to_grid(room_maze, current_x, current_y, width, height)
            num_agents += 1

            
            # Add horizontal door if not last column
            if col_idx < num_rooms_horizontal - 1:
                door_y = current_y + height // 2
                add_door(current_x + width, door_y, 'horizontal')
            
            # Add vertical door if not last row
            if row_idx < num_rooms_vertical - 1:
                door_x = current_x + width // 2
                add_door(door_x, current_y + height, 'vertical')
            
            current_x += column_widths[col_idx] + 1
        
        current_y += max_height_in_row + 1
    
    # # Place start (A) in first room
    # for y in range(2, min_room_size):
    #     for x in range(2, min_room_size):
    #         if grid[(x, y)] == ' ':
    #             grid[(x, y)] = 'A'
    #             break
    #     else:
    #         continue
    #     break
    
    # # Place end (a) in last room
    # for y in range(total_height-2, total_height-min_room_size, -1):
    #     for x in range(total_width-2, total_width-min_room_size, -1):
    #         if grid[(x, y)] == ' ':
    #             grid[(x, y)] = 'a'
    #             break
    #     else:
    #         continue
    #     break

    
    
    return grid, total_width, total_height, num_agents

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
    # Create a multi-room maze

    import argparse
    import os
    import random

    parser = argparse.ArgumentParser(description='Generate a multi-room maze map')
    parser.add_argument('--num_rooms_h', type=int,
                        help='Number of rooms horizontally')
    parser.add_argument('--num_rooms_v', type=int, 
                        help='Number of rooms vertically')
    parser.add_argument('--min_room_size', type=int,
                        help='Minimum room size')
    parser.add_argument('--max_room_size', type=int,
                        help='Maximum room size') 
    parser.add_argument('--branching', type=float,
                        help='Branching factor for maze generation (0-1) (default: 0.7)')
    parser.add_argument('--seed', type=int, default=random.randint(0,10000),
                        help='Random seed (default: random)')
    parser.add_argument('--output', type=str,
                        help='Output file path (default: auto-generated in configs/env/mettagrid/maps/mazes/)')

    args = parser.parse_args()

    NUM_ROOMS_HORIZONTAL = args.num_rooms_h if args.num_rooms_h else random.randint(1,4)
    NUM_ROOMS_VERTICAL = args.num_rooms_v if args.num_rooms_v else random.randint(1,4)
    # Ensure MIN_ROOM_SIZE is odd
    MIN_ROOM_SIZE = args.min_room_size if args.min_room_size else random.choice(range(5, 15, 2))
    MAX_ROOM_SIZE = args.max_room_size if args.max_room_size else MIN_ROOM_SIZE * 2 + 1
    BRANCHING = args.branching if args.branching else 0.7
    grid, width, height, num_agents = create_multiroom_maze(
        num_rooms_horizontal=NUM_ROOMS_HORIZONTAL,
        num_rooms_vertical=NUM_ROOMS_VERTICAL,
        min_room_size=MIN_ROOM_SIZE,
        max_room_size=MAX_ROOM_SIZE,
        branching=BRANCHING,
        seed=42
    )
    
    # Print the maze
    print(f"Generated {width}x{height} multi-room maze:")
    print_grid(grid, width, height)
    
    # Save to file
    os.makedirs("configs/env/mettagrid/maps/mazes", exist_ok=True)
    filename = f"configs/env/mettagrid/maps/mazes/multiroom_maze_{width}x{height}_numagents_{num_agents}.map"
    save_grid(grid, width, height, filename)

    print(f"Number of agents: {num_agents}")