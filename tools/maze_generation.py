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

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Generate a maze with optional parameters')
    parser.add_argument('--width', type=int, help='Width of maze (odd number >= 11)')
    parser.add_argument('--height', type=int, help='Height of maze (odd number >= 11)') 
    parser.add_argument('--start_x', type=int, help='Starting X position (odd number)')
    parser.add_argument('--start_y', type=int, help='Starting Y position (odd number)')
    parser.add_argument('--end_x', type=int, help='Ending X position (odd number)')
    parser.add_argument('--end_y', type=int, help='Ending Y position (odd number)')
    parser.add_argument('--branching', type=float, help='Maze branching complexity between 0 and 1')
    args = parser.parse_args()

    # Set width and height, ensuring they are odd numbers >= 11
    WIDTH = args.width if args.width else random.randrange(11, 25, 2)
    HEIGHT = args.height if args.height else random.randrange(11, 25, 2)

    # Set start position
    if args.start_x and args.start_y:
        START_POS = (args.start_x, args.start_y)
    else:
        START_POS = (1,1)

    # Set end position
    if args.end_x and args.end_y:
        END_POS = (args.end_x, args.end_y)
    else:
        END_POS = (WIDTH-2, HEIGHT-2)

    # Set branching
    BRANCHING = args.branching if args.branching is not None else 0.5
    
    print(f"width: {WIDTH}, height: {HEIGHT}, start pos: {START_POS}, end pos: {END_POS}, branching: {BRANCHING}")

    # Try different complexity values (0.0 to 1.0)
    maze = create_maze(WIDTH, HEIGHT, start_pos=START_POS, end_pos=END_POS, 
                      seed=42, branching=BRANCHING)  # Higher complexity = more complex maze

    maze_str = ""
    for x in range(WIDTH):
        for y in range(HEIGHT):
            maze_str += maze[(x,y)]
        maze_str += "\n"
        
    os.makedirs("configs/env/mettagrid/maps/mazes", exist_ok=True)

    with open(f"configs/env/mettagrid/maps/mazes/maze_{WIDTH}X{HEIGHT}.map", "w") as f:
        f.write(maze_str)