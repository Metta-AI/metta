# mettagrid/config/room/zigzag_room.py

from mettagrid.config.room.base_room import BaseRoom

class ZigZagRoom(BaseRoom):
    """
    A room that displays a zigzag pattern across the grid.
    
    Parameters:
        width (int): Room width.
        height (int): Room height.
        border_width (int): Thickness of the room border.
        zigzag_params (dict): Dictionary with:
            - num_zigs (int): Number of zigzag segments.
            - amplitude (int): Horizontal extent of each zig.
            - frequency (int): Number of rows before switching direction.
    """
    def __init__(self, width, height, border_width, zigzag_params):
        super().__init__(width, height, border_width)
        self.num_zigs = zigzag_params.get("num_zigs", 4)
        self.amplitude = zigzag_params.get("amplitude", 3)
        self.frequency = zigzag_params.get("frequency", 2)
        self._generate_zigzag()

    def _generate_zigzag(self):
        """
        Generate a zigzag pattern on the grid.
        This example alternates drawing segments to the right and left.
        """
        direction = 1  # 1 for right, -1 for left
        current_x = self.border_width
        # Iterate over rows, leaving room for borders.
        for row in range(self.border_width, self.height - self.border_width):
            # Draw a horizontal line segment for `frequency` rows.
            for _ in range(self.frequency):
                if row >= self.height - self.border_width:
                    break
                # Ensure we don't exceed the grid boundaries.
                for offset in range(self.amplitude):
                    x = current_x + offset * direction
                    if self.border_width <= x < self.width - self.border_width:
                        self.grid[row][x] = 1  # Mark this cell as part of the zigzag
                row += 1
            # Reverse direction for the next segment.
            direction *= -1
            # Update starting x position accordingly.
            current_x = max(self.border_width, min(self.width - self.border_width - self.amplitude, current_x + self.amplitude * direction))
