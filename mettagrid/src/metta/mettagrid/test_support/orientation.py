from enum import Enum


class Orientation(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __new__(cls, value):
        """Create new Orientation instance."""
        if isinstance(value, str):
            # Handle string initialization like Orientation("up")
            value = value.upper()
            for member in cls:
                if member.name == value:
                    return member
            raise ValueError(f"Invalid orientation string: '{value}'. Valid options: {[m.name.lower() for m in cls]}")

        # Handle integer initialization (internal use)
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __str__(self) -> str:
        """String representation for printing."""
        return self.name.lower()

    @property
    def movement_delta(self) -> tuple[int, int]:
        """Get the (row_delta, col_delta) for this orientation."""
        deltas = {
            Orientation.UP: (-1, 0),
            Orientation.DOWN: (1, 0),
            Orientation.LEFT: (0, -1),
            Orientation.RIGHT: (0, 1),
        }
        return deltas[self]
