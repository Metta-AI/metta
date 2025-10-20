from enum import Enum


class Orientation(Enum):
    NORTH = 0
    SOUTH = 1
    WEST = 2
    EAST = 3
    NORTHWEST = 4
    NORTHEAST = 5
    SOUTHWEST = 6
    SOUTHEAST = 7

    def __new__(cls, value):
        """Create new Orientation instance."""
        if isinstance(value, str):
            # Handle string initialization like Orientation("north")
            value = value.upper()

            # Handle abbreviations
            abbreviations = {
                "N": "NORTH",
                "S": "SOUTH",
                "W": "WEST",
                "E": "EAST",
                "NW": "NORTHWEST",
                "NE": "NORTHEAST",
                "SW": "SOUTHWEST",
                "SE": "SOUTHEAST",
            }

            if value in abbreviations:
                value = abbreviations[value]

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
    def is_diagonal(self) -> bool:
        """Check if this orientation is diagonal."""
        return self.value >= 4

    def get_opposite(self) -> "Orientation":
        """Get the opposite orientation."""
        opposites = {
            Orientation.NORTH: Orientation.SOUTH,
            Orientation.SOUTH: Orientation.NORTH,
            Orientation.WEST: Orientation.EAST,
            Orientation.EAST: Orientation.WEST,
            Orientation.NORTHWEST: Orientation.SOUTHEAST,
            Orientation.NORTHEAST: Orientation.SOUTHWEST,
            Orientation.SOUTHWEST: Orientation.NORTHEAST,
            Orientation.SOUTHEAST: Orientation.NORTHWEST,
        }
        return opposites[self]

    def get_clockwise(self) -> "Orientation":
        """Get the orientation 90 degrees clockwise."""
        clockwise = {
            Orientation.NORTH: Orientation.EAST,
            Orientation.EAST: Orientation.SOUTH,
            Orientation.SOUTH: Orientation.WEST,
            Orientation.WEST: Orientation.NORTH,
            Orientation.NORTHEAST: Orientation.SOUTHEAST,
            Orientation.SOUTHEAST: Orientation.SOUTHWEST,
            Orientation.SOUTHWEST: Orientation.NORTHWEST,
            Orientation.NORTHWEST: Orientation.NORTHEAST,
        }
        return clockwise[self]

    @classmethod
    def get_orientation_count(cls, allow_diagonals: bool = True) -> int:
        """Get the number of valid orientations."""
        return 8 if allow_diagonals else 4

    def is_valid(self, allow_diagonals: bool = True) -> bool:
        """Check if this orientation is valid given diagonal constraints."""
        return allow_diagonals or not self.is_diagonal


# Short aliases to match C++ style
N = Orientation.NORTH
S = Orientation.SOUTH
W = Orientation.WEST
E = Orientation.EAST
NW = Orientation.NORTHWEST
NE = Orientation.NORTHEAST
SW = Orientation.SOUTHWEST
SE = Orientation.SOUTHEAST
