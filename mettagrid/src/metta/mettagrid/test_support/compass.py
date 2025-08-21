from enum import Enum


class Compass(Enum):
    NORTH = 0
    NORTHEAST = 1
    EAST = 2
    SOUTHEAST = 3
    SOUTH = 4
    SOUTHWEST = 5
    WEST = 6
    NORTHWEST = 7

    def __new__(cls, value):
        """Create new Compass instance."""
        if isinstance(value, str):
            # Handle string initialization like Compass("north")
            value = value.upper()
            for member in cls:
                if member.name == value:
                    return member
            raise ValueError(
                f"Invalid compass string: '{value}'. Valid options: {[m.name.lower() for m in cls if len(m.name) > 2]}"
            )

        # Handle integer initialization (internal use)
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __str__(self) -> str:
        """String representation for printing."""
        return self.name.lower()
