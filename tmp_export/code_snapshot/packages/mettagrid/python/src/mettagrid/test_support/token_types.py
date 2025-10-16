from dataclasses import dataclass

# provide some constants from C++ code @ packages/mettagrid/cpp/include/mettagrid/objects/constants.hpp

# TODO - consider if there is a better way to keep this in sync, perhaps by connecting to the enums via pybind?


@dataclass
class TokenTypes:
    # Object type IDs
    WALL_TYPE_ID: int = 1
    ALTAR_TYPE_ID: int = 10

    # empty token
    EMPTY_TOKEN = [0xFF, 0xFF, 0xFF]

    # three bytes per token
    OBS_TOKEN_SIZE = 3
