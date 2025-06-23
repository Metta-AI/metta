"""Multi-room configuration for MettaGrid environments."""

from metta.mettagrid.room.room import Room
from metta.mettagrid.room.room_list import RoomList


class MultiRoom(RoomList):
    def __init__(
        self, room: Room, num_rooms: int, layout: str = "grid", border_width: int = 0, border_object: str = "wall"
    ):
        room_cfgs = [room] * num_rooms
        super().__init__(room_cfgs, layout=layout, border_width=border_width, border_object=border_object)
