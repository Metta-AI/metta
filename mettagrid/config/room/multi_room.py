from omegaconf import OmegaConf, ListConfig

from mettagrid.config.room.room import OBJECTS, GameObject
from mettagrid.config.room.room_list import RoomList
from mettagrid.config.room.room import Room

class MultiRoom(RoomList):
    def __init__(
        self,
        room: Room,
        num_rooms: int,
        layout: str = "grid",
        border_width: int = 0,
        border_object: GameObject = OBJECTS.Wall):
        room_cfgs = [room] * num_rooms
        super().__init__(
            room_cfgs,
            layout=layout,
            border_width=border_width,
            border_object=border_object)
