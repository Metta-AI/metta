from typing import Dict

from omegaconf import ListConfig

from metta.mettagrid.room.room import Room
from metta.mettagrid.room.room_list import RoomList


class RoomScene(RoomList):
    def __init__(
        self,
        rooms: Dict[str, Room],
        layout: ListConfig,
        border_width: int = 0,
        border_object: str = "wall",
        stack_layout: str = "row",
    ):
        next_stack_layout = "row"
        if stack_layout == "row":
            next_stack_layout = "column"

        room_list = []
        for room_or_list in layout:
            if isinstance(room_or_list, ListConfig):
                room_list.append(RoomScene(rooms, room_or_list, stack_layout=next_stack_layout, border_width=0))
            else:
                room_list.append(rooms[room_or_list])

        super().__init__(room_list, layout=stack_layout, border_width=border_width, border_object=border_object)
