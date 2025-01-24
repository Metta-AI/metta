from omegaconf import OmegaConf, ListConfig
from mettagrid.config.room_list_builder import RoomListBuilder

class RepeatedRoomBuilder(RoomListBuilder):
    def __init__(self, template: OmegaConf, num_rooms: int, border_width: int = 0, border_object: str = "wall"):
        room_cfgs = [template] * num_rooms
        super().__init__(ListConfig(room_cfgs), border_width=border_width, border_object=border_object)
