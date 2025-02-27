# disable pylint for raylib
# pylint: disable=no-member
# type: ignore
import __future__
import os

import pyray as ray
from omegaconf import OmegaConf, DictConfig
from raylib import colors, rl


class ObjectRenderer:
    def __init__(self, sprite_sheet, tile_size=24):
        sprites_dir = "deps/mettagrid/mettagrid/renderer/assets/"
        sprite_sheet_path = os.path.join(sprites_dir, sprite_sheet)
        assert os.path.exists(sprite_sheet_path), f"Sprite sheet {sprite_sheet_path} does not exist"
        self.sprite_sheet = rl.LoadTexture(sprite_sheet_path.encode())
        self.tile_size = tile_size

    def _sprite_sheet_idx(self, obj):
        return (0, 0)

    def render(self, obj, render_tile_size):
        dest_rect = (
            obj["c"] * render_tile_size, obj["r"] * render_tile_size,
            render_tile_size, render_tile_size)
        tile_idx_x, tile_idx_y = self._sprite_sheet_idx(obj)
        src_rect = (
            tile_idx_x * self.tile_size, tile_idx_y * self.tile_size,
            self.tile_size, self.tile_size)

        rl.DrawTexturePro(
            self.sprite_sheet, src_rect, dest_rect,
            (0, 0), 0, colors.WHITE)

class AgentRenderer(ObjectRenderer):
    def __init__(self, cfg: OmegaConf):
        super().__init__("monsters.png", 16)
        self._cfgs = DictConfig({
            **{
                c.id: OmegaConf.merge(cfg.agent, c.props)
                for c in cfg.groups.values()
            }
        })
        self.sprites = {
            c.id: c.sprite
            for c in cfg.groups.values()
        }

        self.obs_width = 11  # Assuming these values, adjust if necessary
        self.obs_height = 11

    def cfg(self, obj):
        return self._cfgs[obj["agent:group"]]

    def _sprite_sheet_idx(self, obj):
        # orientation: 0 = Up, 1 = Down, 2 = Left, 3 = Right
        # sprites: 0 = Right, 1 = Up, 2 = Down, 3 = Left
        orientation_offset = [1, 2, 3, 0][obj["agent:orientation"]]

        # return (4 * ((obj["agent_id"] // 12) % 4) + orientation_offset, 2 * (obj["agent_id"] % 12))
        idx = self.sprites[obj["agent:group"]]
        return (orientation_offset, idx)

    def render(self, obj, render_tile_size):
        super().render(obj, render_tile_size)
        self.draw_frozen_effect(obj, render_tile_size)
        self.draw_observation_area(obj, render_tile_size)

    def draw_hp_bar(self, obj, render_tile_size):
        x = obj["c"] * render_tile_size
        y = obj["r"] * render_tile_size - 4  # 4 pixels above the agent, below energy bar
        width = render_tile_size
        height = 3  # 3 pixels tall

        hp = min(max(obj["hp"], 0), 10)  # Clamp between 0 and 10
        green_width = int(width * hp / 10)

        # Draw red background
        rl.DrawRectangle(x, y, width, height, colors.RED)
        # Draw green foreground based on HP
        rl.DrawRectangle(x, y, green_width, height, colors.GREEN)

    def draw_frozen_effect(self, obj, render_tile_size):
        frozen = obj.get("agent:frozen", 0)
        if frozen > 0:
            x = obj["c"] * render_tile_size + render_tile_size // 2
            y = obj["r"] * render_tile_size + render_tile_size // 2
            radius = render_tile_size // 2

            # Calculate alpha based on frozen value
            base_alpha = 102  # 40% of 255
            alpha = int(base_alpha * (frozen / self.cfg(obj).freeze_duration))

            # Create a semi-transparent gray color
            frozen_color = ray.Color(128, 128, 128, alpha)

            # Draw the semi-transparent circle
            ray.draw_circle(x, y, radius, frozen_color)

    def draw_observation_area(self, obj, render_tile_size):
        x = obj["c"] * render_tile_size
        y = obj["r"] * render_tile_size
        width = self.obs_width * render_tile_size
        height = self.obs_height * render_tile_size

        # Calculate the top-left corner of the observation area
        obs_x = x - (self.obs_width // 2) * render_tile_size
        obs_y = y - (self.obs_height // 2) * render_tile_size

        # Create a semi-transparent grey color
        grey_color = ray.Color(128, 128, 128, 32)  # RGBA: 128, 128, 128, 25% opacity

        # Draw the semi-transparent grey square
        ray.draw_rectangle(obs_x, obs_y, width, height, grey_color)

class WallRenderer(ObjectRenderer):
    def __init__(self):
        super().__init__("wall.png")

class ConverterRenderer(ObjectRenderer):
    def __init__(self):
        super().__init__("items.png", 16)
    
    def _state(self, obj):
        if obj["converting"]:
            return "converting"
        elif obj["inv:ore"] + obj["inv:battery"] + obj["inv:heart"] + obj["inv:laser"] + obj["inv:armor"] + obj["inv:blueprint"] > 0:
            return "has_inventory"
        else:
            return "empty"

    def _sprite_sheet_idx(self, obj):
        return (0, 0)

class MineRenderer(ConverterRenderer):
    def _sprite_sheet_idx(self, obj):
        return {
            "empty": (13, 2),
            "converting": (15, 2),
            "has_inventory": (14, 2),
        }[self._state(obj)]

class GeneratorRenderer(ConverterRenderer):
    def _sprite_sheet_idx(self, obj):
        return {
            "empty": (1, 2),
            "converting": (0, 2),
            "has_inventory": (2, 2),
        }[self._state(obj)]

class AltarRenderer(ConverterRenderer):
    def _sprite_sheet_idx(self, obj):
        return {
            "empty": (11, 2),
            "converting": (10, 3),
            "has_inventory": (12, 2),
        }[self._state(obj)]

class ArmoryRenderer(ConverterRenderer):
    def _sprite_sheet_idx(self, obj):
        return {
            "empty": (1, 3),
            "converting": (11, 3),
            "has_inventory": (6, 3),
        }[self._state(obj)]

class LaseryRenderer(ConverterRenderer):
    def _sprite_sheet_idx(self, obj):
        return {
            "empty": (2, 5),
            "converting": (6, 5),
            "has_inventory": (5, 5),
        }[self._state(obj)]

class LabRenderer(ConverterRenderer):
    def _sprite_sheet_idx(self, obj):
        return {
            "empty": (4, 1),
            "converting": (6, 1),
            "has_inventory": (5, 1),
        }[self._state(obj)]

class FactoryRenderer(ConverterRenderer):
    def _sprite_sheet_idx(self, obj):
        return {
            "empty": (12, 0),
            "converting": (14, 0),
            "has_inventory": (13, 0),
        }[self._state(obj)]

class TempleRenderer(ConverterRenderer):
    def _sprite_sheet_idx(self, obj):
        return {
            "empty": (8, 2),
            "converting": (6, 2),
            "has_inventory": (7, 2),
        }[self._state(obj)]
