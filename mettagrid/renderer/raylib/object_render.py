# disable pylint for raylib
# pylint: disable=no-member
# type: ignore
import os

import pyray as ray
from omegaconf import OmegaConf
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
        self.cfg = cfg
        self.obs_width = 11  # Assuming these values, adjust if necessary
        self.obs_height = 11

    def _sprite_sheet_idx(self, obj):
        # orientation: 0 = Up, 1 = Down, 2 = Left, 3 = Right
        # sprites: 0 = Right, 1 = Up, 2 = Down, 3 = Left
        orientation_offset = [1, 2, 3, 0][obj["agent:orientation"]]

        # return (4 * ((obj["agent_id"] // 12) % 4) + orientation_offset, 2 * (obj["agent_id"] % 12))
        return (orientation_offset, 2 * (obj["agent:species"] % 12))

    def render(self, obj, render_tile_size):
        super().render(obj, render_tile_size)
        self.draw_energy_bar(obj, render_tile_size)
        # self.draw_hp_bar(obj, render_tile_size)
        self.draw_frozen_effect(obj, render_tile_size)
        self.draw_shield_effect(obj, render_tile_size)
        self.draw_observation_area(obj, render_tile_size)

    def draw_energy_bar(self, obj, render_tile_size):
        x = obj["c"] * render_tile_size
        y = obj["r"] * render_tile_size - 8  # 8 pixels above the agent
        width = render_tile_size
        height = 3  # 3 pixels tall
        max_energy = self.cfg.max_energy

        energy = min(max(obj["agent:energy"], 0), max_energy)
        blue_width = int(width * energy / max_energy)

        # Draw red background
        rl.DrawRectangle(x, y, width, height, colors.RED)
        # Draw blue foreground based on energy
        rl.DrawRectangle(x, y, blue_width, height, colors.BLUE)

    def draw_hp_bar(self, obj, render_tile_size):
        x = obj["c"] * render_tile_size
        y = obj["r"] * render_tile_size - 4  # 4 pixels above the agent, below energy bar
        width = render_tile_size
        height = 3  # 3 pixels tall

        hp = min(max(obj["agent:hp"], 0), 10)  # Clamp between 0 and 10
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
            alpha = int(base_alpha * (frozen / self.cfg.freeze_duration))

            # Create a semi-transparent gray color
            frozen_color = ray.Color(128, 128, 128, alpha)

            # Draw the semi-transparent circle
            ray.draw_circle(x, y, radius, frozen_color)

    def draw_shield_effect(self, obj, render_tile_size):
        if obj.get("agent:shield", False):
            x = obj["c"] * render_tile_size + render_tile_size // 2
            y = obj["r"] * render_tile_size + render_tile_size // 2
            radius = render_tile_size // 2 + 2  # Slightly larger than the agent

            # Draw a blue circle
            ray.draw_circle_lines(x, y, radius, ray.BLUE)

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

class GeneratorRenderer(ObjectRenderer):
    def __init__(self):
        super().__init__("items.png", 16)

    def _sprite_sheet_idx(self, obj):
        if obj["generator:ready"]:
            return (14, 2)
        else:
            return (13, 2)

class ConverterRenderer(ObjectRenderer):
    def __init__(self):
        super().__init__("items.png", 16)

    def _sprite_sheet_idx(self, obj):
        if obj["converter:ready"]:
            return (12, 0)
        else:
            return (13, 0)
class AltarRenderer(ObjectRenderer):
    def __init__(self):
        super().__init__("items.png", 16)

    def _sprite_sheet_idx(self, obj):
        if obj["altar:ready"]:
            return (11, 2)
        else:
            return (12, 2)
