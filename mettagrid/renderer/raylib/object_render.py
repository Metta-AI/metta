# disable pylint for raylib
# pylint: disable=no-member
# type: ignore
import __future__
import os

import pyray as ray
from omegaconf import OmegaConf, DictConfig
from raylib import colors, rl


class ObjectRenderer:
    def __init__(self, emoji="❓", size=24):
        self.emoji = emoji
        self.size = size

    def render(self, obj, render_tile_size):
        x = obj["c"] * render_tile_size
        y = obj["r"] * render_tile_size
        ray.draw_text(self.emoji, int(x), int(y), render_tile_size, colors.WHITE)

class AgentRenderer(ObjectRenderer):
    def __init__(self, cfg: OmegaConf):
        super().__init__("🤖")
        self._cfgs = DictConfig({
            **{
                c.id: OmegaConf.merge(cfg.agent, c.props)
                for c in cfg.groups.values()
            }
        })
        self.obs_width = 11
        self.obs_height = 11

    def cfg(self, obj):
        return self._cfgs[obj["agent:group"]]

    def render(self, obj, render_tile_size):
        super().render(obj, render_tile_size)
        self.draw_frozen_effect(obj, render_tile_size)
        self.draw_observation_area(obj, render_tile_size)

    def draw_hp_bar(self, obj, render_tile_size):
        x = obj["c"] * render_tile_size
        y = obj["r"] * render_tile_size - 4
        width = render_tile_size
        height = 3

        hp = min(max(obj["agent:hp"], 0), 10)
        green_width = int(width * hp / 10)

        rl.DrawRectangle(x, y, width, height, colors.RED)
        rl.DrawRectangle(x, y, green_width, height, colors.GREEN)

    def draw_frozen_effect(self, obj, render_tile_size):
        frozen = obj.get("agent:frozen", 0)
        if frozen > 0:
            x = obj["c"] * render_tile_size + render_tile_size // 2
            y = obj["r"] * render_tile_size + render_tile_size // 2
            radius = render_tile_size // 2

            base_alpha = 102
            alpha = int(base_alpha * (frozen / self.cfg(obj).freeze_duration))
            frozen_color = ray.Color(128, 128, 128, alpha)
            ray.draw_circle(x, y, radius, frozen_color)

    def draw_observation_area(self, obj, render_tile_size):
        x = obj["c"] * render_tile_size
        y = obj["r"] * render_tile_size
        width = self.obs_width * render_tile_size
        height = self.obs_height * render_tile_size

        obs_x = x - (self.obs_width // 2) * render_tile_size
        obs_y = y - (self.obs_height // 2) * render_tile_size

        grey_color = ray.Color(128, 128, 128, 32)
        ray.draw_rectangle(obs_x, obs_y, width, height, grey_color)

class WallRenderer(ObjectRenderer):
    def __init__(self):
        super().__init__("🧱")

class MineRenderer(ObjectRenderer):
    def __init__(self):
        super().__init__("💣")

    def render(self, obj, render_tile_size):
        self.emoji = "💥" if obj["mine:ready"] else "💣"
        super().render(obj, render_tile_size)

class GeneratorRenderer(ObjectRenderer):
    def __init__(self):
        super().__init__("⚡")

    def render(self, obj, render_tile_size):
        self.emoji = "🔌" if obj["generator:ready"] else "⚡"
        super().render(obj, render_tile_size)

class AltarRenderer(ObjectRenderer):
    def __init__(self):
        super().__init__("🏛️")

    def render(self, obj, render_tile_size):
        self.emoji = "✨" if obj["altar:ready"] else "🏛️"
        super().render(obj, render_tile_size)

class ArmoryRenderer(ObjectRenderer):
    def __init__(self):
        super().__init__("🗡️")

    def render(self, obj, render_tile_size):
        self.emoji = "⚔️" if obj["armory:ready"] else "🗡️"
        super().render(obj, render_tile_size)

class LaseryRenderer(ObjectRenderer):
    def __init__(self):
        super().__init__("🔫")

    def render(self, obj, render_tile_size):
        self.emoji = "🎯" if obj["lasery:ready"] else "🔫"
        super().render(obj, render_tile_size)

class LabRenderer(ObjectRenderer):
    def __init__(self):
        super().__init__("🧪")

    def render(self, obj, render_tile_size):
        self.emoji = "⚗️" if obj["lab:ready"] else "🧪"
        super().render(obj, render_tile_size)

class FactoryRenderer(ObjectRenderer):
    def __init__(self):
        super().__init__("🏭")

    def render(self, obj, render_tile_size):
        self.emoji = "⚙️" if obj["factory:ready"] else "🏭"
        super().render(obj, render_tile_size)

class TempleRenderer(ObjectRenderer):
    def __init__(self):
        super().__init__("⛩️")

    def render(self, obj, render_tile_size):
        self.emoji = "🎎" if obj["temple:ready"] else "⛩️"
        super().render(obj, render_tile_size)
