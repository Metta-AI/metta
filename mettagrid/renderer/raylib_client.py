# disable pylint for raylib
# pylint: disable=no-member
# type: ignore

import os
import sys
import time
import numpy as np
import pyray as ray
from cffi import FFI
from raylib import colors, rl

class ObjectRenderer:
    def __init__(self, sprite_sheet, tile_size=24):
        sprites_dir = "../mettagrid/mettagrid/renderer/assets/"
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
    def __init__(self):
        super().__init__("monsters.png", 16)

    def _sprite_sheet_idx(self, obj):
        # orientation: 0 = Up, 1 = Down, 2 = Left, 3 = Right
        # sprites: 0 = Right, 1 = Up, 2 = Down, 3 = Left
        orientation_offset = [1, 2, 3, 0][obj["agent:orientation"]]
        return ((obj["agent_id"] // 12) % 4 + orientation_offset, 2 * (obj["agent_id"] % 12))

    def render(self, obj, render_tile_size):
        super().render(obj, render_tile_size)
        self.draw_energy_bar(obj, render_tile_size)
        self.draw_hp_bar(obj, render_tile_size)

    def draw_energy_bar(self, obj, render_tile_size):
        x = obj["c"] * render_tile_size
        y = obj["r"] * render_tile_size - 8  # 8 pixels above the agent
        width = render_tile_size
        height = 3  # 3 pixels tall

        energy = min(max(obj["agent:energy"], 0), 100)  # Clamp between 0 and 100
        blue_width = int(width * energy / 100)

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


class MettaRaylibClient:
    def __init__(self, width, height, tile_size=20):
        self.width = width
        self.height = height
        self.sidebar_width = 250
        self.tile_size = tile_size

        rl.InitWindow(width*tile_size + self.sidebar_width, height*tile_size,
            "PufferLib Ray Grid".encode())

        # Load custom font
        font_path = os.path.join("..", "mettagrid", "mettagrid", "renderer", "assets", "arial.ttf")
        assert os.path.exists(font_path), f"Font {font_path} does not exist"
        self.font = rl.LoadFont(font_path.encode())

        self.sprite_renderers = [
            AgentRenderer(),
            WallRenderer(),
            GeneratorRenderer(),
            ConverterRenderer(),
            AltarRenderer(),
        ]
        rl.SetTargetFPS(10)
        self.colors = colors

        camera = ray.Camera2D()
        camera.target = ray.Vector2(0.0, 0.0)
        camera.rotation = 0.0
        camera.zoom = 1.0
        self.camera = camera

        self.ffi = FFI()
        self.selected_object_id = None
        self.selected_agent_idx = None
        self.mind_control = False
        self.paused = False


    def _cdata_to_numpy(self):
        image = rl.LoadImageFromScreen()
        width, height, channels = image.width, image.height, 4
        cdata = self.ffi.buffer(image.data, width*height*channels)
        return np.frombuffer(cdata, dtype=np.uint8).reshape((height, width, channels))[:, :, :3]

    def render(self, current_timestep: int, game_objects):
        while True:
            rl.BeginDrawing()
            rl.BeginMode2D(self.camera)
            rl.ClearBackground([6, 24, 24, 255])
            for obj_id, obj in game_objects.items():
                obj["id"] = obj_id
                self.sprite_renderers[obj["type"]].render(obj, self.tile_size)
                if obj_id == self.selected_object_id:
                    self.draw_selection(obj)

            self.handle_mouse_input(game_objects)
            self.draw_mouse()
            action = self.get_action()

            rl.EndMode2D()
            self.render_sidebar(current_timestep, game_objects)
            rl.EndDrawing()

            if not self.paused or action is not None:
                return {
                    "cdata": self._cdata_to_numpy(),
                    "action": action,
                    "selected_agent_idx": self.selected_agent_idx,
                    "mind_control": self.mind_control
                }


    def handle_mouse_input(self, game_objects):
        if ray.is_mouse_button_pressed(ray.MOUSE_LEFT_BUTTON):
            pos = ray.get_mouse_position()
            grid_x = int(pos.x // self.tile_size)
            grid_y = int(pos.y // self.tile_size)
            for obj in game_objects.values():
                if obj["c"] == grid_x and obj["r"] == grid_y:
                    self.selected_object_id = obj["id"]
                    if "agent_id" in obj:
                        self.selected_agent_idx = obj["agent_id"]
                    break

    def render_sidebar(self, current_timestep, game_objects):
        font_size = 14
        sidebar_x = int(self.width * self.tile_size)
        sidebar_height = int(self.height * self.tile_size)
        rl.DrawRectangle(sidebar_x, 0, self.sidebar_width, sidebar_height, colors.DARKGRAY)

        if self.selected_object_id and self.selected_object_id in game_objects:
            selected_object = game_objects[self.selected_object_id]
            y = 10
            line_height = font_size + 4

            rl.DrawTextEx(self.font, "Selected Object:".encode(),
                          (sidebar_x + 10, y), font_size + 2, 1, colors.YELLOW)
            y += line_height * 2

            for key, value in selected_object.items():
                text = f"{key}: {value}"
                if len(text) > 25:
                    text = text[:22] + "..."
                rl.DrawTextEx(self.font, text.encode(),
                                (sidebar_x + 10, y), font_size, 1, colors.WHITE)
                y += line_height

            rl.DrawLine(sidebar_x + 5, y, sidebar_x + self.sidebar_width - 5, y, colors.LIGHTGRAY)

        # Display current timestep at the bottom of the sidebar
        timestep_text = f"Timestep: {current_timestep}"
        rl.DrawTextEx(self.font, timestep_text.encode(),
                      (sidebar_x + 10, sidebar_height - 30), font_size, 1, colors.WHITE)

    def draw_selection(self, obj):
        x, y = obj["c"] * self.tile_size, obj["r"] * self.tile_size
        color = ray.GREEN if self.mind_control else ray.LIGHTGRAY
        ray.draw_rectangle_lines(x, y, self.tile_size, self.tile_size, color)

    def get_action(self):
        if rl.IsKeyDown(rl.KEY_ESCAPE):
            sys.exit(0)

        key_actions = {
            # move

            rl.KEY_E: (1, 0),
            rl.KEY_Q: (1, 1),
            # rotate
            rl.KEY_W: (2, 0),
            rl.KEY_S: (2, 1),
            rl.KEY_A: (2, 2),
            rl.KEY_D: (2, 3),
            # use
            rl.KEY_U: (3, 0),
        }

        for key, action in key_actions.items():
            if rl.IsKeyDown(key):
                return action

        if rl.IsKeyDown(rl.KEY_GRAVE) and self.selected_object_id is not None:
            self.mind_control = not self.mind_control

        if rl.IsKeyDown(rl.KEY_SPACE):
            self.paused = not self.paused

        if self.mind_control and not self.paused:
            return (0, 0) # noop action

        return None


    def draw_mouse(self):
        ts = self.tile_size
        pos = ray.get_mouse_position()
        mouse_x = int(pos.x // ts)
        mouse_y = int(pos.y // ts)

        # Draw border around the tile
        ray.draw_rectangle_lines(
            mouse_x * ts,
            mouse_y * ts,
            ts,
            ts,
            ray.RED
        )

    def __del__(self):
        # Unload the font when the object is destroyed
        rl.UnloadFont(self.font)
