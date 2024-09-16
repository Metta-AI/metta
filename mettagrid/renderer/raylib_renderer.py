# disable pylint for raylib
# pylint: disable=no-member
# type: ignore
import os
import sys

import numpy as np
import pyray as ray
import torch
from cffi import FFI
from omegaconf import OmegaConf
from raylib import colors, rl
from types import SimpleNamespace


Actions = SimpleNamespace(
    Noop = 0,
    Move = 1,
    Rotate = 2,
    Use = 3,
    Attack = 4,
    ToggleShield = 5,
)

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

    def _sprite_sheet_idx(self, obj):
        # orientation: 0 = Up, 1 = Down, 2 = Left, 3 = Right
        # sprites: 0 = Right, 1 = Up, 2 = Down, 3 = Left
        orientation_offset = [1, 2, 3, 0][obj["agent:orientation"]]
        return (4 * ((obj["agent_id"] // 12) % 4) + orientation_offset, 2 * (obj["agent_id"] % 12))

    def render(self, obj, render_tile_size):
        super().render(obj, render_tile_size)
        self.draw_energy_bar(obj, render_tile_size)
        # self.draw_hp_bar(obj, render_tile_size)
        self.draw_frozen_effect(obj, render_tile_size)
        self.draw_shield_effect(obj, render_tile_size)

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


class MettaGridRaylibRenderer:
    def __init__(self, map_width: int, map_height: int, cfg: OmegaConf):
        self.width = map_width
        self.height = map_height
        self.sidebar_width = 250
        self.tile_size = cfg.game.tile_size
        self.cfg = cfg

        rl.InitWindow(self.width*self.tile_size + self.sidebar_width, self.height*self.tile_size,
            "PufferLib Ray Grid".encode())

        # Load custom font
        font_path = os.path.join("deps", "mettagrid", "mettagrid", "renderer", "assets", "arial.ttf")
        assert os.path.exists(font_path), f"Font {font_path} does not exist"
        self.font = rl.LoadFont(font_path.encode())

        self.sprite_renderers = [
            AgentRenderer(cfg.game.objects.agent),
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
        self.hover_object_id = None
        self.mind_control = False
        self.paused = False


    def _cdata_to_numpy(self):
        image = rl.LoadImageFromScreen()
        width, height, channels = image.width, image.height, 4
        cdata = self.ffi.buffer(image.data, width*height*channels)
        return np.frombuffer(cdata, dtype=np.uint8).reshape((height, width, channels))[:, :, :3]

    def render(self, current_timestep: int, game_objects, actions):
        while True:
            rl.BeginDrawing()
            rl.BeginMode2D(self.camera)
            rl.ClearBackground([6, 24, 24, 255])

            agents = [None for _ in range(self.cfg.game.num_agents)]
            for obj_id, obj in game_objects.items():
                obj["id"] = obj_id
                self.sprite_renderers[obj["type"]].render(obj, self.tile_size)
                if obj_id == self.selected_object_id:
                    self.draw_selection(obj)
                if "agent_id" in obj:
                    agents[obj["agent_id"]] = obj
            self.handle_mouse_input(game_objects)
            self.draw_mouse()

            if self.mind_control and self.selected_agent_idx is not None:
                actions[self.selected_agent_idx][0] = Actions.Noop

            action = self.get_action()
            if self.selected_agent_idx is not None and action is not None:
                actions[self.selected_agent_idx][0] = action[0]
                actions[self.selected_agent_idx][1] = action[1]

            self.draw_attacks(game_objects, actions, agents)

            rl.EndMode2D()
            self.render_sidebar(current_timestep, game_objects)
            rl.EndDrawing()

            if not self.paused or action is not None:
                return {
                    "cdata": self._cdata_to_numpy(),
                    "actions": actions
                }


    def handle_mouse_input(self, game_objects):
        pos = ray.get_mouse_position()
        grid_x = int(pos.x // self.tile_size)
        grid_y = int(pos.y // self.tile_size)

        self.hover_object_id = None
        for obj_id, obj in game_objects.items():
            if obj["c"] == grid_x and obj["r"] == grid_y:
                self.hover_object_id = obj_id
                break

        if ray.is_mouse_button_pressed(ray.MOUSE_LEFT_BUTTON):
            self.selected_object_id = self.hover_object_id
            if self.selected_object_id is not None and "agent_id" in game_objects[self.selected_object_id]:
                self.selected_agent_idx = game_objects[self.selected_object_id]["agent_id"]

    def render_sidebar(self, current_timestep, game_objects):
        font_size = 14
        sidebar_x = int(self.width * self.tile_size)
        sidebar_height = int(self.height * self.tile_size)
        rl.DrawRectangle(sidebar_x, 0, self.sidebar_width, sidebar_height, colors.DARKGRAY)

        y = 10
        line_height = font_size + 4

        def draw_object_info(title, obj_id, color):
            nonlocal y
            if obj_id and obj_id in game_objects:
                obj = game_objects[obj_id]
                rl.DrawTextEx(self.font, f"{title}:".encode(),
                              (sidebar_x + 10, y), font_size + 2, 1, color)
                y += line_height * 2

                for key, value in obj.items():
                    if ":" in key:
                        key = ":".join(key.split(":")[1:])
                    text = f"{key}: {value}"
                    if len(text) > 25:
                        text = text[:22] + "..."
                    rl.DrawTextEx(self.font, text.encode(),
                                  (sidebar_x + 10, y), font_size, 1, colors.WHITE)
                    y += line_height

                y += line_height
                rl.DrawLine(sidebar_x + 5, y, sidebar_x + self.sidebar_width - 5, y, colors.LIGHTGRAY)
                y += line_height

        mc = "(locked)" if self.mind_control else ""
        draw_object_info("Selected" + mc, self.selected_object_id, colors.YELLOW)
        draw_object_info("Hover", self.hover_object_id, colors.GREEN)

        # Display current timestep at the bottom of the sidebar
        timestep_text = f"Timestep: {current_timestep}"
        rl.DrawTextEx(self.font, timestep_text.encode(),
                      (sidebar_x + 10, sidebar_height - 30), font_size, 1, colors.WHITE)

    def draw_selection(self, obj):
        x, y = obj["c"] * self.tile_size, obj["r"] * self.tile_size
        color = ray.GREEN if self.mind_control else ray.LIGHTGRAY
        ray.draw_rectangle_lines(x, y, self.tile_size, self.tile_size, color)

    def draw_attacks(self, objects, actions, agents):
        for agent_id, action in enumerate(actions):
            if action[0] != Actions.Attack:
                continue
            agent = agents[agent_id]
            if agent["agent:energy"] < self.cfg.game.actions.attack.cost:
                continue

            distance = 1 + (action[1] - 1) // 3
            offset = -((action[1] - 1) % 3 - 1)
            target_loc = self._relative_location(
                agent["r"], agent["c"], agent["agent:orientation"], distance, offset)

            # Draw red rectangle around target
            ray.draw_circle_lines(
                target_loc[1] * self.tile_size + self.tile_size // 2,
                target_loc[0] * self.tile_size + self.tile_size // 2,
                self.tile_size * 0.2,
                ray.RED
            )

            # Draw red line from attacker to target
            start_x = agent["c"] * self.tile_size + self.tile_size // 2
            start_y = agent["r"] * self.tile_size + self.tile_size // 2
            end_x = target_loc[1] * self.tile_size + self.tile_size // 2
            end_y = target_loc[0] * self.tile_size + self.tile_size // 2
            ray.draw_line(int(start_x), int(start_y), int(end_x), int(end_y), ray.RED)

    def get_action(self):
        if rl.IsKeyDown(rl.KEY_ESCAPE):
            sys.exit(0)

        key_actions = {
            # move
            rl.KEY_E: (Actions.Move, 0),
            rl.KEY_Q: (Actions.Move, 1),
            # rotate
            rl.KEY_W: (Actions.Rotate, 0),
            rl.KEY_S: (Actions.Rotate, 1),
            rl.KEY_A: (Actions.Rotate, 2),
            rl.KEY_D: (Actions.Rotate, 3),
            # use
            rl.KEY_U: (Actions.Use, 0),
            # attack
            rl.KEY_KP_1: (Actions.Attack, 1),  # KEY_1
            rl.KEY_KP_2: (Actions.Attack, 2),  # KEY_2
            rl.KEY_KP_3: (Actions.Attack, 3),  # KEY_3
            rl.KEY_KP_4: (Actions.Attack, 4),  # KEY_4
            rl.KEY_KP_5: (Actions.Attack, 5),  # KEY_5
            rl.KEY_KP_6: (Actions.Attack, 6),  # KEY_6
            rl.KEY_KP_7: (Actions.Attack, 7),  # KEY_7
            rl.KEY_KP_8: (Actions.Attack, 8),  # KEY_8
            rl.KEY_KP_9: (Actions.Attack, 9),  # KEY_9
            # toggle shield
            rl.KEY_O: (Actions.ToggleShield, 0),
        }

        for key, action in key_actions.items():
            if rl.IsKeyDown(key):
                return action

        if rl.IsKeyDown(rl.KEY_GRAVE) and self.selected_object_id is not None:
            self.mind_control = not self.mind_control

        if rl.IsKeyDown(rl.KEY_SPACE):
            self.paused = not self.paused

        return None


    def draw_mouse(self):
        ts = self.tile_size
        pos = ray.get_mouse_position()
        mouse_x = int(pos.x // ts)
        mouse_y = int(pos.y // ts)

        # Draw border around the tile
        ray.draw_rectangle_lines(mouse_x * ts, mouse_y * ts, ts, ts, ray.GRAY)

    def __del__(self):
        # Unload the font when the object is destroyed
        rl.UnloadFont(self.font)

    def _selected_agent(self, objects):
        if self.selected_object_id is None:
            return None
        if "agent" not in objects[self.selected_object_id]:
            return None
        return objects[self.selected_object_id]

    def _relative_location(self, r, c, orientation, distance, offset):
        new_r = r
        new_c = c

        if orientation == 0:
            new_r = r - distance
            new_c = c + offset
        elif orientation == 1:
            new_r = r + distance
            new_c = c - offset
        elif orientation == 2:
            new_r = r + offset
            new_c = c - distance
        elif orientation == 3:
            new_r = r - offset
            new_c = c + distance

        new_r = max(0, new_r)
        new_c = max(0, new_c)
        return (new_r, new_c)
