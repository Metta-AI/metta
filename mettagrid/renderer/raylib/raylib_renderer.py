# disable pylint for raylib
# pylint: disable=no-member
# type: ignore
import os
import sys
from collections import defaultdict, deque
import math
import pyray as ray
import torch
from cffi import FFI
from omegaconf import OmegaConf
from raylib import colors, rl

from mettagrid.mettagrid_env import MettaGridEnv
from mettagrid.renderer.raylib.object_render import (
    AgentRenderer,
    AltarRenderer,
    ArmoryRenderer,
    FactoryRenderer,
    GeneratorRenderer,
    LaseryRenderer,
    LabRenderer,
    MineRenderer,
    TempleRenderer,
    WallRenderer,
)
from mettagrid.renderer.raylib.font_renderer import FontRenderer
from mettagrid.renderer.raylib.camera_controller import CameraController

class MettaGridRaylibRenderer:
    def __init__(self, env: MettaGridEnv, cfg: OmegaConf):
        self.cfg = cfg
        self.env = env
        self.grid_width = env.map_width()
        self.grid_height = env.map_height()

        self.window_width = 1280
        self.window_height = 720
        self.num_agents = env.num_agents()

        self.sidebar_width = 250
        self.tile_size = 24
        self._update_layout()

        self.ffi = FFI()

        # Initialize window with default size
        rl.InitWindow(self.window_width, self.window_height, "MettaGrid".encode())
        rl.SetWindowState(rl.FLAG_WINDOW_RESIZABLE)  # Make the window resizable

        font_path = "deps/mettagrid/mettagrid/renderer/assets/Inter-Regular.ttf"
        assert os.path.exists(font_path), f"Font {font_path} does not exist"

        # Load custom font
        self.font_renderer = FontRenderer(self.ffi, font_path)

        self._setup_action_handling()

        self.sprite_renderers = [
            AgentRenderer(cfg),
            WallRenderer(),
            MineRenderer(),
            GeneratorRenderer(),
            AltarRenderer(),
            ArmoryRenderer(),
            LaseryRenderer(),
            LabRenderer(),
            FactoryRenderer(),
            TempleRenderer(),
        ]
        rl.SetTargetFPS(60)
        self.colors = colors

        camera = ray.Camera2D()
        camera.target = ray.Vector2(self.grid_width * self.tile_size / 2, self.grid_height * self.tile_size / 2)
        camera.rotation = 0.0
        camera.zoom = 1.0
        self.camera = camera
        self.camera_controller = CameraController(self.camera)

        self.game_objects = {}
        self.actions = torch.zeros((self.num_agents, 2), dtype=torch.int32)
        self.observations = None
        self.current_timestep = 0
        self.agents = [None for _ in range(self.num_agents)]
        self.action_history = [deque(maxlen=10) for _ in range(self.num_agents)]
        self.action_history_timestep = 0
        self.mind_control = False
        self.user_action = False

        self.selected_object_id = None
        self.selected_agent_idx = None
        self.hover_object_id = None
        self.paused = False
        self.obs_idx = -1

        self.time_accumulator = 0.0
        self.step_time = 1.0 / 10.0

    def update(self, actions, observations, rewards, total_rewards, current_timestep):
        self.actions = actions
        self.observations = observations.permute(0, 3, 1, 2)
        self.current_timestep = current_timestep
        self.game_objects = self.env.grid_objects()
        for obj_id, obj in self.game_objects.items():
            obj["id"] = obj_id
            if "agent_id" in obj:
                agent_id = obj["agent_id"]
                self.agents[agent_id] = obj
                obj["last_reward"] = rewards[agent_id].item()
                obj["total_reward"] = total_rewards[agent_id].item()
        if self.selected_agent_idx is not None and self.mind_control:
            self.actions[self.selected_agent_idx][0] = self.action_ids["noop"]
            self.actions[self.selected_agent_idx][1] = 0

        return self.paused

    def get_actions(self):
        return self.actions

    def render_and_wait(self):
        while True:
            while self.time_accumulator < self.step_time:
                delta_time = rl.GetFrameTime()
                self.time_accumulator += delta_time

                self.camera_controller.update(delta_time, float(rl.GetScreenWidth() - self.sidebar_width), float(rl.GetScreenHeight()))

                self.handle_keyboard_input()
                self.handle_mouse_input()
                self._render()
            self.time_accumulator -= self.step_time

            if self.user_action:
                self.user_action = False
                break
            if not self.paused:
                break
        for agent_id, action in enumerate(self.actions):
            self.action_history[agent_id].append(action)

    def _update_layout(self):
        sidebar_width = 250
        self.tile_size = max(5, min((self.window_width - sidebar_width) // self.grid_width,
                             self.window_height // self.grid_height))
        self.sidebar_width = self.window_width - self.grid_width * self.tile_size

    def _cdata_to_numpy(self):
        return None
        # image = rl.LoadImageFromScreen()
        # width, height, channels = image.width, image.height, 4
        # cdata = self.ffi.buffer(image.data, width*height*channels)
        # return np.frombuffer(cdata, dtype=np.uint8).reshape((height, width, channels))[:, :, :3]

    def _render(self):

        # Update window size if it has changed
        if rl.IsWindowResized():
            self.window_width = rl.GetScreenWidth()
            self.window_height = rl.GetScreenHeight()
            self._update_layout()

        rl.BeginDrawing()
        rl.BeginMode2D(self.camera)
        rl.ClearBackground([6, 24, 24, 255])

        for obj_id, obj in self.game_objects.items():
            self.sprite_renderers[obj["type"]].render(obj, self.tile_size)
            if obj_id == self.selected_object_id:
                self.draw_selection(obj)

        self.draw_mouse()
        self.draw_attacks()

        rl.EndMode2D()
        self.render_sidebar()

        if rl.IsKeyDown(rl.KEY_SLASH): # Also the `?` key.
            self._draw_help_overlay()

        rl.EndDrawing()

    def handle_mouse_input(self):
        mouse_x, mouse_y = self.camera_controller.get_world_mouse_position()

        grid_x = mouse_x // self.tile_size
        grid_y = mouse_y // self.tile_size

        self.hover_object_id = None
        for obj_id, obj in self.game_objects.items():
            if obj["c"] == grid_x and obj["r"] == grid_y:
                self.hover_object_id = obj_id
                break

        if ray.is_mouse_button_pressed(ray.MOUSE_LEFT_BUTTON):
            self.selected_object_id = self.hover_object_id
            if self.selected_object_id is not None and "agent_id" in self.game_objects[self.selected_object_id]:
                self.selected_agent_idx = self.game_objects[self.selected_object_id]["agent_id"]

    def render_sidebar(self):
        font_size = 14
        sidebar_x = rl.GetScreenWidth() - self.sidebar_width
        sidebar_height = rl.GetScreenHeight()
        rl.DrawRectangle(sidebar_x, 0, self.sidebar_width, sidebar_height, colors.DARKGRAY)

        y = 10
        line_height = font_size + 4

        def draw_object_info(title, obj_id, color):
            nonlocal y
            if obj_id and obj_id in self.game_objects:
                obj = self.game_objects[obj_id]
                # rl.DrawTextEx(self.font, f"{title}:".encode(),
                #               (sidebar_x + 10, y), font_size, 1, color)
                self.font_renderer.render_text(f"{title}:", sidebar_x + 10, y, font_size, color)
                y += line_height * 2

                for key, value in obj.items():
                    if ":" in key:
                        key = ":".join(key.split(":")[1:])
                    if isinstance(value, float):
                        text = f"{key}: {value:.2e}"
                    else:
                        text = f"{key}: {value}"
                    if len(text) > 25:
                        text = text[:22] + "..."
                    self.font_renderer.render_text(text, sidebar_x + 10, y, font_size)
                    y += line_height

                if "agent_id" in obj:
                    agent_id = obj["agent_id"]
                    action_txt = ", ".join([
                        f"{self.action_names[action[0]]}({action[1]})"
                        for action in self.action_history[agent_id]])
                    self.font_renderer.render_text(action_txt, sidebar_x + 10, y, font_size)
                    y += line_height

                y += line_height
                rl.DrawLine(sidebar_x + 5, y, sidebar_x + self.sidebar_width - 5, y, colors.LIGHTGRAY)
                y += line_height

        mc = "(locked)" if self.mind_control else ""
        draw_object_info("Selected" + mc, self.selected_object_id, colors.YELLOW)
        draw_object_info("Hover", self.hover_object_id, colors.GREEN)

        if self.selected_agent_idx is not None and self.obs_idx > -1:
            self.obs_idx = min(self.obs_idx, len(self.observations[self.selected_agent_idx]) - 1)
            obs = self.observations[self.selected_agent_idx][self.obs_idx]
            # obs is a 11x11 grid of ints
            # draw a 11x11 grid of text on the sidebar
            for r in range(obs.shape[0]):
                for c in range(obs.shape[1]):
                    text = f"{obs[r][c]}"
                    self.font_renderer.render_text_right_aligned(
                        text,
                        sidebar_x + 10 + (c + 1) * font_size * 3,
                        y + r * font_size,
                        font_size + 2
                    )

        # Display current timestep at the bottom of the sidebar
        timestep_text = f"Timestep: {self.current_timestep}"
        self.font_renderer.render_text(timestep_text, sidebar_x + 10, sidebar_height - 30, font_size)
        feature_name = "disabled"

        # Clamp the observation index to be between -1 and the number of features minus 1
        self.obs_idx = max(-1, min(self.obs_idx, len(self.env.grid_features()) - 1))
        feature_name = self.env.grid_features()[self.obs_idx]

        obs_txt = f"Press ? for help. Obs: {feature_name} (-/=)"
        self.font_renderer.render_text(obs_txt, sidebar_x + 10, sidebar_height - 60, font_size)

    def draw_selection(self, obj):
        x, y = obj["c"] * self.tile_size, obj["r"] * self.tile_size
        color = ray.GREEN if self.mind_control else ray.LIGHTGRAY
        ray.draw_rectangle_lines(x, y, self.tile_size, self.tile_size, color)

    def draw_attacks(self):
        for agent_id, action in enumerate(self.actions):
            agent = self.agents[agent_id]
            attack_color = [
                ray.Color(255, 0, 0, 128),
                ray.Color(0, 255, 0, 128),
                ray.Color(0, 0, 255, 128),
                ray.Color(255, 255, 0, 128),
                ray.Color(0, 255, 255, 128),
                ray.Color(255, 0, 255, 128)][agent["agent:group"] % 5]

            if agent["agent:frozen"]:
                continue
            if self.action_names[action[0]] == "attack_nearest":
                # draw a cone from the agent in the direction it's facing.
                # make it 3 grid squares long and 3 grid squares wide.
                # make it red but transparent.
                # Get base coordinates for agent position
                base_x = agent["c"] * self.tile_size + self.tile_size // 2
                base_y = agent["r"] * self.tile_size + self.tile_size // 2

                # Calculate points based on orientation
                if agent["agent:orientation"] == 0:  # Facing up
                    points = [
                        ray.Vector2(base_x, base_y),
                        ray.Vector2(base_x - self.tile_size * 1.5, base_y - self.tile_size * 3),
                        ray.Vector2(base_x + self.tile_size * 1.5, base_y - self.tile_size * 3)
                    ]
                elif agent["agent:orientation"] == 1:  # Facing down
                    points = [
                        ray.Vector2(base_x, base_y),
                        ray.Vector2(base_x - self.tile_size * 1.5, base_y + self.tile_size * 3),
                        ray.Vector2(base_x + self.tile_size * 1.5, base_y + self.tile_size * 3)
                    ]
                elif agent["agent:orientation"] == 2:  # Facing left
                    points = [
                        ray.Vector2(base_x, base_y),
                        ray.Vector2(base_x - self.tile_size * 3, base_y - self.tile_size * 1.5),
                        ray.Vector2(base_x - self.tile_size * 3, base_y + self.tile_size * 1.5)
                    ]
                else:  # Facing right
                    points = [
                        ray.Vector2(base_x, base_y),
                        ray.Vector2(base_x + self.tile_size * 3, base_y - self.tile_size * 1.5),
                        ray.Vector2(base_x + self.tile_size * 3, base_y + self.tile_size * 1.5)
                    ]
                ray.draw_triangle(points[0], points[1], points[2], attack_color)
            if self.action_names[action[0]] == "attack":
                distance = 1 + (action[1] - 1) // 3
                offset = -((action[1] - 1) % 3 - 1)
                target_loc = self._relative_location(
                    agent["r"], agent["c"], agent["agent:orientation"], distance, offset)

                # Draw red rectangle around target
                ray.draw_circle_lines(
                    target_loc[1] * self.tile_size + self.tile_size // 2,
                    target_loc[0] * self.tile_size + self.tile_size // 2,
                    self.tile_size * 0.2,
                    attack_color
                )

                # Draw red line from attacker to target
                start_x = agent["c"] * self.tile_size + self.tile_size // 2
                start_y = agent["r"] * self.tile_size + self.tile_size // 2
                end_x = target_loc[1] * self.tile_size + self.tile_size // 2
                end_y = target_loc[0] * self.tile_size + self.tile_size // 2
                ray.draw_line(int(start_x), int(start_y), int(end_x), int(end_y), attack_color)

    def _draw_help_overlay(self):
        """ Draws a help overlay on the screen with the available keys and their actions. """

        font_size = 16
        help_text = """
        * Click on an agent to select it
        * `SPACE` toggle pause

        * `-` decrement observation layer index
        * `=` increment observation layer index

        * `E` move forward
        * `Q` move backward

        * `W` face up
        * `S` face down
        * `A` face left
        * `D` face right

        * Key `1` to `9` attacks with that type of an attack.
        * `Z` attack nearest agent
        * `P` put
        * `G` get
        * `O` toggle shield
        * `[` increment color
        * `]` decrement color

        * `~` toggle mind control - the agent will ignore the AI and just sit
            there allowing the player to move it around without the AI making
            decisions.
        """
        # Figure out how big to make the box to fit the text.
        size = self.font_renderer.measure_text(help_text, font_size)
        size_x = int(size.x + 30)
        size_y = int(size.y + 30)
        # Position the box in the middle of the screen.
        pos_x = (rl.GetScreenWidth() - size_x) // 2
        pos_y = (rl.GetScreenHeight() - size_y) // 2
        # Draw a semi-transparent black rectangle behind the text.
        rl.DrawRectangle(pos_x, pos_y, size_x, size_y, (0, 0, 0, 200))
        self.font_renderer.render_text(
            help_text,
            pos_x + 15,
            pos_y + 15,
            font_size
        )

    def handle_keyboard_input(self):
        if rl.IsKeyPressed(rl.KEY_ESCAPE) or rl.WindowShouldClose():
            sys.exit(0)

        if self.selected_agent_idx is not None:
            for key, action in self.key_actions.items():
                if rl.IsKeyPressed(key):
                    self.actions[self.selected_agent_idx][0] = action[0]
                    self.actions[self.selected_agent_idx][1] = action[1]
                    self.user_action = True

            # Do WASD movement actions.
            agent = self.agents[self.selected_agent_idx]
            orientation = agent["agent:orientation"]
            def doAction(action_name, value):
                self.actions[self.selected_agent_idx][0] = self.action_ids[action_name]
                self.actions[self.selected_agent_idx][1] = value
                self.user_action = True

            if rl.IsKeyPressed(rl.KEY_W):
                # Move Up.
                if orientation == 0: doAction("move", 0)
                elif orientation == 1: doAction("move", 1)
                else: doAction("rotate", 0)
            elif rl.IsKeyPressed(rl.KEY_S):
                # Move Down.
                if orientation == 1: doAction("move", 0)
                elif orientation == 0: doAction("move", 1)
                else: doAction("rotate", 1)
            elif rl.IsKeyPressed(rl.KEY_A):
                # Move left.
                if orientation == 2: doAction("move", 0)
                elif orientation == 3: doAction("move", 1)
                else: doAction("rotate", 2)
            elif rl.IsKeyPressed(rl.KEY_D):
                # Move right.
                if orientation == 3: doAction("move", 0)
                elif orientation == 2: doAction("move", 1)
                else: doAction("rotate", 3)

            if rl.IsKeyPressed(rl.KEY_GRAVE):
                self.mind_control = not self.mind_control

        if rl.IsKeyPressed(rl.KEY_MINUS):
            self.obs_idx -= 1
        if rl.IsKeyPressed(rl.KEY_EQUAL):
            self.obs_idx += 1

        if rl.IsKeyPressed(rl.KEY_SPACE):
            self.paused = not self.paused

    def draw_mouse(self):
        ts = self.tile_size
        mouse_x, mouse_y = self.camera_controller.get_world_mouse_position()
        mouse_x = int((mouse_x // ts) * ts)
        mouse_y = int((mouse_y // ts) * ts)

        # Draw border around the tile
        ray.draw_rectangle_lines(mouse_x, mouse_y, ts, ts, ray.GRAY)

    def __del__(self):
        # Unload the font when the object is destroyed
        self.font_renderer.unload()

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
            new_c = c - offset
        elif orientation == 1:
            new_r = r + distance
            new_c = c + offset
        elif orientation == 2:
            new_r = r + offset
            new_c = c - distance
        elif orientation == 3:
            new_r = r - offset
            new_c = c + distance

        new_r = max(0, new_r)
        new_c = max(0, new_c)
        return (new_r, new_c)

    def _setup_action_handling(self):
        # convert any missing actions to noop
        actions_dict = {name: idx for idx, name in enumerate(self.env.action_names())}
        noop_idx = actions_dict["noop"]
        self.action_ids = defaultdict(lambda: noop_idx)
        for name in actions_dict:
            self.action_ids[name] = actions_dict[name]
        self.action_names = self.env.action_names()

        self.key_actions = {
            # move
            rl.KEY_E: (self.action_ids["move"], 0),
            rl.KEY_Q: (self.action_ids["move"], 1),
            # use
            rl.KEY_U: (self.action_ids["use"], 0),
            rl.KEY_P: (self.action_ids["put_recipe_items"], 0),
            rl.KEY_G: (self.action_ids["get_output"], 0),
            # color manipulation
            rl.KEY_RIGHT_BRACKET: (self.action_ids["change_color"], 0), # Increment color
            rl.KEY_LEFT_BRACKET: (self.action_ids["change_color"], 1),  # Decrement color
            # attack
            rl.KEY_KP_1: (self.action_ids["attack"], 1),  # KEY_1
            rl.KEY_KP_2: (self.action_ids["attack"], 2),  # KEY_2
            rl.KEY_KP_3: (self.action_ids["attack"], 3),  # KEY_3
            rl.KEY_KP_4: (self.action_ids["attack"], 4),  # KEY_4
            rl.KEY_KP_5: (self.action_ids["attack"], 5),  # KEY_5
            rl.KEY_KP_6: (self.action_ids["attack"], 6),  # KEY_6
            rl.KEY_KP_7: (self.action_ids["attack"], 7),  # KEY_7
            rl.KEY_KP_8: (self.action_ids["attack"], 8),  # KEY_8
            rl.KEY_KP_9: (self.action_ids["attack"], 9),  # KEY_9

            rl.KEY_ONE: (self.action_ids["attack"], 1),   # KEY_1
            rl.KEY_TWO: (self.action_ids["attack"], 2),   # KEY_2
            rl.KEY_THREE: (self.action_ids["attack"], 3), # KEY_3
            rl.KEY_FOUR: (self.action_ids["attack"], 4),  # KEY_4
            rl.KEY_FIVE: (self.action_ids["attack"], 5),  # KEY_5
            rl.KEY_SIX: (self.action_ids["attack"], 6),   # KEY_6
            rl.KEY_SEVEN: (self.action_ids["attack"], 7), # KEY_7
            rl.KEY_EIGHT: (self.action_ids["attack"], 8), # KEY_8
            rl.KEY_NINE: (self.action_ids["attack"], 9),  # KEY_9

            rl.KEY_Z: (self.action_ids["attack_nearest"], 0),

            # toggle shield
            rl.KEY_O: (self.action_ids["shield"], 0),
            # swap
            # rl.KEY_P: (self.action_ids["swap"], 0),
        }
