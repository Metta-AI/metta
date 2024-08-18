from pdb import set_trace as T
import numpy as np
import os

import pettingzoo
import gymnasium

import pufferlib
from pufferlib.environments.ocean import render

class MettaRaylibClient:
    def __init__(self, width, height, tile_size=32):
        self.width = width
        self.height = height
        self.tile_size = tile_size


        sprite_sheet_path = os.path.join(
            *self.__module__.split('.')[:-1], './puffer_chars.png')
        self.asset_map = {
            1: (0, 0, 128, 128),
            3: (128, 0, 128, 128),
            4: (256, 0, 128, 128),
            # 5: (384, 0, 128, 128),
            5: (512, 0, 128, 128), #star
        }

        from raylib import rl, colors
        rl.InitWindow(width*tile_size, height*tile_size,
            "PufferLib Ray Grid".encode())
        rl.SetTargetFPS(10)
        self.puffer = rl.LoadTexture(sprite_sheet_path.encode())
        self.rl = rl
        self.colors = colors

        import pyray as ray
        camera = ray.Camera2D()
        camera.target = ray.Vector2(0.0, 0.0)
        camera.rotation = 0.0
        camera.zoom = 1.0
        self.camera = camera

        from cffi import FFI
        self.ffi = FFI()

    def _cdata_to_numpy(self):
        image = self.rl.LoadImageFromScreen()
        width, height, channels = image.width, image.height, 4
        cdata = self.ffi.buffer(image.data, width*height*channels)
        return np.frombuffer(cdata, dtype=np.uint8
            ).reshape((height, width, channels))[:, :, :3]

    def render(self, grid):
        rl = self.rl
        colors = self.colors
        ay, ax = None, None

        ts = self.tile_size

        pos = rl.GetMousePosition()
        raw_mouse_x = pos.x
        raw_mouse_y = pos.y
        mouse_x = int(raw_mouse_x // ts)
        mouse_y = int(raw_mouse_y // ts)
        ay = int(np.clip((pos.y - ts*self.height//2) / 50, -3, 3)) + 3
        ax = int(np.clip((pos.x - ts*self.width//2) / 50, -3, 3)) + 3

        if rl.IsKeyDown(rl.KEY_ESCAPE):
            exit(0)

        action_id = 0
        action_arg = 0

        if rl.IsKeyDown(rl.KEY_E):
            action_id = 0
            action_arg = 0
        elif rl.IsKeyDown(rl.KEY_Q):
            action_id = 0
            action_arg = 1

        elif rl.IsKeyDown(rl.KEY_W):
            action_id = 1
            action_arg = 0
        elif rl.IsKeyDown(rl.KEY_S):
            action_id = 1
            action_arg = 1
        elif rl.IsKeyDown(rl.KEY_A):
            action_id = 1
            action_arg = 2
        elif rl.IsKeyDown(rl.KEY_R):
            action_id = 1
            action_arg = 3

        # if rl.IsKeyDown(rl.KEY_LEFT_SHIFT):
        #     target_heros = 2

        action = (action_id, action_arg)

        rl.BeginDrawing()
        rl.BeginMode2D(self.camera)
        rl.ClearBackground([6, 24, 24, 255])
        for y in range(self.height):
            for x in range(self.width):
                tile = grid[y, x]
                tx = x*ts
                ty = y*ts
                if tile == 0:
                    continue
                elif tile == 2:
                    # Wall
                    rl.DrawRectangle(x*ts, y*ts, ts, ts, [0, 0, 0, 255])
                    continue
                else:
                    # Player
                    source_rect = self.asset_map[tile]
                    dest_rect = (tx, ty, ts, ts)
                    rl.DrawTexturePro(self.puffer, source_rect, dest_rect,
                        (0, 0), 0, colors.WHITE)

        # Draw circle at mouse x, y
        rl.DrawCircle(ts*mouse_x + ts//2, ts*mouse_y + ts//8, ts//8, [255, 0, 0, 255])

        rl.EndMode2D()

        # Draw HUD
        # player = entities[0]
        # hud_y = self.height*ts - 2*ts
        # draw_bars(rl, player, 2*ts, hud_y, 10*ts, 24, draw_text=True)

        # off_color = [255, 255, 255, 255]
        # on_color = [0, 255, 0, 255]

        # q_color = on_color if skill_q else off_color
        # w_color = on_color if skill_w else off_color
        # e_color = on_color if skill_e else off_color

        # q_cd = player.q_timer
        # w_cd = player.w_timer
        # e_cd = player.e_timer

        # rl.DrawText(f'Q: {q_cd}'.encode(), 13*ts, hud_y - 20, 40, q_color)
        # rl.DrawText(f'W: {w_cd}'.encode(), 17*ts, hud_y - 20, 40, w_color)
        # rl.DrawText(f'E: {e_cd}'.encode(), 21*ts, hud_y - 20, 40, e_color)
        # rl.DrawText(f'Stun: {player.stun_timer}'.encode(), 25*ts, hud_y - 20, 20, e_color)
        # rl.DrawText(f'Move: {player.move_timer}'.encode(), 25*ts, hud_y, 20, e_color)

        rl.EndDrawing()
        return self._cdata_to_numpy(), action

def draw_bars(rl, entity, x, y, width, height=4, draw_text=False):
    health_bar = entity.health / entity.max_health
    mana_bar = entity.mana / entity.max_mana
    if entity.max_health == 0:
        health_bar = 2
    if entity.max_mana == 0:
        mana_bar = 2
    rl.DrawRectangle(x, y, width, height, [255, 0, 0, 255])
    rl.DrawRectangle(x, y, int(width*health_bar), height, [0, 255, 0, 255])

    if entity.entity_type == 0:
        rl.DrawRectangle(x, y - height - 2, width, height, [255, 0, 0, 255])
        rl.DrawRectangle(x, y - height - 2, int(width*mana_bar), height, [0, 255, 255, 255])

    if draw_text:
        health = int(entity.health)
        mana = int(entity.mana)
        max_health = int(entity.max_health)
        max_mana = int(entity.max_mana)
        rl.DrawText(f'Health: {health}/{max_health}'.encode(),
            x+8, y+2, 20, [255, 255, 255, 255])
        rl.DrawText(f'Mana: {mana}/{max_mana}'.encode(),
            x+8, y+2 - height - 2, 20, [255, 255, 255, 255])

        #rl.DrawRectangle(x, y - 2*height - 4, int(width*mana_bar), height, [255, 255, 0, 255])
        rl.DrawText(f'Experience: {entity.xp}'.encode(),
            x+8, y - 2*height - 4, 20, [255, 255, 255, 255])

    elif entity.entity_type == 0:
        rl.DrawText(f'Level: {entity.level}'.encode(),
            x+4, y -2*height - 12, 12, [255, 255, 255, 255])
