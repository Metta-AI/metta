import pyray as ray
from raylib import rl
import math

_SIN_45 = 2 ** 0.5 / 2

class CameraController:
    def __init__(self, camera):
        self.camera = camera

        self.speed = 1000.0
        self.zoom_speed = 5.0

        self.zoom_lower_bound = 0.5
        self.zoom_upper_bound = 2.0

    def update(self, delta_time: float, viewport_width: float, viewport_height: float):
        effective_speed = (self.speed * delta_time) / self.camera.zoom

        move_up = rl.IsKeyDown(rl.KEY_UP)
        move_down = rl.IsKeyDown(rl.KEY_DOWN)
        move_left = rl.IsKeyDown(rl.KEY_LEFT)
        move_right = rl.IsKeyDown(rl.KEY_RIGHT)

        if move_up != move_down and move_left != move_right:
            effective_speed *= _SIN_45

        if move_up:
            self.camera.target.y -= effective_speed
        if move_down:
            self.camera.target.y += effective_speed
        if move_left:
            self.camera.target.x -= effective_speed
        if move_right:
            self.camera.target.x += effective_speed

        scroll = rl.GetMouseWheelMove()
        mouse_world_x, mouse_world_y = self.get_world_mouse_position()

        if scroll != 0:
            self.camera.zoom *=  math.exp(scroll * self.zoom_speed * delta_time)
            self.camera.zoom = clamp(self.camera.zoom, self.zoom_lower_bound, self.zoom_upper_bound)

        new_mouse_x, new_mouse_y = self.get_world_mouse_position()
        self.camera.target.x += (mouse_world_x - new_mouse_x)
        self.camera.target.y += (mouse_world_y - new_mouse_y)

        self.camera.offset.x = viewport_width / 2.0
        self.camera.offset.y = viewport_height / 2.0


    def get_world_mouse_position(self) -> tuple[float, float]:
        screen_pos = rl.GetMousePosition();
        world_pos = rl.GetScreenToWorld2D(screen_pos, self.camera)
        return world_pos.x, world_pos.y


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))
