"""Interactive loop handler for miniscope."""

import select
import sys
import termios
import time
import tty
from typing import Callable, Dict, Optional

import numpy as np

from .buffer import build_grid_buffer, compute_bounds
from .info_panels import build_agent_info_panel, build_object_info_panel


def _adjust_camera_to_keep_in_view(
    camera_row: int,
    camera_col: int,
    target_row: int,
    target_col: int,
    viewport_height: int,
    viewport_width: int,
    map_height: int,
    map_width: int,
    margin: int = 3,
) -> tuple[int, int]:
    """Adjust camera position to keep target in view with margin from edges.

    Args:
        camera_row: Current camera row position
        camera_col: Current camera column position
        target_row: Target row to keep in view
        target_col: Target column to keep in view
        viewport_height: Height of the viewport
        viewport_width: Width of the viewport
        map_height: Total map height
        map_width: Total map width
        margin: Minimum distance from viewport edge before panning

    Returns:
        Tuple of (new_camera_row, new_camera_col)
    """
    # Calculate viewport bounds
    view_min_row = max(0, camera_row - viewport_height // 2)
    view_min_col = max(0, camera_col - viewport_width // 2)

    # Calculate target position relative to viewport
    target_rel_row = target_row - view_min_row
    target_rel_col = target_col - view_min_col

    new_camera_row = camera_row
    new_camera_col = camera_col

    # Check if target is too close to top edge
    if target_rel_row < margin:
        new_camera_row = max(viewport_height // 2, target_row + margin - viewport_height // 2)

    # Check if target is too close to bottom edge
    if target_rel_row >= viewport_height - margin:
        new_camera_row = min(map_height - viewport_height // 2, target_row - margin + viewport_height // 2)

    # Check if target is too close to left edge
    if target_rel_col < margin:
        new_camera_col = max(viewport_width // 2, target_col + margin - viewport_width // 2)

    # Check if target is too close to right edge
    if target_rel_col >= viewport_width - margin:
        new_camera_col = min(map_width - viewport_width // 2, target_col - margin + viewport_width // 2)

    return new_camera_row, new_camera_col


def run_interactive_loop(
    env,
    object_type_names: list[str],
    get_actions_fn: Callable[[np.ndarray, Optional[int], Optional[int | tuple]], np.ndarray],
    max_steps: Optional[int] = None,
    target_fps: int = 4,
) -> Dict[str, any]:
    """Run interactive rendering loop with keyboard controls.

    Args:
        env: MettaGrid environment
        object_type_names: List mapping type IDs to names
        get_actions_fn: Function that takes (obs, selected_agent, manual_action_direction)
                       and returns actions for all agents
        max_steps: Maximum steps to run (None for unlimited)
        target_fps: Target frames per second when playing

    Returns:
        Dict with final statistics
    """
    # State
    paused = False
    first_frame_rendered = False
    camera_row = env.map_height // 2
    camera_col = env.map_width // 2
    selected_agent: Optional[int] = 0  # Start with first agent selected
    step_count = 0
    total_rewards = np.zeros(env.num_agents)
    frame_delay = 1.0 / target_fps
    last_frame_time = time.time()
    # Mode state: 'pan', 'select', or 'follow'
    mode = "follow"  # Start in follow mode to track selected agent
    cursor_row = env.map_height // 2
    cursor_col = env.map_width // 2

    # Compute bounds once
    obs, _ = env.reset()
    grid_objects = env.grid_objects(0, env.map_height, 0, env.map_width)
    min_row, min_col, height, width = compute_bounds(grid_objects, object_type_names)

    # Clamp cursor to actual playable bounds
    cursor_row = max(min_row, min(min_row + height - 1, cursor_row))
    cursor_col = max(min_col, min(min_col + width - 1, cursor_col))
    camera_row = max(min_row, min(min_row + height - 1, camera_row))
    camera_col = max(min_col, min(min_col + width - 1, camera_col))

    # Setup terminal for non-blocking input
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    try:
        # Render first frame immediately
        first_frame_rendered = False
        should_render = True  # Trigger initial render
        dones = np.zeros(env.num_agents, dtype=bool)
        truncated = np.zeros(env.num_agents, dtype=bool)

        while max_steps is None or step_count < max_steps:
            # Handle keyboard input
            manual_action = None
            should_step = False

            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                should_render = True  # Any keypress triggers render

                if ch == " ":
                    paused = not paused
                elif ch == "q" or ch == "Q":
                    break
                elif ch == "o" or ch == "O":
                    # Cycle through modes: follow -> pan -> select -> follow
                    if mode == "follow":
                        mode = "pan"
                    elif mode == "pan":
                        mode = "select"
                    else:  # select
                        mode = "follow"
                elif ch == "i":
                    # Lowercase i: move 1 space up
                    if mode == "select":
                        cursor_row = max(min_row, cursor_row - 1)
                    else:
                        camera_row = max(min_row, camera_row - 1)
                        if mode == "follow":
                            mode = "pan"  # Switch to pan mode on manual movement
                elif ch == "I":
                    # Uppercase I (Shift+i): move 10 spaces up
                    if mode == "select":
                        cursor_row = max(min_row, cursor_row - 10)
                    else:
                        camera_row = max(min_row, camera_row - 10)
                        if mode == "follow":
                            mode = "pan"  # Switch to pan mode on manual movement
                elif ch == "k":
                    # Lowercase k: move 1 space down
                    if mode == "select":
                        cursor_row = min(min_row + height - 1, cursor_row + 1)
                    else:
                        camera_row = min(min_row + height - 1, camera_row + 1)
                        if mode == "follow":
                            mode = "pan"  # Switch to pan mode on manual movement
                elif ch == "K":
                    # Uppercase K (Shift+k): move 10 spaces down
                    if mode == "select":
                        cursor_row = min(min_row + height - 1, cursor_row + 10)
                    else:
                        camera_row = min(min_row + height - 1, camera_row + 10)
                        if mode == "follow":
                            mode = "pan"  # Switch to pan mode on manual movement
                elif ch == "j":
                    # Lowercase j: move 1 space left
                    if mode == "select":
                        cursor_col = max(min_col, cursor_col - 1)
                    else:
                        camera_col = max(min_col, camera_col - 1)
                        if mode == "follow":
                            mode = "pan"  # Switch to pan mode on manual movement
                elif ch == "J":
                    # Uppercase J (Shift+j): move 10 spaces left
                    if mode == "select":
                        cursor_col = max(min_col, cursor_col - 10)
                    else:
                        camera_col = max(min_col, camera_col - 10)
                        if mode == "follow":
                            mode = "pan"  # Switch to pan mode on manual movement
                elif ch == "l":
                    # Lowercase l: move 1 space right
                    if mode == "select":
                        cursor_col = min(min_col + width - 1, cursor_col + 1)
                    else:
                        camera_col = min(min_col + width - 1, camera_col + 1)
                        if mode == "follow":
                            mode = "pan"  # Switch to pan mode on manual movement
                elif ch == "L":
                    # Uppercase L (Shift+l): move 10 spaces right
                    if mode == "select":
                        cursor_col = min(min_col + width - 1, cursor_col + 10)
                    else:
                        camera_col = min(min_col + width - 1, camera_col + 10)
                        if mode == "follow":
                            mode = "pan"  # Switch to pan mode on manual movement
                elif ch == "[":
                    selected_agent = 0 if selected_agent is None else (selected_agent - 1) % env.num_agents
                elif ch == "]":
                    selected_agent = 0 if selected_agent is None else (selected_agent + 1) % env.num_agents
                elif ch in ["w", "W"] and selected_agent is not None:
                    manual_action = 0  # NORTH
                    should_step = True  # Movement always steps
                elif ch in ["s", "S"] and selected_agent is not None:
                    manual_action = 1  # SOUTH
                    should_step = True
                elif ch in ["a", "A"] and selected_agent is not None:
                    manual_action = 2  # WEST
                    should_step = True
                elif ch in ["d", "D"] and selected_agent is not None:
                    manual_action = 3  # EAST
                    should_step = True
                elif ch in ["r", "R"] and selected_agent is not None:
                    # Get noop action ID from environment
                    noop_action_id = env.action_names.index("noop") if "noop" in env.action_names else 0
                    manual_action = (noop_action_id, 0)  # REST/NOOP with arg=0
                    should_step = True

            # Step simulation if manual action or auto-play is active
            if should_step or not paused:
                # Manual action or auto-step
                actions = get_actions_fn(obs, selected_agent, manual_action if should_step else None)
                obs, rewards, dones, truncated, _ = env.step(actions)
                total_rewards += rewards
                step_count += 1
                should_render = True

                # Frame rate limiting only for auto-play
                if not paused:
                    current_time = time.time()
                    elapsed = current_time - last_frame_time
                    if elapsed < frame_delay:
                        time.sleep(frame_delay - elapsed)
                    last_frame_time = time.time()

            # Render if needed
            if should_render:
                # Get console size for viewport
                try:
                    import shutil

                    terminal_size = shutil.get_terminal_size()
                    viewport_height = max(5, terminal_size.lines - 6)
                    # Reserve space for info panel (23 chars wide)
                    viewport_width = max(5, (terminal_size.columns - 25) // 2 - 2)
                except Exception:
                    viewport_height = 20
                    viewport_width = 40

                # Auto-pan camera to keep cursor in view (in select mode)
                if mode == "select":
                    camera_row, camera_col = _adjust_camera_to_keep_in_view(
                        camera_row,
                        camera_col,
                        cursor_row,
                        cursor_col,
                        viewport_height,
                        viewport_width,
                        env.map_height,
                        env.map_width,
                    )

                # Auto-pan camera to keep selected agent in view (in follow mode)
                if mode == "follow" and selected_agent is not None:
                    # Find the selected agent's position
                    grid_objects_full = env.grid_objects(0, env.map_height, 0, env.map_width)
                    for obj in grid_objects_full.values():
                        if obj.get("agent_id") == selected_agent:
                            agent_row = obj["r"]
                            agent_col = obj["c"]
                            camera_row, camera_col = _adjust_camera_to_keep_in_view(
                                camera_row,
                                camera_col,
                                agent_row,
                                agent_col,
                                viewport_height,
                                viewport_width,
                                env.map_height,
                                env.map_width,
                            )
                            break

                # Calculate viewport bounds for grid_objects filtering (optimization for large maps)
                view_min_row = max(0, camera_row - viewport_height // 2)
                view_max_row = min(env.map_height, view_min_row + viewport_height)
                view_min_col = max(0, camera_col - viewport_width // 2)
                view_max_col = min(env.map_width, view_min_col + viewport_width)

                # Get objects within viewport bounds (performance optimization)
                grid_objects = env.grid_objects(view_min_row, view_max_row, view_min_col, view_max_col)

                # Build display components
                grid_buffer = build_grid_buffer(
                    grid_objects,
                    object_type_names,
                    min_row,
                    min_col,
                    height,
                    width,
                    camera_row,
                    camera_col,
                    viewport_height,
                    viewport_width,
                    cursor_row if mode == "select" else None,
                    cursor_col if mode == "select" else None,
                )

                # Build agent info panel
                agent_panel_height = viewport_height // 2
                info_panel = build_agent_info_panel(
                    grid_objects,
                    object_type_names,
                    selected_agent,
                    env.resource_names,
                    agent_panel_height,
                    total_rewards,
                )

                # Build object info panel if in select mode
                if mode == "select":
                    object_panel_height = viewport_height - agent_panel_height
                    object_panel = build_object_info_panel(
                        grid_objects, object_type_names, cursor_row, cursor_col, object_panel_height
                    )
                    # Combine both panels vertically
                    side_panel = info_panel + object_panel
                else:
                    side_panel = info_panel + ["                       "] * (viewport_height - agent_panel_height)

                # Combine grid and info panel side by side
                grid_lines = grid_buffer.split("\n")
                combined_lines = []
                for i in range(max(len(grid_lines), len(side_panel))):
                    grid_line = grid_lines[i] if i < len(grid_lines) else " " * (viewport_width * 2)
                    info_line = side_panel[i] if i < len(side_panel) else " " * 23
                    combined_lines.append(f"{grid_line}  {info_line}")

                combined_buffer = "\n".join(combined_lines)

                selected_text = f"Agent {selected_agent}" if selected_agent is not None else "AI Control"
                mode_text = mode.upper()
                status = "PAUSED" if paused else "PLAYING"

                output = (
                    f"\033[2J\033[H"  # Clear and home
                    f"Step {step_count} | Reward: {float(sum(total_rewards)):.2f}\n"
                    f"Status: {status} | Mode: {mode_text} | "
                    f"Camera: ({camera_row},{camera_col}) | Control: {selected_text}\n"
                    f"\n{combined_buffer}\n"
                    f"\nSPACE=Play/Pause | O=Cycle Mode (Follow/Pan/Select) | "
                    f"IJKL=Pan/Select (Shift+IJKL=10x) | []=Agent | WASD=Move | R=Rest | Q=Quit"
                )
                print(output, end="", flush=True)

                # Pause after first frame
                if not first_frame_rendered:
                    first_frame_rendered = True
                    paused = True

                # Reset render flag after rendering
                should_render = False

            # Sleep a bit if paused
            if paused:
                time.sleep(0.1)

            # Check for done
            if all(dones) or all(truncated):
                break

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("\033[?25h")  # Show cursor

    return {"steps": step_count, "total_rewards": total_rewards}
