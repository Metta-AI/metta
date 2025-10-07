"""Interactive loop handler for miniscope."""

import select
import sys
import termios
import time
import tty
from typing import Callable, Dict, Optional

import numpy as np
from rich.console import Console

from mettagrid.core import BoundingBox

from .buffer import build_grid_buffer, compute_bounds
from .help import show_help_screen
from .info_panels import build_agent_info_panel, build_object_info_panel, build_symbols_table

try:
    from cogames.cogs_vs_clips.glyphs import GLYPH_DATA, search_glyphs
except ImportError:
    GLYPH_DATA = None
    search_glyphs = None


def _render_glyph_picker_panel(query: str) -> list[str]:
    """Render the glyph picker as a Rich table.

    Args:
        query: Current search query

    Returns:
        List of strings representing the panel lines
    """
    if GLYPH_DATA is None or search_glyphs is None:
        return []

    from rich import box
    from rich.table import Table

    # Create table with border - match agent info table width (46)
    table = Table(title=f"Glyph: {query}", show_header=False, box=box.ROUNDED, padding=(0, 1), width=46)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Symbol", no_wrap=True)
    table.add_column("Name", style="white")

    # Get matches
    if query:
        # Check if query is numeric - support partial matches
        if query.isdigit():
            # Find all glyphs whose ID starts with the query
            results = []
            for i, glyph in enumerate(GLYPH_DATA):
                if str(i).startswith(query):
                    results.append((i, glyph))
                    if len(results) >= 5:
                        break
        else:
            results = search_glyphs(query)[:5]
    else:
        # Show first 5 glyphs when no query
        results = [(i, GLYPH_DATA[i]) for i in range(min(5, len(GLYPH_DATA)))]

    if results:
        for glyph_id, glyph in results:
            table.add_row(str(glyph_id), glyph.symbol, glyph.name)
    else:
        table.add_row("", "", "(no matches)")

    # Add help text row
    table.add_row("", "", "")
    table.add_row("Enter=OK", "", "Esc=Cancel")

    # Render to string - match agent info console width (46)
    temp_console = Console(width=46, legacy_windows=False)
    with temp_console.capture() as capture:
        temp_console.print(table)

    return capture.get().split("\n")


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
    symbol_map: dict[str, str],
    get_actions_fn: Callable[[np.ndarray, Optional[int], Optional[int | tuple], set[int]], np.ndarray],
    max_steps: Optional[int] = None,
    target_fps: int = 4,
    glyphs: list[str] | None = None,
    resource_names: list[str] | None = None,
) -> Dict[str, any]:
    """Run interactive rendering loop with keyboard controls.

    Args:
        env: MettaGrid environment
        object_type_names: List mapping type IDs to names
        symbol_map: Map from object type names to render symbols
        get_actions_fn: Function that takes (obs, selected_agent, manual_action_direction, manual_agents)
                       and returns actions for all agents
        max_steps: Maximum steps to run (None for unlimited)
        target_fps: Target frames per second when playing
        glyphs: Optional list of glyph symbols for display
        resource_names: List mapping resource IDs to names

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
    # Mode state: 'pan', 'select', 'follow', or 'glyph_picker'
    mode = "follow"  # Start in follow mode to track selected agent
    cursor_row = env.map_height // 2
    cursor_col = env.map_width // 2
    glyph_query = ""  # For glyph picker mode
    manual_agents = set()  # Set of agent IDs in manual mode

    # Compute bounds once
    obs, _ = env.reset()
    bbox = BoundingBox(min_row=0, max_row=env.map_height, min_col=0, max_col=env.map_width)
    grid_objects = env.grid_objects(bbox)
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

        # State for manual actions
        manual_action = None
        should_step = False

        while max_steps is None or step_count < max_steps:
            # Handle keyboard input
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                should_render = True  # Any keypress triggers render

                # Handle glyph picker mode
                if mode == "glyph_picker":
                    if ch == "\n" or ch == "\r":
                        # Enter - confirm selection
                        glyph_id = None
                        if glyph_query.isdigit():
                            glyph_id = int(glyph_query)
                        elif glyph_query:
                            results = search_glyphs(glyph_query)
                            if results:
                                glyph_id = results[0][0]

                        if glyph_id is not None and 0 <= glyph_id < len(GLYPH_DATA):
                            action_name = f"change_glyph_{glyph_id}"
                            if action_name not in env.action_names:
                                print(
                                    f"Warning: glyph action '{action_name}' not available in action space.",
                                    flush=True,
                                )
                            else:
                                manual_action = env.action_names.index(action_name)
                                should_step = True

                        mode = "follow"  # Exit glyph picker
                        should_render = True
                    elif ch == "\x1b":  # Escape
                        mode = "follow"  # Exit without selection
                        should_render = True
                    elif ch == "\x7f" or ch == "\x08":  # Backspace
                        glyph_query = glyph_query[:-1] if glyph_query else ""
                    elif ch.isprintable():
                        glyph_query += ch
                    continue  # Don't process other commands in glyph picker mode

                if ch == " ":
                    paused = not paused
                elif ch == "<" or ch == ",":
                    # Decrease speed (increase frame_delay)
                    frame_delay = min(2.0, frame_delay * 1.5)
                    target_fps = 1.0 / frame_delay
                    should_render = True
                elif ch == ">" or ch == ".":
                    # Increase speed (decrease frame_delay)
                    frame_delay = max(0.01, frame_delay / 1.5)
                    target_fps = 1.0 / frame_delay
                    should_render = True
                elif ch == "q" or ch == "Q":
                    break
                elif ch == "?":
                    # Show help screen
                    help_lines = show_help_screen()
                    print("\033[2J\033[H", end="", flush=True)  # Clear screen
                    for line in help_lines:
                        print(line)
                    # Wait for any key press
                    termios.tcflush(sys.stdin, termios.TCIFLUSH)
                    old_settings = termios.tcgetattr(sys.stdin)
                    try:
                        tty.setraw(sys.stdin.fileno())
                        sys.stdin.read(1)
                    finally:
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                    should_render = True  # Re-render after returning from help
                elif ch == "o" or ch == "O":
                    # Cycle through modes: follow -> pan -> select -> follow
                    if mode == "follow":
                        mode = "pan"
                    elif mode == "pan":
                        mode = "select"
                    else:  # select
                        mode = "follow"
                elif ch == "m" or ch == "M":
                    # Toggle manual mode for selected agent
                    if selected_agent is not None:
                        if selected_agent in manual_agents:
                            manual_agents.remove(selected_agent)
                        else:
                            manual_agents.add(selected_agent)
                        should_render = True
                elif ch == "e" or ch == "E":
                    # Emote - toggle glyph picker mode
                    if mode == "glyph_picker":
                        # Exit glyph picker mode
                        mode = "follow"
                        glyph_query = ""
                        should_render = True
                    elif selected_agent is not None and GLYPH_DATA is not None and "change_glyph" in env.action_names:
                        # Enter glyph picker mode
                        mode = "glyph_picker"
                        glyph_query = ""
                        should_render = True
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
                    manual_action = noop_action_id  # REST/NOOP as a single discrete action
                    should_step = True

            # Step simulation if manual action or auto-play is active
            if should_step or not paused:
                # Manual action or auto-step
                manual_action_to_use = manual_action if should_step else None
                actions = get_actions_fn(obs, selected_agent, manual_action_to_use, manual_agents)
                obs, rewards, dones, truncated, _ = env.step(actions)
                total_rewards += rewards
                step_count += 1
                should_render = True

                # Reset manual action after it's been used
                if should_step:
                    manual_action = None
                    should_step = False

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
                    # Reserve 6 lines for header/footer
                    viewport_height = max(5, terminal_size.lines - 6)
                    # Reserve 46 chars for side panel + 2 for spacing, rest for map
                    # Each grid cell takes 2 characters, so divide by 2
                    side_panel_width = 46
                    spacing = 2
                    available_width = max(10, terminal_size.columns - side_panel_width - spacing)
                    viewport_width = available_width // 2
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
                    bbox_full = BoundingBox(min_row=0, max_row=env.map_height, min_col=0, max_col=env.map_width)
                    grid_objects_full = env.grid_objects(bbox_full)
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
                bbox_view = BoundingBox(
                    min_row=view_min_row, max_row=view_max_row, min_col=view_min_col, max_col=view_max_col
                )
                grid_objects = env.grid_objects(bbox_view)

                # Build display components
                grid_buffer = build_grid_buffer(
                    grid_objects,
                    object_type_names,
                    symbol_map,
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

                agent_panel_height = viewport_height // 2
                agent_table = build_agent_info_panel(
                    grid_objects,
                    object_type_names,
                    selected_agent,
                    env.resource_names,
                    agent_panel_height,
                    total_rewards,
                    glyphs,
                    symbol_map,
                    manual_agents,
                )

                # Convert Table to list of strings
                temp_console = Console(width=46, legacy_windows=False)
                with temp_console.capture() as capture:
                    temp_console.print(agent_table)
                info_panel = capture.get().split("\n")

                # Add glyph picker below agent info if in that mode
                if mode == "glyph_picker":
                    glyph_panel = _render_glyph_picker_panel(glyph_query)
                    side_panel = info_panel + glyph_panel

                # Build object info panel if in select mode
                elif mode == "select":
                    object_panel_height = viewport_height - agent_panel_height
                    object_table = build_object_info_panel(
                        grid_objects, object_type_names, cursor_row, cursor_col, object_panel_height, resource_names
                    )
                    # Convert Table to list of strings
                    with temp_console.capture() as capture:
                        temp_console.print(object_table)
                    object_panel = capture.get().split("\n")
                    # Combine both panels vertically
                    side_panel = info_panel + object_panel

                # Show symbols table in follow/pan modes (not in glyph_picker or select)
                else:
                    # Add symbols table below agent info
                    symbols_table = build_symbols_table(object_type_names, symbol_map, max_rows=8)
                    with temp_console.capture() as capture:
                        temp_console.print(symbols_table)
                    symbols_panel = capture.get().split("\n")
                    side_panel = info_panel + symbols_panel

                # Combine grid and info panel side by side
                grid_lines = grid_buffer.split("\n")

                # Determine the max height needed - use viewport_height as minimum
                max_rows = max(len(grid_lines), len(side_panel), viewport_height)

                # Pad both grid and side panel to the same height
                padded_grid_lines = list(grid_lines)
                while len(padded_grid_lines) < max_rows:
                    # Create empty line with viewport_width worth of spaces
                    padded_grid_lines.append(" " * (viewport_width * 2))

                padded_side_panel = list(side_panel)
                while len(padded_side_panel) < max_rows:
                    padded_side_panel.append(" " * 46)

                # Combine side by side
                combined_lines = []
                for i in range(max_rows):
                    grid_line = padded_grid_lines[i]
                    info_line = padded_side_panel[i]
                    combined_lines.append(f"{grid_line}  {info_line}")

                combined_buffer = "\n".join(combined_lines)

                selected_text = f"Agent {selected_agent}" if selected_agent is not None else "AI Control"
                mode_text = mode.upper()
                status = "PAUSED" if paused else "PLAYING"
                sps = f"{target_fps:.1f}" if target_fps < 10 else f"{int(target_fps)}"

                output = (
                    f"\033[2J\033[H"  # Clear and home
                    f"Step {step_count} | Reward: {float(sum(total_rewards)):.2f} | SPS: {sps}\n"
                    f"Status: {status} | Mode: {mode_text} | "
                    f"Camera: ({camera_row},{camera_col}) | Control: {selected_text}\n"
                    f"\n{combined_buffer}\n"
                    f"\n?=Help | SPACE=Play/Pause | <>=Speed | O=Cycle Mode | "
                    f"IJKL=Pan/Select | []=Agent | M=Manual | WASD=Move | E=Emote | R=Rest | Q=Quit"
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
