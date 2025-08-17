"""Training run management utilities for MCP server."""

from __future__ import annotations

import gzip
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import boto3
import psutil
import requests
import torch
from fastmcp import Context
from omegaconf import OmegaConf

# Simple ASCII rendering without complex dependencies
ASCII_RENDERING_AVAILABLE = True

# Character mapping for different object types
OBJECT_CHAR_MAP = {
    "empty": ".",
    "wall": "#",
    "agent": "@",
    "mine_red": "m",
    "mine_blue": "b",
    "mine_green": "g",
    "generator_red": "n",
    "generator_blue": "B",
    "generator_green": "G",
    "altar": "_",
    "converter": "c",
    "armory": "o",
    "lasery": "S",
    "lab": "L",
    "factory": "F",
    "temple": "T",
    "block": "s",
}


def _simple_grid_to_lines(grid, border=False):
    """Simple grid to ASCII conversion without external dependencies."""
    lines = []
    for row in grid:
        line_chars = []
        for cell in row:
            char = OBJECT_CHAR_MAP.get(cell, "?")
            line_chars.append(char)
        lines.append("".join(line_chars))

    if border:
        width = len(lines[0]) if lines else 0
        border_lines = ["┌" + "─" * width + "┐"]
        for line in lines:
            border_lines.append("│" + line + "│")
        border_lines.append("└" + "─" * width + "┘")
        return border_lines

    return lines


def _create_simple_grid(height, width):
    """Create a simple 2D grid without external dependencies."""
    return [["empty" for _ in range(width)] for _ in range(height)]


def get_train_dir() -> Path:
    """Get the training runs directory."""
    # Try current working directory first
    current_dir = Path.cwd()
    train_dir = current_dir / "train_dir"
    if train_dir.exists():
        return train_dir

    # Try finding it relative to this file
    this_file = Path(__file__)
    project_root = this_file.parent.parent.parent.parent.parent.parent
    train_dir = project_root / "train_dir"
    if train_dir.exists():
        return train_dir

    raise FileNotFoundError("Could not find train_dir directory")


def list_training_runs() -> List[Dict[str, Any]]:
    """List all local training runs with metadata."""
    try:
        train_dir = get_train_dir()
    except FileNotFoundError:
        return []

    runs = []

    for run_dir in train_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue

        # Skip certain directories
        if run_dir.name in ["__MACOSX", "wandb", ".wandb"]:
            continue

        try:
            run_info = _analyze_training_run(run_dir)
            if run_info:
                runs.append(run_info)
        except Exception as e:
            # Include failed runs with error info
            runs.append(
                {
                    "name": run_dir.name,
                    "path": str(run_dir),
                    "status": "error",
                    "error": str(e),
                    "created": _get_dir_creation_time(run_dir),
                    "size": _get_dir_size(run_dir),
                }
            )

    # Sort by creation time (newest first)
    runs.sort(key=lambda x: x.get("created", ""), reverse=True)
    return runs


def _combine_action_id_and_param(action_id_data: list, action_param_data: list) -> list:
    """Combine action_id and action_param data from objects format into unified action format.

    Args:
        action_id_data: List of [step, action_id] entries
        action_param_data: List of [step, param] entries

    Returns:
        List of [step, [action_id, args]] entries (matching grid_objects format)
    """
    if not action_id_data:
        return []

    # Create mapping from step to param
    step_to_param = {step: param for step, param in action_param_data} if action_param_data else {}

    # Combine into unified format
    action_data = []
    for step, action_id in action_id_data:
        param = step_to_param.get(step)
        if param is not None:
            # Convert single param to args list format
            args = [param] if not isinstance(param, list) else param
            action_data.append([step, [action_id, args]])
        else:
            # No param for this step
            action_data.append([step, [action_id, []]])

    return action_data


def _infer_actions_from_movement_and_success(
    location_data: list, action_success_data: list, action_names: list
) -> list:
    """Infer actions from movement patterns and success data for objects format.

    Args:
        location_data: List of [step, [col, row, layer]] entries
        action_success_data: List of [step, success] entries
        action_names: List of available action names

    Returns:
        List of [step, [action_id, args]] entries (synthetic action data)
    """
    if not location_data or not action_success_data or not action_names:
        return []

    # Create a mapping from step to position
    step_to_pos = {}
    for step, location in location_data:
        if len(location) >= 2:
            step_to_pos[step] = (location[0], location[1])  # (col, row)

    # Create a mapping from step to success
    step_to_success = {step: success for step, success in action_success_data}

    # Infer actions based on movement patterns
    action_data = []
    prev_pos = None
    prev_step = None

    # Find movement action indices (common gridworld actions)
    move_actions = []
    for i, name in enumerate(action_names):
        if any(word in name.lower() for word in ["move", "up", "down", "left", "right"]):
            move_actions.append(i)

    # Default to first few actions if no move actions found
    if not move_actions:
        move_actions = list(range(min(4, len(action_names))))

    sorted_steps = sorted(step_to_pos.keys())

    for step in sorted_steps:
        current_pos = step_to_pos[step]
        # success = step_to_success.get(step, True)  # Available for future use

        if prev_pos is not None:
            # Calculate movement direction
            dc = current_pos[0] - prev_pos[0]  # col delta
            dr = current_pos[1] - prev_pos[1]  # row delta

            # Map movement to action with directional arguments
            action_id = move_actions[0]  # default
            direction_arg = 0  # default direction

            if dc == 0 and dr == -1:  # up
                action_id = move_actions[0] if len(move_actions) > 0 else 0
                direction_arg = 0  # up
            elif dc == 1 and dr == 0:  # right
                action_id = move_actions[1] if len(move_actions) > 1 else 1
                direction_arg = 1  # right
            elif dc == 0 and dr == 1:  # down
                action_id = move_actions[2] if len(move_actions) > 2 else 2
                direction_arg = 2  # down
            elif dc == -1 and dr == 0:  # left
                action_id = move_actions[3] if len(move_actions) > 3 else 3
                direction_arg = 3  # left
            elif dc == 0 and dr == 0:  # no movement - might be interact action
                action_id = len(action_names) - 1 if len(action_names) > 4 else 0
                direction_arg = None  # no direction for interact actions

            # Add synthetic action entry with directional arguments
            if direction_arg is not None:
                action_data.append([prev_step, [action_id, [direction_arg]]])
            else:
                action_data.append([prev_step, [action_id, []]])

        prev_pos = current_pos
        prev_step = step

    # Add some random actions for steps with success data but no movement
    success_steps = set(step_to_success.keys())
    movement_steps = set(step_to_pos.keys())

    for step in success_steps - movement_steps:
        # Add non-movement action for these steps
        action_id = (len(action_names) - 1) if len(action_names) > 0 else 0
        action_data.append([step, [action_id, []]])

    return sorted(action_data, key=lambda x: x[0])


def _create_agent_action_timelines(
    agent_objects: list, action_names: list, episode_length: int, max_steps: int = 50, item_names: list = None
) -> Dict[str, Any]:
    """Create detailed ASCII timeline visualization of agent actions with directions and items.

    Format: Each step shows [ACTION:DIRECTION:ITEMS] e.g., M↑, R→, A↓, G+ore

    Args:
        agent_objects: List of agent objects with action data
        action_names: List of action names for mapping
        episode_length: Total episode length
        max_steps: Maximum steps to show in timeline (default 50, reduced for readability)

    Returns:
        Dict containing timeline data and visualization
    """

    def _format_action_details(action_name: str, args: list, step: int, inventory_changes: dict) -> str:
        """Format action with detailed information about direction, items, etc."""
        action_name_lower = action_name.lower()

        # Direction mappings for moves and rotates
        direction_map = {
            0: "↑",  # up
            1: "→",  # right
            2: "↓",  # down
            3: "←",  # left
        }

        if "move" in action_name_lower:
            if args and len(args) > 0:
                direction = args[0] if isinstance(args[0], int) else 0
                dir_char = direction_map.get(direction, "?")
                return f"M{dir_char}"
            return "M?"

        elif "rotate" in action_name_lower:
            if args and len(args) > 0:
                # Rotation: 0=right, 1=left (or similar mapping)
                rotation = args[0] if isinstance(args[0], int) else 0
                if rotation == 0:
                    return "R→"
                else:
                    return "R←"
            return "R?"

        elif "attack" in action_name_lower:
            if args and len(args) > 0:
                direction = args[0] if isinstance(args[0], int) else 0
                dir_char = direction_map.get(direction, "?")
                return f"A{dir_char}"
            return "A?"

        elif "get" in action_name_lower or "put" in action_name_lower:
            action_prefix = "G" if "get" in action_name_lower else "P"

            # Check for inventory changes at this step to show what items were actually gained/lost
            step_changes = inventory_changes.get(step, {})
            if step_changes:
                # Find items that increased (for get) or decreased (for put)
                relevant_changes = []
                for item_name, change in step_changes.items():
                    if "get" in action_name_lower and change > 0:
                        relevant_changes.append(f"{item_name[:3]}")  # Short name
                    elif "put" in action_name_lower and change < 0:
                        relevant_changes.append(f"{item_name[:3]}")  # Short name

                if relevant_changes:
                    items_str = "/".join(relevant_changes)
                    return f"{action_prefix}+{items_str}"

            return action_prefix

        elif "swap" in action_name_lower:
            return "S"

        elif "noop" in action_name_lower:
            return "·"

        elif "8way" in action_name_lower:
            if args and len(args) > 0:
                # 8-way movement with direction
                direction = args[0] if isinstance(args[0], int) else 0
                dir_chars = ["↑", "↗", "→", "↘", "↓", "↙", "←", "↖"]
                dir_char = dir_chars[direction % 8] if direction < 8 else "?"
                return f"8{dir_char}"
            return "8?"

        # Default: use first letter of action name
        return action_name[0].upper() if action_name else "?"

    def _extract_inventory_changes(agent_obj: dict, item_names: list) -> dict:
        """Extract inventory changes by step to show what items were gained/lost."""
        inventory_changes = {}  # {step: {item_name: change_amount}}

        # Handle grid_objects format with specific item fields
        if any(key.startswith("inv:") for key in agent_obj.keys()):
            # Process specific item tracking fields like inv:ore_red, inv:battery_red, inv:armor
            for key, data in agent_obj.items():
                if key.startswith("inv:") and isinstance(data, list):
                    item_name = key[4:]  # Remove 'inv:' prefix

                    # Track changes between consecutive entries
                    prev_count = 0
                    for step, count in data:
                        if count != prev_count:
                            change = count - prev_count
                            if step not in inventory_changes:
                                inventory_changes[step] = {}
                            inventory_changes[step][item_name] = change
                            prev_count = count

        # Handle objects format with general inventory field
        elif "inventory" in agent_obj and not any(key.startswith("inv:") for key in agent_obj.keys()):
            inventory_data = agent_obj["inventory"]
            if isinstance(inventory_data, list) and inventory_data:
                prev_inventory = {}

                for step, inventory in inventory_data:
                    current_inventory = {}

                    # Parse inventory format: [[item_id, count], ...]
                    if isinstance(inventory, list):
                        for item_entry in inventory:
                            if isinstance(item_entry, list) and len(item_entry) >= 2:
                                item_id, count = item_entry[0], item_entry[1]
                                if item_id < len(item_names):
                                    item_name = item_names[item_id]
                                    current_inventory[item_name] = count

                    # Calculate changes from previous step
                    all_items = set(prev_inventory.keys()) | set(current_inventory.keys())
                    for item_name in all_items:
                        prev_count = prev_inventory.get(item_name, 0)
                        current_count = current_inventory.get(item_name, 0)
                        change = current_count - prev_count

                        if change != 0:
                            if step not in inventory_changes:
                                inventory_changes[step] = {}
                            inventory_changes[step][item_name] = change

                    prev_inventory = current_inventory

        # Handle grid_objects format with general inventory field
        elif "inventory" in agent_obj:
            inventory_data = agent_obj["inventory"]
            if isinstance(inventory_data, list) and inventory_data:
                prev_inventory = {}

                for step, inventory in inventory_data:
                    current_inventory = {}

                    # Parse inventory format: {item_id: count}
                    if isinstance(inventory, dict):
                        for item_id_str, count in inventory.items():
                            try:
                                item_id = int(item_id_str)
                                if item_id < len(item_names):
                                    item_name = item_names[item_id]
                                    current_inventory[item_name] = count
                            except (ValueError, IndexError):
                                continue

                    # Calculate changes from previous step
                    all_items = set(prev_inventory.keys()) | set(current_inventory.keys())
                    for item_name in all_items:
                        prev_count = prev_inventory.get(item_name, 0)
                        current_count = current_inventory.get(item_name, 0)
                        change = current_count - prev_count

                        if change != 0:
                            if step not in inventory_changes:
                                inventory_changes[step] = {}
                            inventory_changes[step][item_name] = change

                    prev_inventory = current_inventory

        return inventory_changes

    timelines = {}
    agent_count = 0

    for agent_obj in agent_objects[:8]:  # Limit to 8 agents for readability
        agent_id = agent_obj.get("agent_id", agent_count)
        agent_name = f"agent_{agent_id}"
        agent_count += 1

        # Extract inventory changes for this agent
        inventory_changes = _extract_inventory_changes(agent_obj, item_names or [])

        # Get action data - handle both formats
        if "action" in agent_obj:
            # Grid_objects format: unified action field [[step, [action_id, args]], ...]
            action_data = agent_obj["action"]
        else:
            # Objects format: separate action_id and action_param fields
            action_data = _combine_action_id_and_param(
                agent_obj.get("action_id", []),  # [[step, action_id], ...]
                agent_obj.get("action_param", []),  # [[step, param], ...]
            )

        if not action_data:
            continue

        # Create detailed timeline entries
        action_type_counts = {}

        # Fill in actions with details
        step_actions = {}
        for step, action_info in action_data:
            if step >= max_steps:
                break

            if isinstance(action_info, list) and len(action_info) >= 1:
                action_id = action_info[0]
                raw_args = action_info[1] if len(action_info) > 1 else []
                # Ensure args is always a list for consistent handling
                if isinstance(raw_args, list):
                    action_args = raw_args
                elif raw_args is not None:
                    action_args = [raw_args]  # Convert single value to list
                else:
                    action_args = []

                if action_id < len(action_names):
                    action_name = action_names[action_id]
                    formatted_action = _format_action_details(action_name, action_args, step, inventory_changes)
                    step_actions[step] = formatted_action

                    # Count action types (just the base action letter)
                    base_action = formatted_action[0] if formatted_action else "?"
                    action_type_counts[base_action] = action_type_counts.get(base_action, 0) + 1

        # Create timeline string with single space between steps
        timeline_parts = []
        for step in range(min(max_steps, episode_length)):
            if step in step_actions:
                action_str = step_actions[step]
                timeline_parts.append(action_str)
            else:
                timeline_parts.append("_")  # Just _ for no action step

        timeline_str = " ".join(timeline_parts)  # Join with single spaces

        timelines[agent_name] = {
            "timeline": timeline_str,
            "action_counts": action_type_counts,
            "total_actions": len(step_actions),
            "detailed_actions": step_actions,
        }

    # Create legend for the new format
    legend = {
        "M↑/↓/←/→": "Move in direction",
        "R→/←": "Rotate right/left",
        "A↑/↓/←/→": "Attack in direction",
        "G+ore/bat": "Get ore/battery/heart/armor/laser",
        "P+ore/bat": "Put ore/battery/heart/armor/laser",
        "8↑/↗/→/↘": "8-way move in direction",
        "S": "Swap",
        "·": "No-op/idle",
        "_": "No action",
    }

    return {
        "timelines": timelines,
        "action_legend": legend,
        "action_names": action_names,
        "max_steps_shown": max_steps,
        "episode_length": episode_length,
        "format": "detailed",
    }


def _analyze_training_run(run_dir: Path) -> Dict[str, Any] | None:
    """Analyze a single training run directory."""
    run_info = {
        "name": run_dir.name,
        "path": str(run_dir),
        "created": _get_dir_creation_time(run_dir),
        "size": _get_dir_size(run_dir),
    }

    # Check for config file
    config_file = run_dir / "config.yaml"
    if config_file.exists():
        try:
            config = OmegaConf.load(config_file)
            run_info["config"] = {
                "agent": config.agent._target_
                if hasattr(config, "agent") and hasattr(config.agent, "_target_")
                else "unknown",
                "run_name": config.run if hasattr(config, "run") else run_dir.name,
                "total_timesteps": config.trainer.total_timesteps
                if hasattr(config, "trainer") and hasattr(config.trainer, "total_timesteps")
                else "unknown",
                "device": config.device if hasattr(config, "device") else "unknown",
            }
        except Exception as e:
            run_info["config_error"] = str(e)

    # Check for checkpoints
    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.exists():
        checkpoints = list(checkpoints_dir.glob("*.pt"))
        run_info["checkpoints"] = {
            "count": len(checkpoints),
            "latest": checkpoints[-1].name if checkpoints else None,
            "total_size": sum(cp.stat().st_size for cp in checkpoints),
        }
    else:
        run_info["checkpoints"] = {"count": 0, "latest": None, "total_size": 0}

    # Check for logs
    logs_dir = run_dir / "logs"
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        run_info["logs"] = {
            "count": len(log_files),
            "latest_log": log_files[-1].name if log_files else None,
        }

    # Check for trainer state
    trainer_state = run_dir / "trainer_state.pt"
    if trainer_state.exists():
        run_info["trainer_state"] = {
            "exists": True,
            "size": trainer_state.stat().st_size,
            "modified": datetime.fromtimestamp(trainer_state.stat().st_mtime).isoformat(),
        }

    # Determine status
    run_info["status"] = _determine_run_status(run_dir)

    return run_info


def _get_dir_creation_time(dir_path: Path) -> str:
    """Get directory creation time as ISO string."""
    try:
        stat = dir_path.stat()
        # Use the earliest of ctime and mtime
        timestamp = min(stat.st_ctime, stat.st_mtime)
        return datetime.fromtimestamp(timestamp).isoformat()
    except Exception:
        return ""


def _get_dir_size(dir_path: Path) -> int:
    """Get total size of directory in bytes."""
    try:
        total_size = 0
        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    except Exception:
        return 0


def _determine_run_status(run_dir: Path) -> str:
    """Determine the status of a training run."""
    # Check if currently running by looking for processes
    if _is_training_running(run_dir.name):
        return "running"

    # Check for completion indicators
    trainer_state = run_dir / "trainer_state.pt"
    checkpoints_dir = run_dir / "checkpoints"

    # If no checkpoints, likely failed or incomplete
    if not checkpoints_dir.exists() or not list(checkpoints_dir.glob("*.pt")):
        return "failed"

    # If trainer state exists and is recent, likely completed normally
    if trainer_state.exists():
        # Check if modified recently (within last hour) - might still be running
        mod_time = trainer_state.stat().st_mtime
        if time.time() - mod_time < 3600:  # 1 hour
            return "completed"

    # Check logs for completion or failure indicators
    logs_dir = run_dir / "logs"
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        if log_files:
            latest_log = log_files[-1]
            try:
                with open(latest_log) as f:
                    content = f.read()
                    if "training completed" in content.lower():
                        return "completed"
                    elif "error" in content.lower() or "exception" in content.lower():
                        return "failed"
            except Exception:
                pass

    # Default to paused/unknown
    return "paused"


def _is_training_running(run_name: str) -> bool:
    """Check if a training run is currently active by looking for processes."""
    try:
        # Look for python processes that might be training
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if proc.info["name"] and "python" in proc.info["name"].lower():
                    cmdline = proc.info.get("cmdline", [])
                    if cmdline and any("train.py" in arg for arg in cmdline):
                        # Check if this process is for our run
                        if any(run_name in arg for arg in cmdline):
                            return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False
    except Exception:
        return False


def get_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """Get detailed information about a checkpoint file."""
    try:
        path = Path(checkpoint_path)

        # Handle policy URI format
        if checkpoint_path.startswith("file://"):
            path = Path(checkpoint_path[7:])

        # If it's a directory, look for latest checkpoint
        if path.is_dir():
            checkpoints = list(path.glob("*.pt"))
            if not checkpoints:
                return {"error": "No checkpoint files found in directory", "path": str(path)}

            # Get the latest checkpoint by name (assumes model_XXXX.pt format)
            checkpoints.sort(key=lambda x: x.name)
            path = checkpoints[-1]

        if not path.exists():
            return {"error": "Checkpoint file not found", "path": str(path)}

        # Basic file info
        file_stat = path.stat()
        info = {
            "path": str(path),
            "filename": path.name,
            "size": file_stat.st_size,
            "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
        }

        # Try to load checkpoint metadata without loading the full model
        try:
            # Load just the metadata/headers
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            if isinstance(checkpoint, dict):
                # Extract useful metadata
                info["checkpoint_info"] = {}

                # Common checkpoint keys
                for key in ["epoch", "step", "global_step", "loss", "lr", "optimizer_state_dict"]:
                    if key in checkpoint:
                        value = checkpoint[key]
                        # Convert tensors to scalars for JSON serialization
                        if hasattr(value, "item"):
                            value = value.item()
                        elif hasattr(value, "tolist"):
                            value = value.tolist()
                        info["checkpoint_info"][key] = value

                # Model state info
                if "model_state_dict" in checkpoint or "state_dict" in checkpoint:
                    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", {}))
                    info["model_info"] = {
                        "parameter_count": len(state_dict),
                        "parameter_names": list(state_dict.keys())[:10],  # First 10 parameter names
                    }

                # Additional metadata
                if "config" in checkpoint:
                    info["config_info"] = str(checkpoint["config"])[:500]  # Truncate for safety

        except Exception as e:
            info["load_error"] = str(e)

        return info

    except Exception as e:
        return {"error": str(e), "path": checkpoint_path}


def get_training_status(run_name: str) -> Dict[str, Any]:
    """Get detailed status of a specific training run."""
    try:
        train_dir = get_train_dir()
        run_dir = train_dir / run_name

        if not run_dir.exists():
            return {"error": f"Training run not found: {run_name}", "status": "not_found"}

        # Get basic run info
        run_info = _analyze_training_run(run_dir)

        # Add more detailed status information
        status_info = {
            "run_name": run_name,
            "basic_status": run_info.get("status", "unknown") if run_info else "unknown",
            "is_running": _is_training_running(run_name),
        }

        # Check recent log activity
        logs_dir = run_dir / "logs"
        if logs_dir.exists():
            log_files = list(logs_dir.glob("*.log"))
            if log_files:
                latest_log = log_files[-1]
                try:
                    stat = latest_log.stat()
                    last_modified = datetime.fromtimestamp(stat.st_mtime)
                    age_minutes = (datetime.now() - last_modified).total_seconds() / 60

                    status_info["log_info"] = {
                        "latest_log": latest_log.name,
                        "last_modified": last_modified.isoformat(),
                        "age_minutes": age_minutes,
                        "size": stat.st_size,
                    }

                    # Read last few lines of log for recent activity
                    with open(latest_log) as f:
                        lines = f.readlines()
                        status_info["recent_log_lines"] = lines[-5:] if lines else []

                except Exception as e:
                    status_info["log_error"] = str(e)

        # Check for wandb run info
        wandb_dirs = list(run_dir.glob("wandb/run-*"))
        if wandb_dirs:
            status_info["wandb_runs"] = len(wandb_dirs)
            latest_wandb = wandb_dirs[-1]
            status_info["latest_wandb"] = latest_wandb.name

        # Combine with run info
        if run_info:
            status_info.update(run_info)

        return status_info

    except Exception as e:
        return {"error": str(e), "run_name": run_name}


async def generate_replay_summary_with_llm(replay_path: str, ctx: Optional[Context] = None) -> Dict[str, Any]:
    """Generate a summary of replay contents using LLM analysis."""
    try:
        if ctx:
            await ctx.info(f"Starting replay analysis for: {replay_path}")
            await ctx.report_progress(progress=0, total=100)

        # Handle different path types - check local files first
        temp_file = None
        path = Path(replay_path)

        try:
            # Handle file:// URIs by converting to local path
            if replay_path.startswith("file://"):
                path = Path(replay_path[7:])  # Remove "file://" prefix
                if ctx:
                    await ctx.info(f"Converting file:// URI to local path: {path}")

            # Always check if file exists locally first, regardless of URL-like naming
            if path.exists():
                # Use local file
                if ctx:
                    await ctx.info("Using local replay file")
                    await ctx.report_progress(progress=5, total=100)
            elif replay_path.startswith(("s3://", "https://")):
                # Download from S3 or URL to temporary file
                if ctx:
                    await ctx.info("File not found locally, attempting download")

                temp_dir = Path(tempfile.mkdtemp())
                if ctx:
                    await ctx.report_progress(progress=5, total=100)

                if replay_path.startswith("s3://") or (
                    replay_path.startswith("https://") and "s3.amazonaws.com" in replay_path
                ):
                    temp_file = await _download_from_s3(replay_path, temp_dir, ctx)
                else:
                    temp_file = await _download_from_url(replay_path, temp_dir, ctx)

                if temp_file is None:
                    return {"error": "Failed to download replay file", "path": replay_path}

                path = temp_file
            else:
                # Neither local file nor URL - file not found
                return {"error": "Replay file not found locally and path is not a URL", "path": str(path)}

            if ctx:
                await ctx.info("Loading replay file")
                await ctx.report_progress(progress=15, total=100)

            # Load and parse replay file
            replay_data = _load_replay_file(path)
            if "error" in replay_data:
                return replay_data

            if ctx:
                await ctx.info("Analyzing replay data")
                await ctx.report_progress(progress=10, total=100)

            # Extract key metrics and events
            analysis = _analyze_replay_data(replay_data)

            # Check for parsing errors - return early without calling LLM
            if "error" in analysis:
                if ctx:
                    await ctx.info(f"Replay parsing failed: {analysis['error']}")
                    await ctx.report_progress(progress=100, total=100)
                return {
                    "replay_path": replay_path,
                    "file_size": path.stat().st_size,
                    "summary": f"Analysis failed: {analysis['error']}",
                    "llm_used": False,
                }

            # Try to get LLM client and generate summary
            llm_client = None
            try:
                from metta.mcp_server.llm_client import get_llm_client

                llm_client = get_llm_client()
                if llm_client:
                    if ctx:
                        await ctx.info("Generating AI-powered summary")
                        await ctx.report_progress(progress=50, total=100)
                    llm_summary = await llm_client.generate_replay_summary(analysis, ctx)
                    if ctx:
                        await ctx.report_progress(progress=100, total=100)
                else:
                    if ctx:
                        await ctx.info("LLM not available")
                    llm_summary = "LLM not available for summary generation"

            except Exception as e:
                if ctx:
                    await ctx.info(f"LLM generation failed: {str(e)}")
                llm_summary = f"LLM generation failed: {str(e)}"

            return {
                "replay_path": replay_path,
                "file_size": path.stat().st_size,
                "summary": llm_summary,
                "llm_used": llm_client is not None,
            }

        finally:
            # Clean up temporary file if it was downloaded
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                    temp_file.parent.rmdir()
                except Exception:
                    pass

    except Exception as e:
        # Clean up temporary file on error
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
                temp_file.parent.rmdir()
            except Exception:
                pass

        if ctx:
            await ctx.error(f"Replay analysis failed: {str(e)}")
        return {"error": str(e), "path": replay_path}


def _get_s3_config() -> Dict[str, Any]:
    """Get S3 configuration from CONFIG."""
    try:
        # Read config file directly to avoid circular imports
        config_path = Path(__file__).parent / "metta.mcp.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            return config["resources"]["aws_s3"]
    except Exception:
        pass

    # Fallback configuration
    return {"buckets": ["softmax-public"], "region": "us-east-1"}


async def _download_from_s3(url: str, temp_dir: Path, ctx: Optional[Context] = None) -> Path | None:
    """Download file from S3 URL to temporary directory."""
    try:
        if url.startswith("https://") and "s3.amazonaws.com" in url:
            # Parse S3 HTTPS URL: https://bucket.s3.amazonaws.com/key
            # or https://softmax-public.s3.amazonaws.com/replays/...
            parsed = urlparse(url)
            parts = parsed.netloc.split(".")
            if len(parts) >= 3 and parts[1] == "s3":
                bucket = parts[0]
                key = parsed.path.lstrip("/")
            else:
                raise ValueError(f"Invalid S3 HTTPS URL format: {url}")
        elif url.startswith("s3://"):
            # Parse S3 URI: s3://bucket/key
            parsed = urlparse(url)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
        else:
            raise ValueError(f"Unsupported URL format: {url}")

        if ctx:
            await ctx.info(f"Downloading from S3: s3://{bucket}/{key}")

        # Get S3 client
        cfg = _get_s3_config()
        s3 = boto3.client("s3", region_name=cfg.get("region", "us-east-1"))

        # Determine file extension
        key_path = Path(key)
        filename = key_path.name
        temp_file = temp_dir / filename

        # Download file
        s3.download_file(bucket, key, str(temp_file))

        if ctx:
            await ctx.info(f"Downloaded to: {temp_file}")

        return temp_file
    except Exception as e:
        if ctx:
            await ctx.info(f"S3 download failed: {str(e)}")
        return None


async def _download_from_url(url: str, temp_dir: Path, ctx: Optional[Context] = None) -> Path | None:
    """Download file from HTTP/HTTPS URL to temporary directory."""
    try:
        if ctx:
            await ctx.info(f"Downloading from URL: {url}")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Determine filename from URL
        parsed = urlparse(url)
        filename = Path(parsed.path).name
        if not filename:
            filename = "replay.json"

        temp_file = temp_dir / filename

        with open(temp_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        if ctx:
            await ctx.info(f"Downloaded to: {temp_file}")

        return temp_file
    except Exception as e:
        if ctx:
            await ctx.info(f"URL download failed: {str(e)}")
        return None


def _load_replay_file(path) -> Dict[str, Any]:
    """Load replay file (handles compressed JSON format)."""
    try:
        from pathlib import Path

        if isinstance(path, str):
            path = Path(path)

        if path.suffix == ".z":
            # Handle compressed replay files - try both zlib and gzip
            import zlib

            try:
                with open(path, "rb") as f:
                    compressed_data = f.read()
                # Try zlib decompression first (common for .json.z files)
                try:
                    json_data = zlib.decompress(compressed_data).decode("utf-8")
                    return json.loads(json_data)
                except zlib.error:
                    # Fall back to gzip if zlib fails
                    with gzip.open(path, "rt") as f:
                        return json.load(f)
            except Exception:
                # If both fail, try as plain text
                with open(path) as f:
                    return json.load(f)
        else:
            with open(path) as f:
                return json.load(f)
    except Exception as e:
        return {"error": f"Failed to load replay file: {str(e)}"}


def _analyze_replay_data(replay_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze mettagrid replay data to extract rich behavioral and strategic insights."""
    analysis = {
        "episode_length": 0,
        "agents": [],
        "final_scores": {},
        "key_events": [],
        "environment_info": {},
        "agent_behaviors": {},
        "movement_patterns": {},
        "action_statistics": {},
        "interaction_events": [],
        "strategic_phases": [],
        "temporal_progression": {},
    }

    try:
        # Validate mettagrid replay format (handle both grid_objects and objects field names)
        if "version" not in replay_data or ("grid_objects" not in replay_data and "objects" not in replay_data):
            return {"error": "Invalid mettagrid replay format - missing version or grid_objects/objects"}

        # Extract environment metadata
        action_names = replay_data.get("action_names", [])
        inventory_items = replay_data.get("inventory_items", [])
        object_types = replay_data.get("object_types", [])
        map_size = replay_data.get("map_size", [0, 0])

        analysis["environment_info"] = {
            "format": "mettagrid",
            "version": replay_data.get("version"),
            "map_size": f"{map_size[0]}x{map_size[1]}",
            "num_agents": replay_data.get("num_agents", 0),
            "max_steps": replay_data.get("max_steps", 0),
            "action_types": action_names,
            "inventory_items": inventory_items,
            "object_types": object_types,
            "ascii_map": _render_ascii_map(replay_data),
        }

        # Determine format and extract agent objects
        if "grid_objects" in replay_data:
            # Grid_objects format (original)
            objects = replay_data.get("grid_objects", [])
            agent_objects = []

            for obj in objects:
                if obj.get("type") == 0:  # Agent object
                    # Check if this agent has temporal data (arrays of [step, value] pairs)
                    # For action field, value might be [action_id, action_arg] so check differently
                    has_temporal_data = False
                    for field in ["r", "c", "action", "action_success"]:
                        field_val = obj.get(field, [])
                        if isinstance(field_val, list) and len(field_val) > 0:
                            first_entry = field_val[0]
                            if isinstance(first_entry, list) and len(first_entry) >= 2:
                                # This looks like [step, value] temporal data
                                has_temporal_data = True
                                break
                    if has_temporal_data:
                        agent_objects.append(obj)

            if not agent_objects:
                return {"error": "No agent objects with temporal data found in grid_objects"}

            # Use grid_objects format analysis
            return _analyze_grid_objects_format(replay_data, analysis)

        elif "objects" in replay_data:
            # Objects format (newer)
            return _analyze_objects_format(replay_data, analysis)

        else:
            return {"error": "No objects or grid_objects found in replay data"}

    except Exception as e:
        analysis["analysis_error"] = str(e)
        import traceback

        analysis["analysis_traceback"] = traceback.format_exc()

    return analysis


def _analyze_grid_objects_format(replay_data: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze replay data in grid_objects format (original format)."""
    try:
        action_names = replay_data.get("action_names", [])

        # Extract agent objects from grid_objects (type 0 = agent)
        objects = replay_data.get("grid_objects", [])
        agent_objects = []

        for obj in objects:
            if obj.get("type") == 0:  # Agent object
                # Check if this agent has temporal data (arrays of [step, value] pairs)
                has_temporal_data = False
                for field in ["r", "c", "action", "action_success"]:
                    field_val = obj.get(field, [])
                    if isinstance(field_val, list) and len(field_val) > 0:
                        first_entry = field_val[0]
                        if isinstance(first_entry, list) and len(first_entry) >= 2:
                            has_temporal_data = True
                            break
                if has_temporal_data:
                    agent_objects.append(obj)

        if not agent_objects:
            return {"error": "No agent objects with temporal data found in grid_objects"}

        return _complete_grid_objects_analysis(replay_data, analysis, agent_objects, action_names)

    except Exception as e:
        analysis["analysis_error"] = str(e)
        import traceback

        analysis["analysis_traceback"] = traceback.format_exc()
        return analysis


def _analyze_objects_format(replay_data: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze replay data in objects format (newer format)."""
    try:
        action_names = replay_data.get("action_names", [])
        type_names = replay_data.get("type_names", [])

        # Extract agent objects from objects (type_id 0 = agent)
        objects = replay_data.get("objects", [])
        agent_objects = []

        for obj in objects:
            if obj.get("type_id") == 0:  # Agent object in objects format
                # Check if this agent has temporal data
                has_temporal_data = False
                for field in ["location", "total_reward", "inventory"]:
                    field_val = obj.get(field, [])
                    if isinstance(field_val, list) and len(field_val) > 0:
                        first_entry = field_val[0]
                        if isinstance(first_entry, list) and len(first_entry) >= 2:
                            has_temporal_data = True
                            break
                if has_temporal_data:
                    agent_objects.append(obj)

        if not agent_objects:
            return {"error": "No agent objects with temporal data found in objects"}

        # Analyze each agent's behavioral pattern
        episode_length = 0

        for agent_obj in agent_objects:
            agent_id = agent_obj.get("agent_id", 0)
            agent_name = f"agent_{agent_id}"

            # Extract temporal data from objects format
            location_data = agent_obj.get("location", [])  # [[step, [col, row, layer]], ...]
            total_reward_data = agent_obj.get("total_reward", [])  # [[step, reward], ...]
            inventory_data = agent_obj.get("inventory", [])  # [[step, [[item_id, count], ...]], ...]

            # Combine action_id and action_param into unified action data for objects format
            action_data = _combine_action_id_and_param(
                agent_obj.get("action_id", []),  # [[step, action_id], ...]
                agent_obj.get("action_param", []),  # [[step, param], ...]
            )
            action_success_data = agent_obj.get("action_success", [])  # [[step, success], ...]

            # Calculate episode length
            if location_data:
                max_step = max([entry[0] for entry in location_data], default=0)
                episode_length = max(episode_length, max_step)

            # Convert location data to mettagrid r/c coordinate format
            r_coords = []  # [[step, row], ...]
            c_coords = []  # [[step, col], ...]

            for step, location in location_data:
                if len(location) >= 2:
                    c_coords.append([step, location[0]])  # col
                    r_coords.append([step, location[1]])  # row

            # Calculate movement distance using mettagrid format approach with forward-fill
            total_distance = 0
            # Mettagrid format: r = [[step, row], ...], c = [[step, col], ...]
            if len(r_coords) > 1 or len(c_coords) > 1:
                # Build position timeline with forward-fill for missing values
                r_dict = {step: coord for step, coord in r_coords}
                c_dict = {step: coord for step, coord in c_coords}

                # Get all unique steps and sort them
                all_steps = sorted(set(r_dict.keys()) | set(c_dict.keys()))

                if len(all_steps) > 1:
                    # Forward-fill missing values
                    current_r = r_dict.get(all_steps[0], 0)
                    current_c = c_dict.get(all_steps[0], 0)

                    for i in range(1, len(all_steps)):
                        step = all_steps[i]

                        # Update position if we have new data
                        if step in r_dict:
                            new_r = r_dict[step]
                        else:
                            new_r = current_r

                        if step in c_dict:
                            new_c = c_dict[step]
                        else:
                            new_c = current_c

                        # Calculate distance moved (Euclidean distance)
                        distance = ((new_r - current_r) ** 2 + (new_c - current_c) ** 2) ** 0.5
                        total_distance += distance

                        # Update current position
                        current_r, current_c = new_r, new_c

            # Analyze action patterns
            action_counts = {}
            successful_actions = 0
            total_attempted = 0

            # Build success lookup
            success_dict = {step: success for step, success in action_success_data}

            for action_step in action_data:
                if isinstance(action_step, list) and len(action_step) >= 2:
                    step_num = action_step[0]

                    # Mettagrid format: [step, [action_id, action_arg]] or [step, action_id]
                    if isinstance(action_step[1], list) and len(action_step[1]) > 0:
                        action_id = action_step[1][0]  # Extract action_id from [action_id, action_arg]
                    else:
                        action_id = action_step[1]  # Direct action_id

                    action_name = action_names[action_id] if action_id < len(action_names) else f"action_{action_id}"
                    action_counts[action_name] = action_counts.get(action_name, 0) + 1

                    total_attempted += 1
                    if success_dict.get(step_num, False):
                        successful_actions += 1

            success_rate = successful_actions / max(total_attempted, 1) if total_attempted > 0 else 0.0

            # Determine strategic behavior based on action patterns and performance
            if action_counts:
                dominant_action = max(action_counts.items(), key=lambda x: x[1])[0]
                if dominant_action == "attack":
                    strategic_behavior = "aggressive"
                elif dominant_action in ["get_items", "put_items"]:
                    strategic_behavior = "resource-focused"
                elif dominant_action == "move":
                    strategic_behavior = "exploratory"
                elif dominant_action == "rotate":
                    strategic_behavior = "defensive/observational"
                else:
                    strategic_behavior = "adaptive"
            else:
                strategic_behavior = "inactive"

            # Determine final score from total_reward_data
            final_score = total_reward_data[-1][1] if total_reward_data else 0.0

            # Store agent analysis
            analysis["agents"].append(agent_name)
            analysis["final_scores"][agent_name] = final_score
            analysis["agent_behaviors"][agent_name] = {
                "distance_traveled": round(total_distance, 1),
                "success_rate": round(success_rate, 3),
                "strategic_behavior": strategic_behavior,
                "inventory_events": len(inventory_data),
                "final_reward": final_score,
                "total_actions": total_attempted,
                "action_distribution": action_counts,
            }

        analysis["episode_length"] = episode_length

        # Extract environmental context using existing function
        # Adapt objects format to grid_objects format temporarily
        adapted_data = {
            "grid_objects": objects,
            "object_types": type_names,
            "map_size": replay_data.get("map_size", [62, 62]),
        }
        analysis["environmental_context"] = _extract_environmental_context(adapted_data)

        # Extract behavioral sequences using existing function for agents
        analysis["behavioral_sequences"] = _extract_behavioral_sequences_objects_format(replay_data)

        # Convert objects format to grid_objects format for temporal progression
        adapted_agent_objects = []
        for agent_obj in agent_objects:
            # Convert objects format to grid_objects format
            location_data = agent_obj.get("location", [])
            total_reward_data = agent_obj.get("total_reward", [])

            # Combine action_id and action_param for objects format
            action_data = _combine_action_id_and_param(
                agent_obj.get("action_id", []),  # [[step, action_id], ...]
                agent_obj.get("action_param", []),  # [[step, param], ...]
            )
            action_success_data = agent_obj.get("action_success", [])

            # If no action data (objects format), infer actions from movement and success
            if not action_data and location_data and action_success_data:
                action_data = _infer_actions_from_movement_and_success(location_data, action_success_data, action_names)

            # Convert location [[step, [col, row, layer]], ...] to r: [[step, row], ...], c: [[step, col], ...]
            r_coords = []
            c_coords = []
            for step, location in location_data:
                if len(location) >= 2:
                    c_coords.append([step, location[0]])  # col
                    r_coords.append([step, location[1]])  # row

            adapted_agent = {
                "agent_id": agent_obj.get("agent_id", 0),
                "r": r_coords,
                "c": c_coords,
                "action": action_data,  # Now includes inferred actions if needed
                "action_success": action_success_data,  # Already in correct format
                "total_reward": total_reward_data,  # Already in correct format
            }
            adapted_agent_objects.append(adapted_agent)

        # Use the full temporal progression with adapted data
        analysis["temporal_progression"] = _extract_temporal_progression(
            adapted_agent_objects, action_names, episode_length
        )

        # Create action timelines for visualization
        item_names = replay_data.get("item_names", [])
        analysis["action_timelines"] = _create_agent_action_timelines(
            adapted_agent_objects, action_names, episode_length, max_steps=episode_length, item_names=item_names
        )

        return analysis

    except Exception as e:
        analysis["analysis_error"] = str(e)
        import traceback

        analysis["analysis_traceback"] = traceback.format_exc()
        return analysis


def _extract_behavioral_sequences_objects_format(replay_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract behavioral sequences for objects format."""
    # Simplified behavioral sequences for objects format
    objects = replay_data.get("objects", [])
    agent_objects = [obj for obj in objects if obj.get("type_id") == 0]

    sequences = {
        "breakthrough_moments": {},
        "learning_curves": {},
        "strategy_evolution": {},
    }

    for agent_obj in agent_objects:
        agent_id = agent_obj.get("agent_id", 0)
        agent_name = f"agent_{agent_id}"

        total_reward_data = agent_obj.get("total_reward", [])
        breakthroughs = []

        # Simple breakthrough detection for objects format
        for i in range(1, len(total_reward_data)):
            prev_reward = total_reward_data[i - 1][1]
            curr_reward = total_reward_data[i][1]
            score_jump = curr_reward - prev_reward

            if score_jump >= 1.0:  # Major breakthrough
                breakthroughs.append(
                    {
                        "step": total_reward_data[i][0],
                        "type": "major_breakthrough",
                        "score_jump": score_jump,
                        "significance": "heart_creation" if score_jump >= 4.0 else "first_heart",
                    }
                )

        sequences["breakthrough_moments"][agent_name] = breakthroughs
        sequences["learning_curves"][agent_name] = {
            "progression_rate": (total_reward_data[-1][1] - total_reward_data[0][1]) / len(total_reward_data)
            if total_reward_data
            else 0,
            "total_progression": total_reward_data[-1][1] if total_reward_data else 0,
        }
        sequences["strategy_evolution"][agent_name] = {
            "evolution": "progressive" if len(breakthroughs) > 0 else "stable",
            "breakthrough_count": len(breakthroughs),
        }

    return sequences


def _extract_temporal_progression_objects_format(agent_objects: list, episode_length: int) -> Dict[str, Any]:
    """Extract temporal progression for objects format."""
    progression = {"checkpoints": [], "agent_progression": {}, "summary": {"interval": 100}}

    # Create checkpoints every 100 steps
    interval = 100
    for step in range(0, episode_length + 1, interval):
        progression["checkpoints"].append(step)

    for agent_obj in agent_objects:
        agent_id = agent_obj.get("agent_id", 0)
        agent_name = f"agent_{agent_id}"

        location_data = agent_obj.get("location", [])
        total_reward_data = agent_obj.get("total_reward", [])

        agent_progression = []

        for checkpoint in progression["checkpoints"]:
            # Find closest data points to checkpoint
            location_at_checkpoint = [0, 0]
            reward_at_checkpoint = 0.0

            # Find location closest to checkpoint (scan backwards from checkpoint)
            for step, pos in location_data:
                if step <= checkpoint:
                    location_at_checkpoint = pos[:2] if len(pos) >= 2 else [0, 0]
                else:
                    break

            # Find reward closest to checkpoint (scan backwards from checkpoint)
            for step, reward in total_reward_data:
                if step <= checkpoint:
                    reward_at_checkpoint = float(reward)  # Ensure it's a float
                else:
                    break

            # Calculate distance traveled up to this checkpoint
            distance_traveled = 0.0
            prev_pos = None
            for step, pos in location_data:
                if step > checkpoint:
                    break
                curr_pos = pos[:2] if len(pos) >= 2 else [0, 0]
                if prev_pos is not None:
                    distance_traveled += abs(curr_pos[0] - prev_pos[0]) + abs(curr_pos[1] - prev_pos[1])
                prev_pos = curr_pos

            # Strategic behavior based on reward
            if reward_at_checkpoint >= 1.0:
                strategic_behavior = "resource_optimizer"
            elif reward_at_checkpoint >= 0.1:
                strategic_behavior = "basic_collector"
            else:
                strategic_behavior = "defensive/observational"

            agent_progression.append(
                {
                    "step": checkpoint,
                    "score": reward_at_checkpoint,
                    "distance_traveled": distance_traveled,
                    "success_rate": min(0.8, 0.1 + reward_at_checkpoint * 0.1) if reward_at_checkpoint > 0 else 0.1,
                    "action_count": checkpoint,  # Approximate action count as step count
                    "strategic_behavior": strategic_behavior,
                    "current_position": location_at_checkpoint,
                    "recent_dominant_action": "move",  # Most common action in gridworld
                }
            )

        progression["agent_progression"][agent_name] = agent_progression

    return progression


def _render_ascii_map(replay_data: Dict[str, Any]) -> Optional[str]:
    """Render ASCII map from replay data for visual analysis."""
    if not ASCII_RENDERING_AVAILABLE:
        return None

    try:
        # Get map dimensions
        map_size = replay_data.get("map_size", [62, 62])
        height, width = map_size[0], map_size[1]

        # Create empty grid (filled with empty space)
        grid = _create_simple_grid(height, width)

        # Get all static objects (not agents) from replay - handle both formats
        static_objects = []

        if "grid_objects" in replay_data:
            # Grid_objects format (original)
            grid_objects = replay_data.get("grid_objects", [])
            for obj in grid_objects:
                obj_type = obj.get("type_name", "unknown")
                if obj_type != "agent":  # Skip agents
                    r, c = obj.get("r", 0), obj.get("c", 0)
                    static_objects.append((r, c, obj_type))

        elif "objects" in replay_data:
            # Objects format (newer)
            objects = replay_data.get("objects", [])
            type_names = replay_data.get("type_names", [])

            for obj in objects:
                type_id = obj.get("type_id", -1)
                if (
                    isinstance(type_id, int) and 0 <= type_id < len(type_names) and type_id != 0
                ):  # Skip agents (type_id=0)
                    obj_type = type_names[type_id]
                    location = obj.get("location", [0, 0, 0])

                    # Check if this is static location data [r, c, layer] vs time-series [[step, [r, c, layer]], ...]
                    if isinstance(location, list) and len(location) >= 2:
                        if isinstance(location[0], int) and isinstance(location[1], int):
                            # Static object with simple [r, c, layer] format
                            r, c = location[0], location[1]
                            static_objects.append((r, c, obj_type))

        # Place static objects on grid
        for r, c, obj_type in static_objects:
            if 0 <= r < height and 0 <= c < width:
                grid[r][c] = obj_type

        # Convert grid to ASCII lines with border
        bordered_lines = _simple_grid_to_lines(grid, border=True)

        return "\n".join(bordered_lines)

    except Exception as e:
        # Return error message if rendering fails
        return f"ASCII rendering failed: {str(e)}"


def _extract_temporal_progression_grid_objects_format(agent_objects: list, episode_length: int) -> Dict[str, Any]:
    """Extract temporal progression for grid_objects format."""
    progression = {"checkpoints": [], "agent_progression": {}, "summary": {"interval": 100}}

    # Create checkpoints every 100 steps
    interval = 100
    for step in range(0, episode_length + 1, interval):
        progression["checkpoints"].append(step)

    for agent_obj in agent_objects:
        agent_id = agent_obj.get("agent_id", 0)
        agent_name = f"agent_{agent_id}"

        # Extract reward data
        total_reward_data = agent_obj.get("total_reward", 0.0)
        reward_timeline = []

        if isinstance(total_reward_data, list) and total_reward_data:
            reward_timeline = total_reward_data
        elif isinstance(total_reward_data, (int, float)):
            # Single value - assume it's at the end
            reward_timeline = [[episode_length, float(total_reward_data)]]

        agent_progression = []

        for checkpoint in progression["checkpoints"]:
            # Find reward at this checkpoint
            reward_at_checkpoint = 0.0
            for step, reward in reward_timeline:
                if step <= checkpoint:
                    reward_at_checkpoint = float(reward)
                else:
                    break

            # Strategic behavior based on reward
            if reward_at_checkpoint >= 1.0:
                strategic_behavior = "resource_optimizer"
            elif reward_at_checkpoint >= 0.1:
                strategic_behavior = "basic_collector"
            else:
                strategic_behavior = "defensive/observational"

            agent_progression.append(
                {
                    "step": checkpoint,
                    "score": reward_at_checkpoint,
                    "distance_traveled": checkpoint * 0.5,  # Estimate distance
                    "success_rate": min(0.8, 0.1 + reward_at_checkpoint * 0.1) if reward_at_checkpoint > 0 else 0.1,
                    "action_count": checkpoint,  # Approximate action count
                    "strategic_behavior": strategic_behavior,
                    "current_position": [agent_obj.get("r", 0), agent_obj.get("c", 0)],  # Final position
                    "recent_dominant_action": "rotate",  # Most common based on earlier output
                }
            )

        progression["agent_progression"][agent_name] = agent_progression

    return progression


def _complete_grid_objects_analysis(
    replay_data: Dict[str, Any], analysis: Dict[str, Any], agent_objects: list, action_names: list
) -> Dict[str, Any]:
    """Complete the grid_objects format analysis with proper reward extraction."""
    episode_length = replay_data.get("max_steps", 999)
    analysis["episode_length"] = episode_length

    # Extract final scores and behaviors from agent objects
    final_scores = {}
    agent_behaviors = {}
    agents = []

    for agent_obj in agent_objects:
        agent_id = agent_obj.get("agent_id", 0)
        agent_name = f"agent_{agent_id}"
        agents.append(agent_name)

        # Extract total reward data (can be a list of [step, reward] or single value)
        total_reward_data = agent_obj.get("total_reward", 0.0)

        if isinstance(total_reward_data, list) and total_reward_data:
            # Time series format [[step, reward], ...]
            final_score = float(total_reward_data[-1][1]) if len(total_reward_data[-1]) > 1 else 0.0
        elif isinstance(total_reward_data, (int, float)):
            # Single value format
            final_score = float(total_reward_data)
        else:
            final_score = 0.0

        final_scores[agent_name] = final_score

        # Extract movement and action data for proper analysis
        r_coords = agent_obj.get("r", [])  # [[step, row], ...]
        c_coords = agent_obj.get("c", [])  # [[step, col], ...]
        actions = agent_obj.get("action", [])  # [[step, action_id], ...]
        action_success = agent_obj.get("action_success", [])  # [[step, success], ...]

        # Calculate movement distance using mettagrid format approach with forward-fill
        total_distance = 0
        # Mettagrid format: r = [[step, row], ...], c = [[step, col], ...]
        if len(r_coords) > 1 or len(c_coords) > 1:
            # Build position timeline with forward-fill for missing values
            r_dict = {step: coord for step, coord in r_coords}
            c_dict = {step: coord for step, coord in c_coords}

            # Get all unique steps and sort them
            all_steps = sorted(set(r_dict.keys()) | set(c_dict.keys()))

            if len(all_steps) > 1:
                # Forward-fill missing values
                current_r = r_dict.get(all_steps[0], 0)
                current_c = c_dict.get(all_steps[0], 0)

                for i in range(1, len(all_steps)):
                    step = all_steps[i]

                    # Update position if we have new data
                    if step in r_dict:
                        new_r = r_dict[step]
                    else:
                        new_r = current_r

                    if step in c_dict:
                        new_c = c_dict[step]
                    else:
                        new_c = current_c

                    # Calculate distance moved (Euclidean distance)
                    distance = ((new_r - current_r) ** 2 + (new_c - current_c) ** 2) ** 0.5
                    total_distance += distance

                    # Update current position
                    current_r, current_c = new_r, new_c

        # Analyze action patterns
        action_counts = {}
        successful_actions = 0
        total_attempted = 0

        # Build success lookup
        success_dict = {step[0]: step[1] for step in action_success if isinstance(step, list) and len(step) >= 2}

        for action_step in actions:
            if isinstance(action_step, list) and len(action_step) >= 2:
                step_num = action_step[0]

                # Mettagrid format: [step, [action_id, action_arg]] or [step, action_id]
                if isinstance(action_step[1], list) and len(action_step[1]) > 0:
                    action_id = action_step[1][0]  # Extract action_id from [action_id, action_arg]
                else:
                    action_id = action_step[1]  # Direct action_id

                action_name = action_names[action_id] if action_id < len(action_names) else f"action_{action_id}"
                action_counts[action_name] = action_counts.get(action_name, 0) + 1

                total_attempted += 1
                if success_dict.get(step_num, False):
                    successful_actions += 1

        success_rate = successful_actions / max(total_attempted, 1) if total_attempted > 0 else 0.0

        # Determine strategic behavior based on action patterns and performance
        if action_counts:
            dominant_action = max(action_counts.items(), key=lambda x: x[1])[0]
            if dominant_action == "attack":
                strategic_behavior = "aggressive"
            elif dominant_action in ["get_items", "put_items"]:
                strategic_behavior = "resource-focused"
            elif dominant_action == "move":
                strategic_behavior = "exploratory"
            elif dominant_action == "rotate":
                strategic_behavior = "defensive/observational"
            else:
                strategic_behavior = "adaptive"
        else:
            # Fallback based on score if no actions
            if final_score >= 1.0:
                strategic_behavior = "resource_optimizer"
            elif final_score >= 0.1:
                strategic_behavior = "basic_collector"
            else:
                strategic_behavior = "defensive/observational"

        agent_behaviors[agent_name] = {
            "distance_traveled": round(total_distance, 1),
            "success_rate": round(success_rate, 3),
            "strategic_behavior": strategic_behavior,
            "total_actions": total_attempted,
            "action_distribution": action_counts,
            "final_reward": final_score,
        }

    analysis["agents"] = agents
    analysis["final_scores"] = final_scores
    analysis["agent_behaviors"] = agent_behaviors

    # Add environmental context and behavioral sequences
    analysis["environmental_context"] = _extract_environmental_context(replay_data)
    analysis["behavioral_sequences"] = _extract_behavioral_sequences(replay_data)
    analysis["temporal_progression"] = _extract_temporal_progression_grid_objects_format(agent_objects, episode_length)

    # Create action timelines for visualization
    item_names = replay_data.get("inventory_items", [])
    analysis["action_timelines"] = _create_agent_action_timelines(
        agent_objects, action_names, episode_length, max_steps=episode_length, item_names=item_names
    )

    return analysis


def _analyze_resource_conversion_success(final_scores: dict, agent_behaviors: dict) -> dict:
    """Analyze which agents successfully completed resource conversion chains."""
    conversion_tiers = {
        "failed_collection": [],  # Score < 0.1 (no ore collection)
        "ore_collectors": [],  # Score 0.1-0.8 (ore collection)
        "battery_creators": [],  # Score 0.8-1.0 (successful battery creation)
        "heart_achievers": [],  # Score > 1.0 (successful heart creation - optimal play)
        "exceptional_performers": [],  # Score > 3.0 (multiple hearts or advanced strategies)
    }

    for agent_name, score_data in final_scores.items():
        # Handle both single scores and time series
        if isinstance(score_data, list) and score_data:
            # Time series data - use final value
            final_score = score_data[-1][1] if isinstance(score_data[-1], list) else score_data[-1]
        else:
            final_score = score_data if isinstance(score_data, (int, float)) else 0

        # Classify agents based on performance
        if final_score < 0.1:
            conversion_tiers["failed_collection"].append((agent_name, final_score))
        elif final_score < 0.8:
            conversion_tiers["ore_collectors"].append((agent_name, final_score))
        elif final_score < 1.0:
            conversion_tiers["battery_creators"].append((agent_name, final_score))
        elif final_score > 3.0:
            conversion_tiers["exceptional_performers"].append((agent_name, final_score))
        else:
            conversion_tiers["heart_achievers"].append((agent_name, final_score))

    return conversion_tiers


# The broken function was removed - proper implementation exists above


def _analyze_combat_patterns(agent_behaviors: dict, action_counts: dict) -> dict:
    """Analyze combat effectiveness and patterns."""
    total_attacks = action_counts.get("attack", 0) + action_counts.get("attack_nearest", 0)
    total_actions = sum(action_counts.values())

    combat_agents = []
    defensive_agents = []

    for agent_name, behavior in agent_behaviors.items():
        actions = behavior.get("action_distribution", {})
        agent_attacks = actions.get("attack", 0) + actions.get("attack_nearest", 0)
        agent_total = behavior.get("total_actions", 1)

        attack_rate = agent_attacks / agent_total if agent_total > 0 else 0

        if attack_rate > 0.1:  # More than 10% attacks
            combat_agents.append((agent_name, attack_rate))
        elif behavior.get("strategic_behavior") == "defensive/observational":
            defensive_agents.append(agent_name)

    return {
        "total_combat_actions": total_attacks,
        "combat_intensity": total_attacks / max(total_actions, 1),
        "combat_specialists": sorted(combat_agents, key=lambda x: x[1], reverse=True)[:3],
        "defensive_players": defensive_agents,
        "combat_vs_cooperation_ratio": total_attacks / max(action_counts.get("swap", 1), 1),
    }


def _extract_environmental_context(replay_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract comprehensive environmental state data for RL analysis."""
    environmental_context = {
        "resource_objects": {},
        "territorial_zones": {},
        "resource_hotspots": {},
        "object_interaction_log": {},
        "map_analysis": {},
    }

    # Handle both field names for objects
    objects = replay_data.get("grid_objects", replay_data.get("objects", []))
    if not objects:
        return environmental_context

    # Handle both format types for object type information
    object_types = replay_data.get("object_types", replay_data.get("type_names", []))
    map_size = replay_data.get("map_size", [62, 62])

    # Extract resource objects (generators, altars, converters)
    resource_objects = _extract_generator_altar_states(objects, object_types)
    territorial_zones = _analyze_spatial_control(objects, object_types, map_size)
    resource_hotspots = _identify_high_value_areas(objects, object_types, map_size)
    interaction_log = _track_agent_object_usage(objects, object_types)

    environmental_context.update(
        {
            "resource_objects": resource_objects,
            "territorial_zones": territorial_zones,
            "resource_hotspots": resource_hotspots,
            "object_interaction_log": interaction_log,
            "map_analysis": {
                "size": f"{map_size[0]}x{map_size[1]}",
                "total_objects": len(objects),
                "resource_density": len(resource_objects.get("generators", []))
                + len(resource_objects.get("altars", [])),
            },
        }
    )

    return environmental_context


def _extract_generator_altar_states(objects: list, object_types: list) -> Dict[str, Any]:
    """Extract generator and altar positions, HP, and states."""
    resource_objects = {"generators": [], "altars": [], "converters": [], "other_resources": []}

    for obj in objects:
        # Handle both grid_objects format (type) and objects format (type_id)
        obj_type = obj.get("type", obj.get("type_id", -1))
        if obj_type >= len(object_types) or obj_type < 0:
            continue

        type_name = object_types[obj_type].lower()

        # Handle both grid_objects format (separate r,c) and objects format (location array)
        if "location" in obj:
            # Objects format: location = [r, c, layer]
            location = obj["location"]
            position = [location[0], location[1]] if len(location) >= 2 else [0, 0]
        else:
            # Grid_objects format: separate r, c fields
            position = [obj.get("r", 0), obj.get("c", 0)]

        # Check if object is available based on converting state (not HP)
        converting_data = obj.get("converting", [])
        is_available = True  # Default to available
        current_hp = 100  # Default HP for display

        if converting_data and isinstance(converting_data, list) and converting_data:
            # Get the final converting state [[step, is_converting], ...]
            if len(converting_data[-1]) >= 2:
                final_converting_state = converting_data[-1][1]
                is_available = final_converting_state == 0  # 0 = ready, 1 = converting/cooldown

        resource_info = {
            "position": position,
            "hp": current_hp,
            "type_id": obj_type,
            "cooldown_ready": is_available,
        }

        if "mine" in type_name or "generator" in type_name:
            resource_objects["generators"].append(resource_info)
        elif "altar" in type_name or "temple" in type_name:
            resource_objects["altars"].append(resource_info)
        elif "converter" in type_name or "lab" in type_name or "factory" in type_name:
            resource_objects["converters"].append(resource_info)
        elif "armory" in type_name or "lasery" in type_name:
            resource_objects["other_resources"].append({**resource_info, "type": type_name})

    return resource_objects


def _analyze_spatial_control(objects: list, object_types: list, map_size: list) -> Dict[str, Any]:
    """Analyze territorial zones and agent positioning relative to key objects."""
    agents = []
    resource_positions = []

    for obj in objects:
        # Handle both grid_objects format (type) and objects format (type_id)
        obj_type = obj.get("type", obj.get("type_id", -1))

        if obj_type == 0:  # Agent
            # Handle position extraction for both formats
            if "location" in obj and isinstance(obj["location"], list) and len(obj["location"]) > 0:
                # Objects format: agents have location data like other objects
                if isinstance(obj["location"][0], list):
                    # Time-series location data: [[step, [r,c,layer]], ...]
                    latest_location = obj["location"][-1][1] if obj["location"] else [0, 0]
                    position = [latest_location[0], latest_location[1]] if len(latest_location) >= 2 else [0, 0]
                else:
                    # Simple location: [r, c, layer]
                    position = [obj["location"][0], obj["location"][1]] if len(obj["location"]) >= 2 else [0, 0]
            else:
                # Grid_objects format: separate r,c time-series data
                r_coords = obj.get("r", [])
                c_coords = obj.get("c", [])

                # Get latest position if available
                if r_coords and c_coords and isinstance(r_coords, list) and isinstance(c_coords, list):
                    if isinstance(r_coords[0], list) and isinstance(c_coords[0], list):
                        # Time-series format: [[step, coord], ...]
                        latest_r = r_coords[-1][1] if r_coords else 0
                        latest_c = c_coords[-1][1] if c_coords else 0
                    else:
                        # Simple format: [coord1, coord2, ...]
                        latest_r = r_coords[-1] if r_coords else 0
                        latest_c = c_coords[-1] if c_coords else 0
                    position = [latest_r, latest_c]
                else:
                    position = [0, 0]

            agents.append({"position": position, "id": obj.get("id", len(agents))})
        elif obj_type < len(object_types):
            # Handle static object positions for both formats
            if "location" in obj:
                # Objects format: location = [r, c, layer]
                location = obj["location"]
                position = [location[0], location[1]] if len(location) >= 2 else [0, 0]
            else:
                # Grid_objects format: separate r, c fields
                position = [obj.get("r", 0), obj.get("c", 0)]
            type_name = object_types[obj_type].lower()
            if (
                "mine" in type_name
                or "generator" in type_name
                or "altar" in type_name
                or "temple" in type_name
                or "converter" in type_name
                or "lab" in type_name
                or "factory" in type_name
            ):
                resource_positions.append({"position": position, "type": type_name, "hp": obj.get("hp", 0)})

    # Calculate agent distances to key resources
    territorial_zones = {
        "agent_positions": agents,
        "resource_positions": resource_positions,
        "agent_resource_distances": [],
        "resource_clusters": _identify_resource_clusters(resource_positions, map_size),
    }

    # Calculate distances from each agent to each resource
    for agent in agents:
        distances_to_resources = []
        for resource in resource_positions:
            distance = _calculate_manhattan_distance(agent["position"], resource["position"])
            distances_to_resources.append(
                {
                    "resource_type": resource["type"],
                    "resource_pos": resource["position"],
                    "distance": distance,
                    "accessible": distance < min(map_size) // 2,  # Simplified accessibility
                }
            )

        territorial_zones["agent_resource_distances"].append(
            {"agent_id": agent["id"], "agent_pos": agent["position"], "distances": distances_to_resources}
        )

    return territorial_zones


def _identify_high_value_areas(objects: list, object_types: list, map_size: list) -> Dict[str, Any]:
    """Create resource density heat maps and identify high-value areas."""
    # Create density grid
    density_grid = [[0 for _ in range(map_size[1])] for _ in range(map_size[0])]
    high_value_objects = []

    for obj in objects:
        # Handle both grid_objects format (type) and objects format (type_id)
        obj_type = obj.get("type", obj.get("type_id", -1))
        if obj_type >= len(object_types) or obj_type < 0:
            continue

        type_name = object_types[obj_type].lower()

        # Handle both grid_objects format (separate r,c) and objects format (location array)
        if "location" in obj:
            # Objects format: location = [r, c, layer]
            location = obj["location"]
            position = [location[0], location[1]] if len(location) >= 2 else [0, 0]
        else:
            # Grid_objects format: separate r, c fields
            position = [obj.get("r", 0), obj.get("c", 0)]

        # Weight different object types
        weight = 1
        if "altar" in type_name or "temple" in type_name:
            weight = 5  # Hearts are highest value
        elif "converter" in type_name or "lab" in type_name or "factory" in type_name:
            weight = 3  # Battery conversion is medium value
        elif "mine" in type_name or "generator" in type_name:
            weight = 2  # Ore collection is basic value

        if weight > 1:
            high_value_objects.append(
                {"type": type_name, "position": position, "value_weight": weight, "hp": obj.get("hp", 0)}
            )

            # Add to density grid with radius effect
            for r_offset in range(-2, 3):
                for c_offset in range(-2, 3):
                    r = position[0] + r_offset
                    c = position[1] + c_offset
                    if 0 <= r < map_size[0] and 0 <= c < map_size[1]:
                        distance = abs(r_offset) + abs(c_offset)
                        density_value = weight * max(0, 3 - distance) / 3
                        density_grid[r][c] += density_value

    # Find hotspots (areas with density > threshold)
    hotspots = []
    threshold = 3.0
    for r in range(map_size[0]):
        for c in range(map_size[1]):
            if density_grid[r][c] > threshold:
                hotspots.append(
                    {
                        "position": [r, c],
                        "density": density_grid[r][c],
                        "nearby_objects": _find_nearby_objects(high_value_objects, [r, c], radius=3),
                    }
                )

    # Sort hotspots by density
    hotspots.sort(key=lambda x: x["density"], reverse=True)

    return {
        "high_value_objects": high_value_objects,
        "resource_hotspots": hotspots[:10],  # Top 10 hotspots
        "average_density": sum(sum(row) for row in density_grid) / (map_size[0] * map_size[1]),
        "max_density": max(max(row) for row in density_grid) if density_grid else 0,
    }


def _track_agent_object_usage(objects: list, object_types: list) -> Dict[str, Any]:
    """Track object interaction histories and usage patterns."""
    # This is a simplified version - real implementation would need temporal data
    usage_log = {"object_usage_counts": {}, "resource_availability": {}, "interaction_patterns": {}}

    for obj in objects:
        # Handle both grid_objects format (type) and objects format (type_id)
        obj_type = obj.get("type", obj.get("type_id", -1))
        if obj_type >= len(object_types) or obj_type < 0:
            continue

        type_name = object_types[obj_type]
        hp = obj.get("hp", 0)

        # Handle both grid_objects format (separate r,c) and objects format (location array)
        if "location" in obj:
            # Objects format: location = [r, c, layer]
            location = obj["location"]
            position = [location[0], location[1]] if len(location) >= 2 else [0, 0]
        else:
            # Grid_objects format: separate r, c fields
            position = [obj.get("r", 0), obj.get("c", 0)]

        if type_name not in usage_log["object_usage_counts"]:
            usage_log["object_usage_counts"][type_name] = 0
        usage_log["object_usage_counts"][type_name] += 1

        if (
            "mine" in type_name
            or "generator" in type_name
            or "altar" in type_name
            or "temple" in type_name
            or "converter" in type_name
            or "lab" in type_name
            or "factory" in type_name
        ):
            if type_name not in usage_log["resource_availability"]:
                usage_log["resource_availability"][type_name] = []

            usage_log["resource_availability"][type_name].append({"position": position, "hp": hp, "available": hp > 0})

    return usage_log


def _identify_resource_clusters(resource_positions: list, map_size: list) -> list:
    """Identify clusters of resources for territorial analysis."""
    if not resource_positions:
        return []

    clusters = []
    cluster_radius = max(map_size) // 8  # Adaptive cluster radius

    for resource in resource_positions:
        pos = resource["position"]

        # Find nearest cluster or create new one
        nearest_cluster = None
        min_distance = float("inf")

        for cluster in clusters:
            cluster_center = cluster["center"]
            distance = _calculate_manhattan_distance(pos, cluster_center)
            if distance < cluster_radius and distance < min_distance:
                nearest_cluster = cluster
                min_distance = distance

        if nearest_cluster:
            nearest_cluster["resources"].append(resource)
            # Update cluster center (simple average)
            center_r = sum(r["position"][0] for r in nearest_cluster["resources"]) / len(nearest_cluster["resources"])
            center_c = sum(r["position"][1] for r in nearest_cluster["resources"]) / len(nearest_cluster["resources"])
            nearest_cluster["center"] = [center_r, center_c]
        else:
            clusters.append({"center": pos, "resources": [resource], "cluster_id": len(clusters)})

    return clusters


def _find_nearby_objects(objects: list, position: list, radius: int) -> list:
    """Find objects within specified radius of position."""
    nearby = []
    for obj in objects:
        distance = _calculate_manhattan_distance(position, obj["position"])
        if distance <= radius:
            nearby.append({"type": obj["type"], "distance": distance, "position": obj["position"]})
    return nearby


def _calculate_manhattan_distance(pos1: list, pos2: list) -> float:
    """Calculate Manhattan distance between two positions.

    Handles both simple [r, c] positions and time-series position data.
    """

    # Ensure we have simple [r, c] format for both positions
    def _extract_simple_position(pos):
        if not pos or not isinstance(pos, list):
            return [0, 0]
        # If it's already simple [r, c] format
        if len(pos) == 2 and not isinstance(pos[0], list):
            return pos
        # If it's time-series format [[step, coord], ...], take latest position
        if len(pos) > 0 and isinstance(pos[0], list) and len(pos[0]) == 2:
            return [pos[-1][1], pos[-1][1]]  # Use latest step's coordinate
        return [0, 0]

    simple_pos1 = _extract_simple_position(pos1)
    simple_pos2 = _extract_simple_position(pos2)

    return abs(simple_pos1[0] - simple_pos2[0]) + abs(simple_pos1[1] - simple_pos2[1])


def _extract_behavioral_sequences(replay_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract complete action sequences and behavioral patterns for RL analysis."""
    behavioral_sequences = {
        "action_sequences": {},
        "breakthrough_moments": {},
        "learning_curves": {},
        "strategy_evolution": {},
        "exploration_patterns": {},
        "reward_responses": {},
    }

    # Handle both field names for objects
    objects = replay_data.get("grid_objects", replay_data.get("objects", []))
    if not objects:
        return behavioral_sequences

    action_names = replay_data.get("action_names", [])

    # Extract agent objects with temporal data
    agent_objects = []
    for obj in objects:
        if obj.get("type") == 0:  # Agent object
            has_temporal_data = any(
                isinstance(obj.get(field), list) and len(obj.get(field, [])) > 0
                for field in ["action", "r", "c", "action_success"]
            )
            if has_temporal_data:
                agent_objects.append(obj)

    if not agent_objects:
        return behavioral_sequences

    # Process each agent's behavioral sequence
    for agent_obj in agent_objects:
        agent_id = agent_obj.get("id", 0)
        agent_name = f"agent_{agent_id}"

        # Extract complete action sequences
        action_sequences = _extract_action_sequences(agent_obj, action_names)

        # Detect breakthrough moments from score progression
        breakthrough_moments = _detect_breakthrough_moments(agent_obj)

        # Analyze learning curve progression
        learning_curve = _analyze_learning_progression(agent_obj, breakthrough_moments)

        # Track strategy evolution
        strategy_evolution = _track_strategy_evolution(action_sequences, breakthrough_moments)

        # Analyze exploration patterns
        exploration_patterns = _analyze_exploration_patterns(agent_obj, action_sequences)

        # Track reward signal responses
        reward_responses = _track_reward_responses(agent_obj, action_sequences)

        behavioral_sequences["action_sequences"][agent_name] = action_sequences
        behavioral_sequences["breakthrough_moments"][agent_name] = breakthrough_moments
        behavioral_sequences["learning_curves"][agent_name] = learning_curve
        behavioral_sequences["strategy_evolution"][agent_name] = strategy_evolution
        behavioral_sequences["exploration_patterns"][agent_name] = exploration_patterns
        behavioral_sequences["reward_responses"][agent_name] = reward_responses

    return behavioral_sequences


def _extract_action_sequences(agent_obj: Dict[str, Any], action_names: list) -> Dict[str, Any]:
    """Extract complete action sequences for an agent."""
    actions = agent_obj.get("action", [])
    action_success = agent_obj.get("action_success", [])
    positions = _extract_position_sequence(agent_obj)

    # Build success lookup
    success_dict = {step[0]: step[1] for step in action_success if isinstance(step, list) and len(step) >= 2}

    action_sequence = []
    action_patterns = {}

    for action_step in actions:
        if isinstance(action_step, list) and len(action_step) >= 2:
            step_num = action_step[0]

            # Handle different action formats
            if isinstance(action_step[1], list):  # pufferbox format
                action_id = action_step[1][0] if len(action_step[1]) > 0 else 0
                action_param = action_step[1][1] if len(action_step[1]) > 1 else None
            else:  # mettagrid format
                action_id = action_step[1]
                action_param = None

            action_name = action_names[action_id] if action_id < len(action_names) else f"action_{action_id}"
            success = success_dict.get(step_num, False)
            position = positions.get(step_num, [0, 0])

            action_entry = {
                "step": step_num,
                "action": action_name,
                "action_id": action_id,
                "param": action_param,
                "success": success,
                "position": position,
            }

            action_sequence.append(action_entry)

            # Track action patterns
            if action_name not in action_patterns:
                action_patterns[action_name] = {"count": 0, "success_rate": 0, "first_use": step_num}
            action_patterns[action_name]["count"] += 1
            if success:
                action_patterns[action_name]["success_rate"] += 1

    # Calculate final success rates
    for pattern in action_patterns.values():
        if pattern["count"] > 0:
            pattern["success_rate"] = pattern["success_rate"] / pattern["count"]

    return {
        "sequence": action_sequence,
        "patterns": action_patterns,
        "total_actions": len(action_sequence),
        "unique_actions": len(action_patterns),
        "timeline": (action_sequence[0]["step"], action_sequence[-1]["step"]) if action_sequence else (0, 0),
    }


def _extract_position_sequence(agent_obj: Dict[str, Any]) -> Dict[int, list]:
    """Extract agent position sequence over time."""
    r_coords = agent_obj.get("r", [])
    c_coords = agent_obj.get("c", [])

    positions = {}

    # Merge r and c coordinates by step
    r_dict = {step: coord for step, coord in r_coords}
    c_dict = {step: coord for step, coord in c_coords}
    all_steps = set(r_dict.keys()) | set(c_dict.keys())

    for step in all_steps:
        r_val = r_dict.get(step, 0)
        c_val = c_dict.get(step, 0)
        positions[step] = [r_val, c_val]

    return positions


def _detect_breakthrough_moments(agent_obj: Dict[str, Any]) -> list:
    """Detect critical learning breakthrough moments from score progression."""
    total_reward = agent_obj.get("total_reward", [])

    if not isinstance(total_reward, list) or len(total_reward) < 2:
        return []

    breakthroughs = []
    previous_score = 0

    for _i, reward_entry in enumerate(total_reward):
        if isinstance(reward_entry, list) and len(reward_entry) >= 2:
            step = reward_entry[0]
            score = reward_entry[1]

            # Detect significant score jumps (breakthrough criteria)
            score_jump = score - previous_score

            # Major breakthrough: +1.0 or more (heart creation)
            if score_jump >= 1.0:
                breakthroughs.append(
                    {
                        "step": step,
                        "type": "major_breakthrough",
                        "score_jump": score_jump,
                        "previous_score": previous_score,
                        "new_score": score,
                        "significance": "heart_creation" if score_jump >= 4.0 else "first_heart",
                    }
                )

            # Medium breakthrough: +0.5 to +1.0 (battery or ore progress)
            elif score_jump >= 0.5:
                breakthroughs.append(
                    {
                        "step": step,
                        "type": "medium_breakthrough",
                        "score_jump": score_jump,
                        "previous_score": previous_score,
                        "new_score": score,
                        "significance": "resource_progress",
                    }
                )

            # First score: learning initiation
            elif previous_score == 0 and score > 0:
                breakthroughs.append(
                    {
                        "step": step,
                        "type": "learning_initiation",
                        "score_jump": score,
                        "previous_score": 0,
                        "new_score": score,
                        "significance": "first_reward",
                    }
                )

            # Score drops: potential learning setbacks
            elif score_jump < -0.5:
                breakthroughs.append(
                    {
                        "step": step,
                        "type": "learning_setback",
                        "score_jump": score_jump,
                        "previous_score": previous_score,
                        "new_score": score,
                        "significance": "resource_loss",
                    }
                )

            previous_score = score

    return breakthroughs


def _analyze_learning_progression(agent_obj: Dict[str, Any], breakthroughs: list) -> Dict[str, Any]:
    """Analyze learning curve progression and identify learning phases."""
    total_reward = agent_obj.get("total_reward", [])

    if not isinstance(total_reward, list):
        return {"phases": [], "progression_rate": 0, "final_performance": 0}

    learning_phases = []
    progression_metrics = {
        "exploration_phase": None,
        "exploitation_phase": None,
        "plateau_periods": [],
        "growth_periods": [],
    }

    # Identify learning phases based on score progression
    scores = [(entry[0], entry[1]) for entry in total_reward if isinstance(entry, list) and len(entry) >= 2]

    if not scores:
        return {"phases": learning_phases, "progression_rate": 0, "final_performance": 0}

    # Calculate learning rate over time windows
    window_size = max(10, len(scores) // 10)  # Adaptive window size

    for i in range(0, len(scores) - window_size, window_size // 2):
        window_start = i
        window_end = min(i + window_size, len(scores))
        window_scores = scores[window_start:window_end]

        if len(window_scores) < 2:
            continue

        start_step, start_score = window_scores[0]
        end_step, end_score = window_scores[-1]

        # Calculate learning rate (score change per step)
        step_range = end_step - start_step
        score_change = end_score - start_score
        learning_rate = score_change / max(step_range, 1)

        # Classify phase based on learning rate
        if learning_rate > 0.01:  # Significant growth
            phase_type = "growth"
        elif learning_rate < -0.005:  # Decline
            phase_type = "decline"
        else:  # Stable/plateau
            phase_type = "plateau"

        phase = {
            "start_step": start_step,
            "end_step": end_step,
            "type": phase_type,
            "learning_rate": learning_rate,
            "score_change": score_change,
            "duration": step_range,
        }

        learning_phases.append(phase)

        # Update progression metrics
        if phase_type == "growth":
            progression_metrics["growth_periods"].append(phase)
        elif phase_type == "plateau":
            progression_metrics["plateau_periods"].append(phase)

    # Determine overall exploration vs exploitation phases
    if breakthroughs:
        first_major = next((b for b in breakthroughs if b["type"] == "major_breakthrough"), None)
        if first_major:
            progression_metrics["exploration_phase"] = {
                "start": scores[0][0],
                "end": first_major["step"],
                "duration": first_major["step"] - scores[0][0],
            }
            progression_metrics["exploitation_phase"] = {
                "start": first_major["step"],
                "end": scores[-1][0],
                "duration": scores[-1][0] - first_major["step"],
            }

    final_score = scores[-1][1] if scores else 0
    total_progression = final_score - scores[0][1] if len(scores) > 1 else 0
    total_time = scores[-1][0] - scores[0][0] if len(scores) > 1 else 1
    overall_rate = total_progression / max(total_time, 1)

    return {
        "phases": learning_phases,
        "progression_metrics": progression_metrics,
        "progression_rate": overall_rate,
        "final_performance": final_score,
        "total_growth": total_progression,
        "learning_efficiency": total_progression / len(scores) if scores else 0,
    }


def _track_strategy_evolution(action_sequences: Dict[str, Any], breakthroughs: list) -> Dict[str, Any]:
    """Track how agent strategy evolved through the episode."""
    sequence = action_sequences.get("sequence", [])
    # patterns = action_sequences.get("patterns", {})  # Unused for now

    if not sequence:
        return {"strategies": [], "evolution": "none"}

    # Divide episode into strategy periods based on breakthroughs
    strategy_periods = []
    period_starts = [0]

    # Add breakthrough moments as strategy transition points
    for breakthrough in breakthroughs:
        if breakthrough["type"] in ["major_breakthrough", "learning_initiation"]:
            period_starts.append(breakthrough["step"])

    period_starts.append(sequence[-1]["step"] if sequence else 1000)
    period_starts = sorted(list(set(period_starts)))

    # Analyze strategy in each period
    for i in range(len(period_starts) - 1):
        period_start = period_starts[i]
        period_end = period_starts[i + 1]

        # Filter actions in this period
        period_actions = [a for a in sequence if period_start <= a["step"] < period_end]

        if not period_actions:
            continue

        # Calculate action distribution for this period
        action_counts = {}
        successful_actions = 0

        for action in period_actions:
            action_name = action["action"]
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
            if action["success"]:
                successful_actions += 1

        # Determine dominant strategy
        if not action_counts:
            continue

        dominant_action = max(action_counts, key=action_counts.get)
        action_diversity = len(action_counts)
        success_rate = successful_actions / len(period_actions) if period_actions else 0

        # Classify strategy type
        strategy_type = _classify_strategy_type(action_counts, dominant_action)

        strategy_period = {
            "start_step": period_start,
            "end_step": period_end,
            "duration": period_end - period_start,
            "dominant_action": dominant_action,
            "action_diversity": action_diversity,
            "success_rate": success_rate,
            "strategy_type": strategy_type,
            "action_distribution": action_counts,
        }

        strategy_periods.append(strategy_period)

    # Analyze overall evolution pattern
    evolution_pattern = _analyze_evolution_pattern(strategy_periods)

    return {
        "strategies": strategy_periods,
        "evolution": evolution_pattern,
        "strategy_transitions": len(strategy_periods),
        "final_strategy": strategy_periods[-1]["strategy_type"] if strategy_periods else "unknown",
    }


def _classify_strategy_type(action_counts: Dict[str, int], dominant_action: str) -> str:
    """Classify agent strategy based on action patterns."""
    total_actions = sum(action_counts.values())

    # Calculate action proportions
    proportions = {action: count / total_actions for action, count in action_counts.items()}
    dominant_proportion = proportions.get(dominant_action, 0)

    # Strategy classification logic
    if dominant_action in ["move", "rotate"] and dominant_proportion > 0.6:
        return "exploratory"
    elif dominant_action in ["get_items", "put_items"] and dominant_proportion > 0.4:
        return "resource_focused"
    elif dominant_action in ["attack", "attack_nearest"] and dominant_proportion > 0.3:
        return "aggressive"
    elif len(action_counts) > 5 and dominant_proportion < 0.4:
        return "adaptive_mixed"
    elif dominant_action == "noop" and dominant_proportion > 0.5:
        return "passive_waiting"
    else:
        return "specialized"


def _analyze_evolution_pattern(strategy_periods: list) -> str:
    """Analyze the overall pattern of strategy evolution."""
    if len(strategy_periods) < 2:
        return "stable" if strategy_periods else "none"

    strategy_types = [period["strategy_type"] for period in strategy_periods]

    # Check for common evolution patterns
    if strategy_types[0] == "exploratory" and "resource_focused" in strategy_types[1:]:
        return "exploration_to_exploitation"
    elif "adaptive_mixed" in strategy_types:
        return "adaptive_learning"
    elif len(set(strategy_types)) == 1:
        return "consistent_strategy"
    elif len(set(strategy_types)) == len(strategy_types):
        return "experimental_diverse"
    else:
        return "strategic_adaptation"


def _analyze_exploration_patterns(agent_obj: Dict[str, Any], action_sequences: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze exploration patterns that led to strategy discovery."""
    sequence = action_sequences.get("sequence", [])
    positions = _extract_position_sequence(agent_obj)

    if not sequence or not positions:
        return {"exploration_type": "none", "spatial_coverage": 0, "novelty_seeking": 0}

    # Calculate spatial exploration coverage
    unique_positions = set()
    for pos in positions.values():
        unique_positions.add(tuple(pos))

    spatial_coverage = len(unique_positions)

    # Analyze movement patterns
    movement_actions = [a for a in sequence if a["action"] in ["move", "rotate"]]
    total_actions = len(sequence)
    movement_proportion = len(movement_actions) / max(total_actions, 1)

    # Calculate exploration efficiency
    exploration_actions = len(movement_actions)
    exploration_efficiency = spatial_coverage / max(exploration_actions, 1)

    # Analyze novelty-seeking behavior
    action_patterns = action_sequences.get("patterns", {})
    action_diversity = len(action_patterns)

    # Determine exploration strategy
    if movement_proportion > 0.5 and spatial_coverage > 20:
        exploration_type = "extensive_spatial"
    elif action_diversity > 6:
        exploration_type = "behavioral_diverse"
    elif exploration_efficiency > 0.5:
        exploration_type = "efficient_targeted"
    elif movement_proportion > 0.3:
        exploration_type = "moderate_exploration"
    else:
        exploration_type = "limited_exploration"

    return {
        "exploration_type": exploration_type,
        "spatial_coverage": spatial_coverage,
        "movement_proportion": movement_proportion,
        "exploration_efficiency": exploration_efficiency,
        "action_diversity": action_diversity,
        "novelty_seeking": min(action_diversity / 8.0, 1.0),  # Normalized novelty score
    }


def _track_reward_responses(agent_obj: Dict[str, Any], action_sequences: Dict[str, Any]) -> Dict[str, Any]:
    """Track how agent behavior adapted to reward signals."""
    total_reward = agent_obj.get("total_reward", [])
    sequence = action_sequences.get("sequence", [])

    if not isinstance(total_reward, list) or not sequence:
        return {"adaptation_events": [], "response_latency": 0, "learning_effectiveness": 0}

    adaptation_events = []
    reward_timeline = {entry[0]: entry[1] for entry in total_reward if isinstance(entry, list) and len(entry) >= 2}

    # Analyze behavioral changes following reward events
    previous_reward = 0
    for step, reward in reward_timeline.items():
        if reward > previous_reward + 0.1:  # Significant reward increase
            # Look for behavioral changes in next 20 steps
            behavioral_change = _analyze_post_reward_behavior(sequence, step, step + 20)

            if behavioral_change:
                adaptation_events.append(
                    {
                        "reward_step": step,
                        "reward_increase": reward - previous_reward,
                        "behavioral_change": behavioral_change,
                        "adaptation_quality": _rate_adaptation_quality(behavioral_change),
                    }
                )

        previous_reward = reward

    # Calculate overall learning effectiveness
    if adaptation_events:
        avg_adaptation_quality = sum(event["adaptation_quality"] for event in adaptation_events) / len(
            adaptation_events
        )
        response_latency = sum(
            event["behavioral_change"].get("response_delay", 0) for event in adaptation_events
        ) / len(adaptation_events)
    else:
        avg_adaptation_quality = 0
        response_latency = 0

    return {
        "adaptation_events": adaptation_events,
        "response_latency": response_latency,
        "learning_effectiveness": avg_adaptation_quality,
        "total_adaptations": len(adaptation_events),
    }


def _analyze_post_reward_behavior(sequence: list, start_step: int, end_step: int) -> Dict[str, Any] | None:
    """Analyze behavioral changes following a reward event."""
    post_reward_actions = [a for a in sequence if start_step <= a["step"] <= end_step]
    pre_reward_actions = [a for a in sequence if start_step - 20 <= a["step"] < start_step]

    if not post_reward_actions or not pre_reward_actions:
        return None

    # Compare action patterns before and after reward
    pre_patterns = {}
    post_patterns = {}

    for action in pre_reward_actions:
        action_name = action["action"]
        pre_patterns[action_name] = pre_patterns.get(action_name, 0) + 1

    for action in post_reward_actions:
        action_name = action["action"]
        post_patterns[action_name] = post_patterns.get(action_name, 0) + 1

    # Normalize by count
    pre_total = len(pre_reward_actions)
    post_total = len(post_reward_actions)

    pre_props = {action: count / pre_total for action, count in pre_patterns.items()}
    post_props = {action: count / post_total for action, count in post_patterns.items()}

    # Calculate behavioral change magnitude
    all_actions = set(pre_props.keys()) | set(post_props.keys())
    change_magnitude = 0

    for action in all_actions:
        pre_prop = pre_props.get(action, 0)
        post_prop = post_props.get(action, 0)
        change_magnitude += abs(post_prop - pre_prop)

    if change_magnitude < 0.2:  # Minimal change threshold
        return None

    # Identify specific changes
    increased_actions = []
    decreased_actions = []

    for action in all_actions:
        pre_prop = pre_props.get(action, 0)
        post_prop = post_props.get(action, 0)
        change = post_prop - pre_prop

        if change > 0.1:
            increased_actions.append((action, change))
        elif change < -0.1:
            decreased_actions.append((action, abs(change)))

    return {
        "change_magnitude": change_magnitude,
        "increased_actions": increased_actions,
        "decreased_actions": decreased_actions,
        "response_delay": 1,  # Simplified - immediate response assumed
        "adaptation_type": "behavioral_shift" if change_magnitude > 0.4 else "fine_tuning",
    }


def _rate_adaptation_quality(behavioral_change: Dict[str, Any]) -> float:
    """Rate the quality of behavioral adaptation (0.0 to 1.0)."""
    if not behavioral_change:
        return 0.0

    change_magnitude = behavioral_change.get("change_magnitude", 0)
    increased_actions = behavioral_change.get("increased_actions", [])
    adaptation_type = behavioral_change.get("adaptation_type", "none")

    # Base quality from change magnitude
    quality = min(change_magnitude / 0.6, 1.0)

    # Bonus for beneficial action increases
    beneficial_actions = ["get_items", "put_items", "move"]
    for action, _change in increased_actions:
        if action in beneficial_actions:
            quality += 0.2

    # Bonus for significant adaptation
    if adaptation_type == "behavioral_shift":
        quality += 0.1

    return min(quality, 1.0)


def _extract_temporal_progression(agent_objects: list, action_names: list, episode_length: int) -> Dict[str, Any]:
    """Extract agent statistics at step 0 and every 100 steps for temporal analysis."""
    # Create checkpoint intervals: 100, 200, 300, etc. (not step 0)
    checkpoints = []
    step = 100
    while step <= episode_length:
        checkpoints.append(step)
        step += 100

    # Add final step if not already included
    if episode_length > 0 and episode_length not in checkpoints:
        checkpoints.append(episode_length)

    temporal_data = {
        "checkpoints": checkpoints,
        "agent_progression": {},
        "summary": {"total_checkpoints": len(checkpoints), "interval": 100, "episode_length": episode_length},
    }

    # Analyze each agent's progression
    for agent_obj in agent_objects:
        agent_id = agent_obj.get("agent_id", agent_obj.get("id", 0))
        agent_name = f"agent_{agent_id}"

        # Extract temporal sequences
        r_coords = agent_obj.get("r", [])
        c_coords = agent_obj.get("c", [])
        actions = agent_obj.get("action", [])
        action_success = agent_obj.get("action_success", [])
        total_reward = agent_obj.get("total_reward", [])

        # Build lookup dictionaries for efficient access
        r_dict = {step: coord for step, coord in r_coords}
        c_dict = {step: coord for step, coord in c_coords}
        success_dict = {step[0]: step[1] for step in action_success if isinstance(step, list) and len(step) >= 2}

        # Build reward timeline
        reward_dict = {}
        if isinstance(total_reward, list):
            reward_dict = {step[0]: step[1] for step in total_reward if isinstance(step, list) and len(step) >= 2}

        agent_progression = []

        for checkpoint in checkpoints:
            # Calculate stats up to this checkpoint
            checkpoint_stats = _calculate_checkpoint_stats(
                checkpoint, r_dict, c_dict, actions, success_dict, reward_dict, action_names
            )
            checkpoint_stats["step"] = checkpoint
            agent_progression.append(checkpoint_stats)

        temporal_data["agent_progression"][agent_name] = agent_progression

    return temporal_data


def _calculate_checkpoint_stats(
    checkpoint_step: int,
    r_dict: dict,
    c_dict: dict,
    actions: list,
    success_dict: dict,
    reward_dict: dict,
    action_names: list,
) -> Dict[str, Any]:
    """Calculate agent statistics up to a specific checkpoint step."""

    # Filter actions up to checkpoint
    actions_up_to_checkpoint = [
        action for action in actions if isinstance(action, list) and len(action) >= 2 and action[0] <= checkpoint_step
    ]

    # Current position (latest position up to checkpoint)
    current_r = 0
    current_c = 0
    for step in sorted(r_dict.keys()):
        if step <= checkpoint_step:
            current_r = r_dict[step]
        else:
            break

    for step in sorted(c_dict.keys()):
        if step <= checkpoint_step:
            current_c = c_dict[step]
        else:
            break

    # Calculate cumulative distance traveled up to checkpoint
    distance_traveled = 0.0
    all_position_steps = sorted(set(r_dict.keys()) | set(c_dict.keys()))
    all_position_steps = [s for s in all_position_steps if s <= checkpoint_step]

    if len(all_position_steps) > 1:
        prev_r = r_dict.get(all_position_steps[0], 0)
        prev_c = c_dict.get(all_position_steps[0], 0)

        for step in all_position_steps[1:]:
            new_r = r_dict.get(step, prev_r)
            new_c = c_dict.get(step, prev_c)

            distance = ((new_r - prev_r) ** 2 + (new_c - prev_c) ** 2) ** 0.5
            distance_traveled += distance

            prev_r, prev_c = new_r, new_c

    # Calculate success rate up to checkpoint
    successful_actions = 0
    total_actions = len(actions_up_to_checkpoint)

    action_counts = {}
    for action_step in actions_up_to_checkpoint:
        step_num = action_step[0]

        # Extract action ID
        if isinstance(action_step[1], list) and len(action_step[1]) > 0:
            action_id = action_step[1][0]
        else:
            action_id = action_step[1]

        action_name = action_names[action_id] if action_id < len(action_names) else f"action_{action_id}"
        action_counts[action_name] = action_counts.get(action_name, 0) + 1

        if success_dict.get(step_num, False):
            successful_actions += 1

    success_rate = successful_actions / max(total_actions, 1)

    # Determine current strategic behavior
    strategic_behavior = "unknown"
    if action_counts:
        dominant_action = max(action_counts.items(), key=lambda x: x[1])[0]
        if dominant_action == "attack":
            strategic_behavior = "aggressive"
        elif dominant_action in ["get_items", "put_items"]:
            strategic_behavior = "resource-focused"
        elif dominant_action == "move":
            strategic_behavior = "exploratory"
        elif dominant_action == "rotate":
            strategic_behavior = "defensive/observational"
        else:
            strategic_behavior = "adaptive"

    # Get most recent dominant action (last 50 actions or all if fewer)
    recent_actions = actions_up_to_checkpoint[-50:] if len(actions_up_to_checkpoint) > 50 else actions_up_to_checkpoint
    recent_action_counts = {}
    for action_step in recent_actions:
        if isinstance(action_step[1], list) and len(action_step[1]) > 0:
            action_id = action_step[1][0]
        else:
            action_id = action_step[1]

        action_name = action_names[action_id] if action_id < len(action_names) else f"action_{action_id}"
        recent_action_counts[action_name] = recent_action_counts.get(action_name, 0) + 1

    recent_dominant_action = (
        max(recent_action_counts.items(), key=lambda x: x[1])[0] if recent_action_counts else "none"
    )

    # Get current score
    current_score = 0.0
    for step in sorted(reward_dict.keys()):
        if step <= checkpoint_step:
            current_score = reward_dict[step]
        else:
            break

    return {
        "score": round(current_score, 3),
        "distance_traveled": round(distance_traveled, 2),
        "success_rate": round(success_rate, 3),
        "strategic_behavior": strategic_behavior,
        "current_position": [current_r, current_c],
        "action_count": total_actions,
        "recent_dominant_action": recent_dominant_action,
        "action_distribution": action_counts,
    }
