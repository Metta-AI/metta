import asyncio
import logging
import webbrowser
from pathlib import Path

import numpy as np
import torch as th
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import DictConfig

import mettascope.replays as replays
from metta.common.util.constants import DEV_METTASCOPE_FRONTEND_URL
from metta.mettagrid.grid_object_formatter import format_grid_object
from metta.mettagrid.mettagrid_c import PackedCoordinate
from metta.util.metta_script import metta_script

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomStaticFiles(StaticFiles):
    """StaticFiles that disables caching for specific file extensions and sets custom content types."""

    def __init__(self, *args, no_cache_extensions=None, custom_content_types=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.no_cache_extensions = no_cache_extensions or {".js", ".json"}
        self.custom_content_types = custom_content_types or {}

    def file_response(
        self,
        full_path,
        stat_result,
        scope,
        status_code: int = 200,
    ):
        """Override file_response to selectively disable caching and set custom content types."""
        file_path = Path(full_path)
        file_ext = file_path.suffix.lower()

        # Create the response
        response = FileResponse(full_path, status_code=status_code, stat_result=stat_result)

        # Handle custom content types - set header directly to avoid FastAPI overriding
        if file_ext in self.custom_content_types:
            response.headers["content-type"] = self.custom_content_types[file_ext]

        # Handle caching
        if file_ext in self.no_cache_extensions:
            response.headers["Cache-Control"] = "no-store"

        return response


def clear_memory(sim: replays.Simulation, what: str, agent_id: int) -> None:
    """Clear the memory of the policy."""
    policy_state = sim.get_policy_state()

    if policy_state is None or policy_state.lstm_c is None or policy_state.lstm_h is None:
        logger.error("No policy state to clear")
        return

    if what == "0":
        policy_state.lstm_c[:, agent_id, :].zero_()
        policy_state.lstm_h[:, agent_id, :].zero_()
    elif what == "1":
        policy_state.lstm_c[:, agent_id, :].fill_(1)
        policy_state.lstm_h[:, agent_id, :].fill_(1)
    elif what == "random":
        policy_state.lstm_c[:, agent_id, :].normal_(mean=0, std=1)
        policy_state.lstm_h[:, agent_id, :].normal_(mean=0, std=1)


def copy_memory(sim: replays.Simulation, agent_id: int) -> tuple[list[float], list[float]]:
    """Copy the memory of the policy."""
    policy_state = sim.get_policy_state()
    if policy_state is None or policy_state.lstm_c is None or policy_state.lstm_h is None:
        logger.error("No policy state to copy")
        return [], []

    # Copy the memory of the policy.
    lstm_c = policy_state.lstm_c[:, agent_id, :].clone()
    lstm_h = policy_state.lstm_h[:, agent_id, :].clone()
    return lstm_c.tolist(), lstm_h.tolist()


def paste_memory(sim: replays.Simulation, agent_id: int, memory: tuple[list[float], list[float]]):
    """Paste the memory of the policy."""
    policy_state = sim.get_policy_state()
    if policy_state is None or policy_state.lstm_c is None or policy_state.lstm_h is None:
        logger.error("No policy state to paste")
        return

    [lstm_c, lstm_h] = memory
    policy_state.lstm_c[:, agent_id, :] = th.tensor(lstm_c)
    policy_state.lstm_h[:, agent_id, :] = th.tensor(lstm_h)


def make_app(cfg: DictConfig):
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    async def get_client():
        try:
            with open("mettascope/index.html", "r") as file:
                html_content = file.read()
            return HTMLResponse(content=html_content)
        except FileNotFoundError as err:
            raise HTTPException(status_code=404, detail="Client HTML file not found") from err

    @app.get("/{path}.css")
    async def get_style_css(path: str):
        if "/" in path or "." in path:
            raise HTTPException(status_code=400, detail="Path must not contain '/' or '.'")
        try:
            with open(f"mettascope/{path}.css", "r") as file:
                css_content = file.read()
            return HTMLResponse(content=css_content, media_type="text/css")
        except FileNotFoundError as err:
            raise HTTPException(status_code=404, detail="Client HTML file not found") from err

    # Mount a directory for static files
    app.mount("/data", StaticFiles(directory="mettascope/data"), name="data")
    app.mount(
        "/dist",
        CustomStaticFiles(directory="mettascope/dist", no_cache_extensions={".js", ".json", ".css"}),
        name="dist",
    )
    app.mount(
        "/local", CustomStaticFiles(directory=".", custom_content_types={".z": "application/x-compress"}), name="local"
    )

    # Direct favicon.ico to the data/ui dir.
    @app.get("/favicon.ico")
    async def get_favicon():
        return FileResponse("mettascope/data/ui/logo@2x.png")

    @app.websocket("/ws")
    async def websocket_endpoint(
        websocket: WebSocket,
    ):
        await websocket.accept()

        async def send_message(**kwargs):
            await websocket.send_json(kwargs)

        logger.info("Received websocket connection!")
        await send_message(type="message", message="Connecting!")

        # Create a simulation that we are going to play.
        sim = replays.create_simulation(cfg)
        sim.start_simulation()
        env = sim.get_env()
        replay = sim.get_replay()

        await send_message(type="replay", replay=replay)

        current_step = 0
        action_message = None
        actions = np.zeros((env.num_agents, 2))
        total_rewards = np.zeros(env.num_agents)

        # Visual overlay state for play mode only.
        overlay_enabled: bool = False
        overlay_agent_id: int = 0
        overlay_layer_id: int = 0

        # Build and send the available observation layers once.
        try:
            feature_spec = env.get_observation_features()
            # feature_spec: name -> {"id": int, ...}
            layers = [{"id": int(spec["id"]), "name": str(name)} for name, spec in feature_spec.items() if "id" in spec]
            # Sort by id for a stable order.
            layers.sort(key=lambda x: x["id"])  # type: ignore
            await send_message(type="visual_layers", layers=layers)
        except Exception as e:
            logger.warning(f"Failed to fetch observation features for visual overlay: {e}")

        def extract_visual_grid(agent_id: int, layer_id: int):
            """
            Extract a (height, width) grid for the given agent and feature layer from the observation buffer.

            Observations are token-encoded as uint8 triplets per token: [packed_coord, feature_id, feature_value].
            packed_coord of 0xFF indicates padding/empty. Coordinates are within the observation window
            centered on the agent with size (obs_height, obs_width).
            """
            obs = (
                env.observations
            )  # The observation buffer has shape [num_agents, tokens_per_agent, 3] with dtype uint8.
            height = int(env.obs_height)
            width = int(env.obs_width)
            grid = np.zeros((height, width), dtype=np.int32)

            if agent_id < 0 or agent_id >= obs.shape[0]:
                return grid, width, height, 0, 0

            tokens = obs[agent_id]
            # Tokens have shape [M, 3].
            for token in tokens:
                packed = int(token[0])
                if PackedCoordinate.is_empty(packed):
                    # Values beyond this token are padding.
                    break
                fid = int(token[1])
                if fid != int(layer_id):
                    continue
                coords = PackedCoordinate.unpack(packed)
                if coords is None:
                    continue
                row, col = int(coords[0]), int(coords[1])
                if 0 <= row < height and 0 <= col < width:
                    grid[row, col] = int(token[2])

            nonzero = grid[grid != 0]
            vmin = int(nonzero.min()) if nonzero.size > 0 else 0
            vmax = int(nonzero.max()) if nonzero.size > 0 else 0
            return grid, width, height, vmin, vmax

        async def send_replay_step():
            grid_objects = []
            for i, grid_object in enumerate(env.grid_objects.values()):
                if len(grid_objects) <= i:
                    grid_objects.append({})

                if "agent_id" in grid_object:
                    agent_id = grid_object["agent_id"]
                    total_rewards[agent_id] += env.rewards[agent_id]

                update_object = format_grid_object(grid_object, actions, env.action_success, env.rewards, total_rewards)

                grid_objects[i] = update_object

            await send_message(type="replay_step", replay_step={"step": current_step, "objects": grid_objects})

            # If the visual overlay is enabled, send the current agent/layer grid as well.
            if overlay_enabled:
                grid, width, height, vmin, vmax = extract_visual_grid(overlay_agent_id, overlay_layer_id)
                await send_message(
                    type="visual_grid",
                    agentId=int(overlay_agent_id),
                    layerId=int(overlay_layer_id),
                    width=int(width),
                    height=int(height),
                    values=grid.reshape(-1).tolist(),
                    valueRange={"min": int(vmin), "max": int(vmax)},
                )

        # Send the first replay step.
        await send_replay_step()

        while True:
            # Main message loop.

            try:
                message = await websocket.receive_json()
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected by client")
                break

            if message["type"] == "action":
                action_message = message

            elif message["type"] == "advance":
                action_message = None

            elif message["type"] == "visual_overlay_enable":
                try:
                    overlay_enabled = bool(message.get("enabled", False))
                except Exception:
                    overlay_enabled = False
                # Send an immediate grid snapshot so the client can render without waiting for a step.
                if overlay_enabled:
                    grid, width, height, vmin, vmax = extract_visual_grid(overlay_agent_id, overlay_layer_id)
                    await send_message(
                        type="visual_grid",
                        agentId=int(overlay_agent_id),
                        layerId=int(overlay_layer_id),
                        width=int(width),
                        height=int(height),
                        values=grid.reshape(-1).tolist(),
                        valueRange={"min": int(vmin), "max": int(vmax)},
                    )
                continue

            elif message["type"] == "visual_set_agent":
                try:
                    overlay_agent_id = int(message.get("agent_id", 0))
                except Exception:
                    overlay_agent_id = 0
                # If overlay is enabled, send an immediate grid snapshot for the new agent.
                if overlay_enabled:
                    grid, width, height, vmin, vmax = extract_visual_grid(overlay_agent_id, overlay_layer_id)
                    await send_message(
                        type="visual_grid",
                        agentId=int(overlay_agent_id),
                        layerId=int(overlay_layer_id),
                        width=int(width),
                        height=int(height),
                        values=grid.reshape(-1).tolist(),
                        valueRange={"min": int(vmin), "max": int(vmax)},
                    )
                continue

            elif message["type"] == "visual_set_layer":
                try:
                    overlay_layer_id = int(message.get("layer_id", 0))
                except Exception:
                    overlay_layer_id = 0
                # If overlay is enabled, send an immediate grid snapshot for the new layer.
                if overlay_enabled:
                    grid, width, height, vmin, vmax = extract_visual_grid(overlay_agent_id, overlay_layer_id)
                    await send_message(
                        type="visual_grid",
                        agentId=int(overlay_agent_id),
                        layerId=int(overlay_layer_id),
                        width=int(width),
                        height=int(height),
                        values=grid.reshape(-1).tolist(),
                        valueRange={"min": int(vmin), "max": int(vmax)},
                    )
                continue

            elif message["type"] == "clear_memory":
                clear_memory(sim, message["what"], message["agent_id"])
                continue

            elif message["type"] == "copy_memory":
                memory = copy_memory(sim, message["agent_id"])
                await send_message(
                    type="memory_copied",
                    memory=memory,
                )
                continue

            elif message["type"] == "paste_memory":
                paste_memory(sim, message["agent_id"], message["memory"])
                continue

            else:
                raise ValueError(f"Unknown type: {message['type']}")

            if current_step < sim._vecenv.driver_env.max_steps:
                await send_message(type="message", message="Step!")

                actions = sim.generate_actions()
                if action_message is not None:
                    agent_id = action_message["agent_id"]
                    actions[agent_id][0] = action_message["action_id"]
                    actions[agent_id][1] = action_message["action_param"]
                sim.step_simulation(actions)

                await send_replay_step()
                current_step += 1

            # yield control to other coroutines
            await asyncio.sleep(0)

        sim.end_simulation()

    return app


def run(cfg: DictConfig, open_url: str | None = None):
    app = make_app(cfg)

    if open_url:
        server_url = DEV_METTASCOPE_FRONTEND_URL

        @app.on_event("startup")
        async def _open_browser():
            webbrowser.open(f"{server_url}{open_url}")

    uvicorn.run(app, host="0.0.0.0", port=8000)


metta_script(run, "replay_job")
