import asyncio
import logging
import webbrowser

import hydra
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import DictConfig

import mettascope.replays as replays

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    app.mount("/dist", StaticFiles(directory="mettascope/dist"), name="dist")
    app.mount("/local", StaticFiles(directory="mettascope/local"), name="local")

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
        total_rewards = np.zeros(env.num_agents)

        while True:
            # While the client we are sending messages to it.

            if current_step < 1000:
                await send_message(type="message", message="Step!")

                actions = sim.generate_actions()
                if action_message is not None:
                    agent_id = action_message["agent_id"]
                    actions[agent_id][0] = action_message["action"][0]
                    actions[agent_id][1] = action_message["action"][1]
                sim.step_simulation(actions)

                grid_objects = []
                for i, grid_object in enumerate(env.grid_objects.values()):
                    if len(grid_objects) <= i:
                        grid_objects.append({})
                    for key, value in grid_object.items():
                        grid_objects[i][key] = value
                    if "agent_id" in grid_object:
                        agent_id = grid_object["agent_id"]
                        grid_objects[i]["action_success"] = bool(env.action_success[agent_id])
                        grid_objects[i]["action"] = actions[agent_id].tolist()
                        grid_objects[i]["reward"] = env.rewards[agent_id].item()
                        total_rewards[agent_id] += env.rewards[agent_id]
                        grid_objects[i]["total_reward"] = total_rewards[agent_id].item()

                await send_message(type="replay_step", replay_step={"step": current_step, "grid_objects": grid_objects})

                current_step += 1

            if current_step > 1:
                message = await websocket.receive_json()
                if message["type"] == "action":
                    action_message = message
                # yield control to other coroutines
                await asyncio.sleep(0)

        sim.end_simulation()

    return app


def run(cfg: DictConfig, open_url: str | None = None):
    app = make_app(cfg)

    if open_url:
        server_url = "http://localhost:8000"

        @app.on_event("startup")
        async def _open_browser():
            webbrowser.open(f"{server_url}{open_url}")

    uvicorn.run(app, host="0.0.0.0", port=8000)


@hydra.main(version_base=None, config_path="../configs", config_name="replay_job")
def main(cfg):
    run(cfg)


if __name__ == "__main__":
    main()
