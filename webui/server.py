import json
import logging

import hydra
from fastapi import FastAPI, HTTPException, Query, WebSocket
from fastapi.responses import HTMLResponse
from hydra import compose, initialize

from mettagrid.mettagrid_env import MettaGridEnv
from mettagrid.tests.utils import generate_valid_random_actions

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.get("/", response_class=HTMLResponse)
async def get_client():
    try:
        with open("webui/client.html", "r") as file:
            html_content = file.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail="Client HTML file not found") from err


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    env: str = Query(..., description="Name of the environment to use"),
    args: str = Query("", description="Hydra Args"),
):
    await websocket.accept()

    async def send_message(message: str):
        await websocket.send_json({"message": message})

    logger.info(f"Received websocket connection for environment {env} with args {args}")
    await send_message(f"Connecting to environment: {env}")

    # Initialize Hydra and load the specified configuration
    try:
        with initialize(config_path="../configs/"):
            overrides = args.split() if args else []
            cfg = compose(
                config_name="config",
                overrides=[
                    f"env=mettagrid/{env}",
                    "cmd=webui",
                    "experiment=webui",
                ]
                + overrides,
            )
        logger.info(f"Loaded configuration for {env}")
        await send_message(f"Loaded configuration for {env}")
    except Exception as e:
        error_message = f"Failed to load configuration: {str(e)}"
        logger.error(error_message)
        await websocket.send_json({"error": error_message})
        await send_message(error_message)
        await websocket.close()
        return

    try:
        step_count = 0
        metta_grid_env: MettaGridEnv = hydra.utils.instantiate(cfg.env, render_mode="websocket")  # type: ignore
        assert isinstance(metta_grid_env, MettaGridEnv)

        obs, infos = metta_grid_env.reset()
        logger.info("Environment initialized and reset")
        await send_message("Environment initialized and reset")
        await websocket.send_json(
            {
                "message": "Initial state",
                # "observation": obs,
                "info": infos,
                "objects": metta_grid_env.grid_objects,
            }
        )

        while True:
            data = await websocket.receive_text()
            try:
                actions = json.loads(data)
                # Validate client actions here if needed
            except (json.JSONDecodeError, ValueError):
                # Fall back to random actions only if there's an issue with client data
                logger.warning("Invalid actions received, falling back to random actions")
                actions = generate_valid_random_actions(metta_grid_env, metta_grid_env.num_agents)

            # Step the environment
            obs, rewards, terminated, truncated, infos = metta_grid_env.step(actions)

            # Send results back to client
            await websocket.send_json(
                {
                    # "observation": obs,
                    # "rewards": list(rewards),
                    # "terminated": terminated,
                    # "truncated": truncated,
                    "infos": infos,
                    "objects": metta_grid_env.grid_objects,
                }
            )

            step_count += 1
            logger.info(f"Completed step {step_count}")
            await send_message(f"Completed step {step_count}")

            if terminated or truncated:
                logger.info("Episode finished")
                await send_message("Episode finished")
                break

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logger.error(error_message)
        await websocket.send_json({"error": error_message})
        await send_message(error_message)
    finally:
        logger.info("Closing WebSocket connection")
        await send_message("Closing WebSocket connection")
        await websocket.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
