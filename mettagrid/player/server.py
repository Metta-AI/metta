import asyncio
import logging

import hydra
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import DictConfig

import mettagrid.player.replays as replays


class App(FastAPI):
    cfg: DictConfig


app = App()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.get("/", response_class=HTMLResponse)
async def get_client():
    try:
        with open("mettagrid/player/index.html", "r") as file:
            html_content = file.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail="Client HTML file not found") from err


@app.get("/style.css")
async def get_style_css():
    try:
        with open("mettagrid/player/style.css", "r") as file:
            css_content = file.read()
        return HTMLResponse(content=css_content, media_type="text/css")
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail="Client HTML file not found") from err


# Mount a directory for static files
app.mount("/data", StaticFiles(directory="mettagrid/player/data"), name="data")
app.mount("/dist", StaticFiles(directory="mettagrid/player/dist"), name="dist")


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
):
    await websocket.accept()

    async def send_message(**kwargs):
        await websocket.send_json(kwargs)

    logger.info("Received websocket connection!")
    await send_message(message="Connecting!")

    replay = replays.generate_replay(app.cfg)
    await send_message(replay=replay)

    while True:
        # Receive action from client
        await send_message(message="Step!")

        # Wait 1 second
        await asyncio.sleep(1)


if __name__ == "__main__":
    import uvicorn

    @hydra.main(version_base=None, config_path="../../configs", config_name="replay_job")
    def main(cfg):
        app.cfg = cfg

        uvicorn.run(app, host="0.0.0.0", port=8000)

    main()
