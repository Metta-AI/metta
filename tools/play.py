# Starts a websocket server that allows you to play as a metta agent.
import webbrowser

import hydra
import uvicorn

import mettascope.server as server


@hydra.main(version_base=None, config_path="../configs", config_name="replay_job")
def main(cfg):
    app = server.make_app(cfg)

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    webbrowser.open("http://localhost:8000/?wsUrl=%2Fws")
    main()
