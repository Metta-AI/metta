# Starts a websocket server that allows you to play as a metta agent.
<<<<<<< HEAD
import webbrowser
=======
>>>>>>> 13c12a2fdf120e435aa056c95de09aa7ccaa5a87

import hydra
import uvicorn

import mettascope.server as server


@hydra.main(version_base=None, config_path="../configs", config_name="replay_job")
def main(cfg):
<<<<<<< HEAD
    server.app.cfg = cfg

    uvicorn.run(server.app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    webbrowser.open("http://localhost:8000/?wsUrl=%2Fws")
=======
    server.run(cfg, open_url="?wsUrl=%2Fws")


if __name__ == "__main__":
>>>>>>> 13c12a2fdf120e435aa056c95de09aa7ccaa5a87
    main()
