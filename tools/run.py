import os
import hydra
from omegaconf import OmegaConf
from rich import traceback
import util.replay as replay
import signal # Aggressively exit on ctrl+c
from rl.wandb.wandb import init_wandb

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    traceback.install(show_locals=False)
    print(OmegaConf.to_yaml(cfg))
    framework = hydra.utils.instantiate(cfg.framework, cfg, _recursive_=False)
    if cfg.wandb.track:
        framework.wandb = init_wandb(cfg)

    try:
        if cfg.cmd == "train":
            framework.train()

        if cfg.cmd == "evaluate":
            result = framework.evaluate()
            if cfg.eval.video_path is not None:
                replay.generate_replay_video(cfg.eval.video_path, result.frames, cfg.eval.fps)
            if cfg.eval.gif_path is not None:
                replay.generate_replay_gif(cfg.eval.gif_path, result.frames, cfg.eval.fps)

        if cfg.cmd == "play":
            result = framework.evaluate()

    except KeyboardInterrupt:
        os._exit(0)

if __name__ == "__main__":
    main()
