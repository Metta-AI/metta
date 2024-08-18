import os

import cv2
import numpy as np
from huggingface_hub import HfApi, Repository, repocard, upload_folder

from sample_factory.utils.typing import Config
from sample_factory.utils.utils import log, project_tmp_dir
import imageio

MIN_FRAME_SIZE = 180


def generate_replay_video(path: str, frames: list, fps: int):
    if not path.endswith(".mp4"):
        path += ".mp4"

    tmp_name = os.path.join(project_tmp_dir(), os.path.basename(path))
    if frames[0].shape[0] == 3:
        frame_size = (frames[0].shape[2], frames[0].shape[1])
    else:
        frame_size = (frames[0].shape[1], frames[0].shape[0])
    resize = False

    if min(frame_size) < MIN_FRAME_SIZE:
        resize = True
        scaling_factor = MIN_FRAME_SIZE / min(frame_size)
        frame_size = (int(frame_size[0] * scaling_factor), int(frame_size[1] * scaling_factor))

    video = cv2.VideoWriter(tmp_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    for frame in frames:
        if frame.shape[0] == 3:
            frame = frame.transpose(1, 2, 0)
        if resize:
            frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
        video.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    video.release()
    os.system(f"ffmpeg -y -i {tmp_name} -vcodec libx264 {path}")
    log.debug(f"Replay video saved to {path}!")

def generate_replay_gif(path: str, frames: list, fps: int):
    imageio.mimsave(path, frames, fps=fps)
    log.debug(f"Replay gif saved to {path}!")

