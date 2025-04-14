import hydra
import pufferlib
import pufferlib.utils
import pufferlib.vector
from omegaconf import DictConfig, OmegaConf

def make_env_func(cfg: DictConfig, buf=None, render_mode="rgb_array"):
    return hydra.utils.instantiate(cfg, cfg, render_mode=render_mode, buf=buf)


def make_vecenv(
    env_cfg: OmegaConf, vectorization: str, num_envs=1, batch_size=None, num_workers=1, render_mode=None, **kwargs
):
    vec = vectorization
    if vec == "serial" or num_workers == 1:
        vec = pufferlib.vector.Serial
    elif vec == "multiprocessing":
        vec = pufferlib.vector.Multiprocessing
    elif vec == "ray":
        vec = pufferlib.vector.Ray
    else:
        raise ValueError("Invalid --vector (serial/multiprocessing/ray).")

    vecenv_args = dict(
        env_kwargs=dict(cfg=env_cfg, render_mode=render_mode),
        num_envs=num_envs,
        num_workers=num_workers,
        batch_size=batch_size or num_envs,
        backend=vec,
        **kwargs,
    )

    vecenv = pufferlib.vector.make(make_env_func, **vecenv_args)
    return vecenv
