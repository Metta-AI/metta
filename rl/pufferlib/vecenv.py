import copy
from omegaconf import OmegaConf
from omegaconf import OmegaConf, DictConfig
import pufferlib
import pufferlib.utils
import pufferlib.vector
import hydra
from mettagrid.mettagrid_env import MettaGridEnv, MettaGridEnvSelector

def make_env_func(cfg: DictConfig, buf=None, render_mode='rgb_array'):
    return MettaGridEnvSelector(cfg, buf=buf, render_mode=render_mode)

def make_env_func_single_env(cfg: DictConfig, buf=None, render_mode='rgb_array'):
    return MettaGridEnv(cfg, buf=buf, render_mode=render_mode)


def make_vecenv(
    env_cfg: OmegaConf | list[OmegaConf],
    vectorization: str,
    num_envs=1,
    batch_size=None,
    num_workers=1,
    render_mode=None,
    **kwargs
):

    if not isinstance(env_cfg, list):
        return make_vecenv_single_env(env_cfg, vectorization, num_envs, batch_size, num_workers, render_mode, **kwargs)

    vec = vectorization
    if vec == 'serial' or num_workers == 1:
        vec = pufferlib.vector.Serial
    elif vec == 'multiprocessing':
        vec = pufferlib.vector.Multiprocessing
    elif vec == 'ray':
        vec = pufferlib.vector.Ray
    else:
        raise ValueError('Invalid --vector (serial/multiprocessing/ray).')

    vecenv_args = dict(
        env_kwargs=dict(cfg = env_cfg, render_mode=render_mode),
        num_envs=num_envs,
        num_workers=num_workers,
        batch_size=batch_size or num_envs,
        backend=vec,
        **kwargs
    )

    vecenv = pufferlib.vector.make(make_env_func, **vecenv_args)
    return vecenv

def make_vecenv_single_env(
    env_cfg: OmegaConf,
    vectorization: str,
    num_envs=1,
    batch_size=None,
    num_workers=1,
    render_mode=None,
    **kwargs
):

    vec = vectorization
    if vec == 'serial' or num_workers == 1:
        vec = pufferlib.vector.Serial
    elif vec == 'multiprocessing':
        vec = pufferlib.vector.Multiprocessing
    elif vec == 'ray':
        vec = pufferlib.vector.Ray
    else:
        raise ValueError('Invalid --vector (serial/multiprocessing/ray).')

    vecenv_args = dict(
        env_kwargs=dict(cfg = dict(**env_cfg), render_mode=render_mode),
        num_envs=num_envs,
        num_workers=num_workers,
        batch_size=batch_size or num_envs,
        backend=vec,
        **kwargs
    )

    vecenv = pufferlib.vector.make(make_env_func_single_env, **vecenv_args)
    return vecenv
