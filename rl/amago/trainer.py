from functools import partial

from amago.experiment import Experiment
import hydra
from omegaconf import OmegaConf
from agent.policy_store import PolicyStore

from amago.envs import AMAGOEnv
from amago.cli_utils import *

from rl.amago.cnn import MettaAmagoCNN
from rl.amago.wrapper import SingleAgentSpaceWrapper


class AmagoTrainer:
    def __init__(self, 
                 cfg: OmegaConf,
                 wandb_run,
                 policy_store: PolicyStore,
                 **kwargs) -> None:

        config = {
            # dictionary that sets default value for kwargs of classes that are marked as `gin.configurable`
            # see https://github.com/google/gin-config for more information. For example:
            # "amago.nets.traj_encoders.TformerTrajEncoder.attention_type": amago.nets.transformer.FlashAttention,
        }

        prototype_env = hydra.utils.instantiate(cfg.env, render_mode=None)
        grid_features = prototype_env.grid_features
        num_agents = prototype_env.num_agents
        
        cnn_type = partial(MettaAmagoCNN, grid_features=grid_features)

        tstep_encoder_type = switch_tstep_encoder(
            config,
            arch="cnn",
            cnn_type=cnn_type,
            channels_first=True,
            drqv2_aug=True,
        )
        
        traj_encoder_type = switch_traj_encoder(
            config,
            arch=cfg.trainer.traj_encoder,
            memory_size=cfg.trainer.memory_size,
            layers=cfg.trainer.memory_layers,
        )

        agent_type = switch_agent(
            config,
            cfg.trainer.agent_type,
        )

        for param, val in config.items():
            gin.bind_parameter(param, val)
        gin.finalize()

        make_train_env = lambda: AMAGOEnv(
            SingleAgentSpaceWrapper(hydra.utils.instantiate(cfg.env, render_mode=None)),
            batched_envs=num_agents,
            env_name='Metta',
        )
        
        self.experiment = Experiment(
            run_name="test",
            dset_name="test",
            dset_root="./amago_buffers",
            wandb_group_name="test",
            env_mode='already_vectorized',
            log_to_wandb=True,
            parallel_actors=num_agents,
            exploration_wrapper_type=EpsilonGreedy,
            agent_type=agent_type,
            tstep_encoder_type=tstep_encoder_type,
            traj_encoder_type=traj_encoder_type,
            make_train_env=make_train_env,
            make_val_env=make_train_env,
            max_seq_len=cfg.trainer.max_seq_len,
            traj_save_len=cfg.trainer.max_seq_len * 8,
            dset_max_size=cfg.trainer.dset_max_size,
            dloader_workers=1,
            batch_size=cfg.trainer.batch_size,
            epochs=cfg.trainer.epochs,
            train_timesteps_per_epoch=cfg.trainer.training_timesteps_per_epoch,
            train_batches_per_epoch=cfg.trainer.train_batches_per_epoch,
            start_learning_at_epoch=cfg.trainer.start_learning_at_epoch,
            val_interval=cfg.trainer.val_interval,
            ckpt_interval=cfg.trainer.ckpt_interval,
            mixed_precision=cfg.trainer.mixed_precision,
        )

    def train(self):
        self.experiment.start()
        self.experiment.learn()

    def close(self):
        pass
