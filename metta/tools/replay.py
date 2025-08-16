# Generate a replay file that can be used in MettaScope to visualize a single run.

import logging
import platform
from urllib.parse import quote

import mettascope.server as server
from metta.agent.policy_store import PolicyStore
from metta.common.util.constants import DEV_METTASCOPE_FRONTEND_URL
from metta.common.util.tool import Tool
from metta.common.wandb.wandb_context import WandbConfig, WandbConfigOff
from metta.rl.system_config import SystemConfig
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationConfig

logger = logging.getLogger(__name__)


# TODO: This job can be replaced with sim now that Simulations create replays
class ReplayTool(Tool):
    system: SystemConfig = SystemConfig()
    wandb: WandbConfig = WandbConfigOff()
    sim: SimulationConfig
    policy_uri: str | None = None
    selector_type: str = "latest"
    replay_dir: str = "./train_dir/replays"
    stats_dir: str = "./train_dir/stats"
    open_browser_on_start: bool = True

    def invoke(self) -> None:
        # Create policy store directly without WandbContext
        policy_store = PolicyStore.create(
            device=self.system.device,
            wandb_config=self.wandb,
            data_dir=self.system.data_dir,
            wandb_run=None,
        )

        # Create simulation using the helper method with explicit parameters
        sim = Simulation.create(
            sim_config=self.sim,
            policy_store=policy_store,
            device=self.system.device,
            vectorization=self.system.vectorization,
            stats_dir=self.stats_dir,
            replay_dir=self.replay_dir,
            policy_uri=self.policy_uri,
            run_name="replay_run",
        )

        result = sim.simulate()
        key, version = result.stats_db.key_and_version(sim.policy_record)
        replay_url = result.stats_db.get_replay_urls(key, version)[0]

        open_browser(replay_url, self)


def open_browser(replay_url: str, cfg: ReplayTool) -> None:
    # Only on macos open a browser to the replay
    if platform.system() == "Darwin":
        if not replay_url.startswith("http"):
            # Remove ./ prefix if it exists
            clean_path = replay_url.removeprefix("./")
            local_url = f"{DEV_METTASCOPE_FRONTEND_URL}/local/{clean_path}"
            full_url = f"/?replayUrl={quote(local_url)}"

            # Run a metascope server that serves the replay
            # Import PlayToolConfig to use with the server
            from metta.tools.play import PlayTool

            # Create a PlayTool from ReplayTool (they have the same fields)
            play_cfg = PlayTool(
                system=cfg.system,
                wandb=cfg.wandb,
                sim=cfg.sim,
                policy_uri=cfg.policy_uri,
                selector_type=cfg.selector_type,
                replay_dir=cfg.replay_dir,
                stats_dir=cfg.stats_dir,
                open_browser_on_start=cfg.open_browser_on_start,
            )

            if cfg.open_browser_on_start:
                server.run(play_cfg, open_url=full_url)
            else:
                logger.info(f"Enter MettaGrid @ {full_url}")
                server.run(play_cfg)
