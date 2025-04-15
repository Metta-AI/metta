# Generate a graphical trace of multiple runs.

import json
import os
import zlib

import boto3
import wandb
from omegaconf import OmegaConf

from metta.agent.policy_store import PolicyRecord
from metta.rl.pufferlib.simulator import Simulator
from metta.rl.wandb.wandb_context import WandbContext

class ReplayHelper:
    """Helper class for generating and uploading replays."""

    def __init__(self, cfg: OmegaConf, env_cfg: OmegaConf, policy_record: PolicyRecord, wandb_run: WandbContext):
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.policy_record = policy_record
        self.wandb_run = wandb_run

        self.s3_client = boto3.client("s3")

    def _add_sequence_key(self, grid_object: dict, key: str, step: int, value):
        """Add a key to the replay that is a sequence of values."""
        if key not in grid_object:
            # Add new key.
            grid_object[key] = [[step, value]]
        else:
            # Only add new entry if it has changed:
            if grid_object[key][-1][1] != value:
                grid_object[key].append([step, value])

    def generate_replay(self, replay_path: str):
        """Generate a replay and save it to a file."""
        simulator = Simulator(self.cfg, self.env_cfg, self.policy_record)

        grid_objects = []

        replay = {
            "version": 1,
            "action_names": simulator.env.action_names(),
            "object_types": [],
            "map_size": [simulator.env.map_width, simulator.env.map_height],
            "num_agents": simulator.num_agents,
            "max_steps": simulator.num_steps,
            "grid_objects": grid_objects,
        }

        replay["object_types"] = simulator.env.object_type_names()

        step = 0
        while not simulator.done():
            actions = simulator.actions()

            actions_array = actions.cpu().numpy()

            for i, grid_object in enumerate(simulator.grid_objects()):
                if len(grid_objects) <= i:
                    # Add new grid object.
                    grid_objects.append({})
                for key, value in grid_object.items():
                    self._add_sequence_key(grid_objects[i], key, step, value)

                if "agent_id" in grid_object:
                    agent_id = grid_object["agent_id"]
                    self._add_sequence_key(grid_objects[i], "action", step, actions_array[agent_id].tolist())
                    self._add_sequence_key(
                        grid_objects[i], "action_success", step, bool(simulator.env.action_success[agent_id])
                    )
                    self._add_sequence_key(grid_objects[i], "reward", step, simulator.rewards[agent_id].item())
                    self._add_sequence_key(
                        grid_objects[i], "total_reward", step, simulator.total_rewards[agent_id].item()
                    )

            simulator.step(actions)

            step += 1

        replay["max_steps"] = step

        # Trim value changes to make them more compact.
        for grid_object in grid_objects:
            for key, changes in grid_object.items():
                if len(changes) == 1:
                    grid_object[key] = changes[0][1]

        # Compress it with deflate.
        replay_data = json.dumps(replay)  # Convert to JSON string
        replay_bytes = replay_data.encode("utf-8")  # Encode to bytes
        compressed_data = zlib.compress(replay_bytes)  # Compress the bytes

        # Make sure the directory exists.
        os.makedirs(os.path.dirname(replay_path), exist_ok=True)

        with open(replay_path, "wb") as f:
            f.write(compressed_data)

    def upload_replay(self, replay_path: str, replay_url: str, epoch: int):
        """Upload the replay to S3 and log the link to WandB."""
        s3_bucket = "softmax-public"
        self.s3_client.upload_file(
            Filename=replay_path, Bucket=s3_bucket, Key=replay_url, ExtraArgs={"ContentType": "application/x-compress"}
        )
        link = f"https://{s3_bucket}.s3.us-east-1.amazonaws.com/{replay_url}"

        # Log the link to WandB
        player_url = "https://metta-ai.github.io/mettagrid/?replayUrl=" + link
        link_summary = {"replays/link": wandb.Html(f'<a href="{player_url}">MetaScope Replay (Epoch {epoch})</a>')}
        self.wandb_run.log(link_summary)

    def generate_and_upload_replay(self, epoch: int):
        """Generate a replay and upload it to S3 and log the link to WandB."""
        replay_path = f"{self.cfg.run_dir}/replays/replay.{epoch}.json.z"
        self.generate_replay(replay_path)

        replay_url = f"replays/{self.cfg.run}/replay.{epoch}.json.z"
        self.upload_replay(replay_path, replay_url, epoch)
