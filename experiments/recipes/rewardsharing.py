from typing import Optional, Sequence, cast

import metta.cogworks.curriculum as cc
from metta.common.config import Config
from metta.map.mapgen import MapGen
from metta.map.scene import ChildrenAction, Scene
from metta.map.scenes.room_grid import RoomGrid
from metta.map.types import AreaWhere
from metta.mettagrid.config.envs import make_arena
from metta.mettagrid.mettagrid_config import ActionConfig, ActionsConfig, EnvConfig
from metta.rl.trainer_config import (
    CheckpointConfig,
    EvaluationConfig,
    OptimizerConfig,
    PPOConfig,
    TrainerConfig,
)
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool


class SafePlacementParams(Config):
    """Parameters for safe placement of objects and agents that won't block generator access."""

    objects: dict[str, int] = {}
    agents: int = 0
    prefer_corners: bool = True  # Place objects in corners when possible


class SafePlacement(Scene[SafePlacementParams]):
    """
    Places objects and agents in a way that avoids blocking access to the center column
    where the generator opening will be located.
    """

    def render(self):
        height, width = self.grid.shape

        # Identify center column(s) to avoid
        center_col = width // 2
        # For even width, we have two center columns
        avoid_cols = [center_col]
        if width % 2 == 0:
            avoid_cols.append(center_col - 1)

        # Collect positions, prioritizing corners and edges
        positions = []

        if self.params.prefer_corners:
            # Add corner positions first (most out of the way)
            corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
            for r, c in corners:
                if self.grid[r, c] == "empty":
                    positions.append((r, c, 0))  # Priority 0 for corners

            # Add edge positions (avoiding center columns)
            for r in range(1, height - 1):
                for c in [0, width - 1]:  # Left and right edges
                    if self.grid[r, c] == "empty":
                        positions.append((r, c, 1))  # Priority 1 for edges

            for c in range(1, width - 1):
                if c not in avoid_cols:  # Avoid center columns
                    for r in [0, height - 1]:  # Top and bottom edges
                        if self.grid[r, c] == "empty":
                            positions.append((r, c, 1))

        # Add remaining positions (avoiding center columns)
        for r in range(height):
            for c in range(width):
                if c not in avoid_cols and self.grid[r, c] == "empty":
                    # Check if not already added
                    if not any(pos[0] == r and pos[1] == c for pos in positions):
                        # Interior positions get lower priority
                        priority = (
                            2 if (0 < r < height - 1 and 0 < c < width - 1) else 1
                        )
                        positions.append((r, c, priority))

        # If we still need more positions, reluctantly use center columns
        # but prefer positions farther from the middle row
        if len(positions) < self.params.agents + sum(self.params.objects.values()):
            for c in avoid_cols:
                for r in range(height):
                    if self.grid[r, c] == "empty":
                        # Penalize positions near the middle row where generator will be
                        distance_from_middle = abs(r - height // 2)
                        priority = 3 + (
                            height - distance_from_middle
                        )  # Higher number = lower priority
                        positions.append((r, c, priority))

        # Sort by priority (lower numbers first)
        positions.sort(key=lambda x: x[2])

        # Place agents first (they need to move around)
        for _ in range(self.params.agents):
            if positions:
                r, c, _ = positions.pop(0)
                self.grid[r, c] = "agent.agent"

        # Place objects
        for obj_name, count in self.params.objects.items():
            for _ in range(count):
                if positions:
                    r, c, _ = positions.pop(0)
                    self.grid[r, c] = obj_name


class PlaceGeneratorInWallParams(Config):
    """Parameters for placing a generator in the divider wall."""

    pass


class PlaceGeneratorInWall(Scene[PlaceGeneratorInWallParams]):
    """
    Custom scene that places a generator in the horizontal divider wall.
    This creates an opening in the wall that both agents can access.
    """

    def render(self):
        # Find the horizontal wall row (should be in the middle)
        height, width = self.grid.shape

        # Find the row that's mostly walls (the divider)
        for row in range(1, height - 1):
            wall_count = sum(1 for col in range(width) if self.grid[row, col] == "wall")
            # If this row is mostly walls, it's our divider
            if wall_count > width * 0.7:
                # Place generator at a random position in this wall
                valid_positions = []
                # For small maps (width <= 5), allow edge positions too
                start_col = 1 if width <= 5 else 2
                end_col = width - 1 if width <= 5 else width - 2

                for col in range(start_col, end_col):
                    if self.grid[row, col] == "wall":
                        valid_positions.append(col)

                if valid_positions:
                    # Choose random position and place generator
                    chosen_col = self.rng.choice(valid_positions)
                    self.grid[row, chosen_col] = "generator_red"
                break


def make_env() -> EnvConfig:
    # 2 agents, no combat
    env = make_arena(num_agents=2, combat=False)

    # Minimal actions for pickup/drop
    env.game.actions = ActionsConfig(
        move=ActionConfig(),
        rotate=ActionConfig(),
        get_items=ActionConfig(),
        put_items=ActionConfig(),
    )

    # 100% team reward sharing
    env.game.groups["agent"].group_reward_pct = 1.0

    # Create exactly 2 rooms with a divider wall between them
    # The generator will be placed in the divider wall at a random position
    scene = RoomGrid.factory(
        RoomGrid.Params(
            rows=2,  # Exactly 2 rooms
            columns=1,
            border_width=1,  # Creates a 1-tile wall between the rooms
            border_object="wall",
        ),
        children_actions=[
            # Top room: 1 agent + 1 mine (placed safely away from center)
            ChildrenAction(
                scene=SafePlacement.factory(
                    SafePlacementParams(
                        objects={"mine_red": 1}, agents=1, prefer_corners=True
                    )
                ),
                where=AreaWhere(tags=["room_0_0"]),
            ),
            # Bottom room: 1 agent + 1 altar (placed safely away from center)
            ChildrenAction(
                scene=SafePlacement.factory(
                    SafePlacementParams(
                        objects={"altar": 1}, agents=1, prefer_corners=True
                    )
                ),
                where=AreaWhere(tags=["room_1_0"]),
            ),
            # Place generator in the divider wall (creates an opening)
            ChildrenAction(
                scene=PlaceGeneratorInWall.factory(PlaceGeneratorInWallParams()),
                where="full",
            ),
        ],
    )

    # Map size for 2 rooms with a divider wall
    env.game.map_builder = MapGen.Config(
        width=11,
        height=11,  # Creates 2 equal-sized rooms with a 1-tile wall between them
        border_width=0,  # no outer border for simplicity on small maps
        instance_border_width=0,
        root=scene,
        seed=None,  # Random seed ensures generator position varies each episode
    )

    return env


def make_curriculum(env: Optional[EnvConfig] = None):
    base_env = env or make_env()
    tasks = cc.bucketed(base_env)
    # Train on sizes 3,4,5,6 as requested
    tasks.add_bucket("game.map_builder.width", [3, 4, 5, 6])
    tasks.add_bucket("game.map_builder.height", [3, 4, 5, 6])
    return tasks.to_curriculum()


def make_evals(env: Optional[EnvConfig] = None) -> list[SimulationConfig]:
    base_env = env or make_env()
    evals: list[SimulationConfig] = []
    # Evaluation on sizes 3,4,5,6,7 as requested (square maps)
    eval_sizes = [3, 4, 5, 6, 7]
    for s in eval_sizes:
        e = base_env.model_copy(deep=True)
        # Rebuild map_builder explicitly; cast to access the root field with proper typing
        prev_builder = cast(MapGen.Config, e.game.map_builder)
        e.game.map_builder = MapGen.Config(
            width=s,
            height=s,
            border_width=prev_builder.border_width,
            instance_border_width=prev_builder.instance_border_width,
            root=prev_builder.root,
            seed=None,
        )
        evals.append(SimulationConfig(name=f"eval_size_{s}", env=e))
    return evals


def train() -> TrainTool:
    trainer_cfg = TrainerConfig(
        # Main training parameters
        total_timesteps=10_000_000_000,  # Total training steps
        batch_size=524288,  # Number of steps collected before each update
        minibatch_size=16384,  # Size of minibatches for gradient updates
        update_epochs=1,  # Number of epochs to train on each batch
        # Number of parallel rollout workers (adjust based on your CPU cores)
        rollout_workers=32,  # 1,  # Increase for faster data collection
        # PPO hyperparameters
        ppo=PPOConfig(
            # Learning hyperparameters
            clip_coef=0.1,  # PPO clip coefficient (0.1-0.3 typical)
            ent_coef=0.0021,  # Entropy bonus for exploration
            gae_lambda=0.916,  # GAE lambda for advantage estimation
            gamma=0.977,  # Discount factor for rewards
            # Value function settings
            vf_coef=0.44,  # Value function loss coefficient
            vf_clip_coef=0.1,  # Value function clipping
            max_grad_norm=0.5,  # Gradient clipping
        ),
        # Optimizer settings
        optimizer=OptimizerConfig(
            type="adam",
            learning_rate=0.000457,  # Learning rate (important!)
            beta1=0.9,
            beta2=0.999,
            eps=1e-12,
            weight_decay=0,  # L2 regularization (usually 0 for RL)
        ),
        # Curriculum and evaluation
        curriculum=make_curriculum(),
        evaluation=EvaluationConfig(
            simulations=make_evals(),
            evaluate_interval=50,  # Evaluate every N epochs
            evaluate_local=True,  # Run evaluations locally
        ),
        # Checkpointing
        checkpoint=CheckpointConfig(
            checkpoint_interval=50,  # Save model every N epochs
            wandb_checkpoint_interval=50,  # Upload to wandb at same interval
        ),
    )
    return TrainTool(trainer=trainer_cfg)


def play(env: Optional[EnvConfig] = None) -> PlayTool:
    eval_env = env or make_env()
    return PlayTool(sim=SimulationConfig(env=eval_env, name="rewardsharing"))


def replay(env: Optional[EnvConfig] = None) -> ReplayTool:
    eval_env = env or make_env()
    return ReplayTool(sim=SimulationConfig(env=eval_env, name="rewardsharing"))


def evaluate(
    policy_uri: str,
    simulations: Optional[Sequence[SimulationConfig]] = None,
) -> SimTool:
    simulations = simulations or make_evals()
    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )
