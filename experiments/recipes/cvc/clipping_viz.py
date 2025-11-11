"""Recipe for visualizing and testing clipper behavior.

This recipe creates a playable environment with aggressive clipping parameters
to help validate the new bit-shifting clipping implementation.

Usage:
    uv run ./tools/run.py experiments.recipes.cvc.clipping_viz.play
    uv run ./tools/run.py experiments.recipes.cvc.clipping_viz.play_slow
    uv run ./tools/run.py experiments.recipes.cvc.clipping_viz.play_aggressive
"""

from __future__ import annotations

from experiments.recipes import cogs_v_clips
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from mettagrid.config.mettagrid_config import (
    ClipperConfig,
    MettaGridConfig,
    ProtocolConfig,
)


def _make_clipping_env(
    mission: str = "extractor_hub_50",
    num_cogs: int = 4,
    clip_period: int = 20,
    length_scale: float = 0.0,
    scaled_cutoff_distance: int = 3,
    start_hub_clipped: bool = True,
) -> MettaGridConfig:
    """Create an environment optimized for visualizing clipping.

    Args:
        mission: Which mission map to use
        num_cogs: Number of agents
        clip_period: Steps between clipping events (lower = more frequent)
        length_scale: Spatial spread rate (0.0 = auto-calculate via percolation)
        scaled_cutoff_distance: Max distance in units of length_scale
        start_hub_clipped: Whether to start hub stations clipped
    """
    # Start with base mission environment
    variants = []
    if start_hub_clipped:
        variants.append("clip_hub_stations")

    env = cogs_v_clips.make_training_env(
        num_cogs=num_cogs,
        mission=mission,
        variants=variants if variants else None,
    )

    # Override clipper configuration with our test parameters
    env.game.clipper = ClipperConfig(
        unclipping_protocols=[
            ProtocolConfig(input_resources={"decoder": 1}, cooldown=1),
            ProtocolConfig(input_resources={"modulator": 1}, cooldown=1),
            ProtocolConfig(input_resources={"scrambler": 1}, cooldown=1),
            ProtocolConfig(input_resources={"resonator": 1}, cooldown=1),
        ],
        clip_period=clip_period,
        length_scale=length_scale,
        scaled_cutoff_distance=scaled_cutoff_distance,
    )

    # Make episodes a bit longer to see clipping spread
    env.game.max_steps = 2000

    return env


def play(
    policy_uri: str | None = None,
    clip_period: int = 20,
    length_scale: float = 0.0,
    scaled_cutoff_distance: int = 3,
) -> PlayTool:
    """Play with default clipping parameters.

    This uses auto-calculated length_scale (based on percolation theory) and
    default scaled_cutoff_distance of 3. Clipping happens every ~20 steps.

    Args:
        policy_uri: Optional policy to control agents (None = scripted/manual)
        clip_period: Steps between clipping events
        length_scale: Spatial spread rate (0.0 = auto-calculate)
        scaled_cutoff_distance: Max distance in units of length_scale
    """
    env = _make_clipping_env(
        mission="extractor_hub_50",
        num_cogs=4,
        clip_period=clip_period,
        length_scale=length_scale,
        scaled_cutoff_distance=scaled_cutoff_distance,
        start_hub_clipped=True,
    )

    sim = SimulationConfig(
        suite="cogs_vs_clips",
        name="clipping_viz_default",
        env=env,
    )

    return PlayTool(sim=sim, policy_uri=policy_uri)


def play_slow(policy_uri: str | None = None) -> PlayTool:
    """Play with slower clipping to observe the spread more carefully.

    Clips every ~50 steps instead of ~20, giving more time to observe
    the spatial diffusion pattern.
    """
    env = _make_clipping_env(
        mission="extractor_hub_50",
        num_cogs=4,
        clip_period=50,
        length_scale=0.0,  # auto
        scaled_cutoff_distance=3,
        start_hub_clipped=True,
    )

    sim = SimulationConfig(
        suite="cogs_vs_clips",
        name="clipping_viz_slow",
        env=env,
    )

    return PlayTool(sim=sim, policy_uri=policy_uri)


def play_aggressive(policy_uri: str | None = None) -> PlayTool:
    """Play with very aggressive clipping parameters.

    Fast clipping (every 10 steps) with shorter cutoff distance to see
    more localized spreading. Good for testing edge cases.
    """
    env = _make_clipping_env(
        mission="extractor_hub_50",
        num_cogs=4,
        clip_period=10,
        length_scale=0.0,  # auto
        scaled_cutoff_distance=2,  # shorter reach
        start_hub_clipped=True,
    )

    sim = SimulationConfig(
        suite="cogs_vs_clips",
        name="clipping_viz_aggressive",
        env=env,
    )

    return PlayTool(sim=sim, policy_uri=policy_uri)


def play_large_map(policy_uri: str | None = None) -> PlayTool:
    """Play on a larger map (70x70) to see clipping over more distance.

    Uses extractor_hub_70 mission to test clipping behavior on a larger
    spatial scale with more buildings.
    """
    env = _make_clipping_env(
        mission="extractor_hub_70",
        num_cogs=8,
        clip_period=20,
        length_scale=0.0,  # auto (should calculate larger value)
        scaled_cutoff_distance=4,  # slightly longer reach for bigger map
        start_hub_clipped=True,
    )

    sim = SimulationConfig(
        suite="cogs_vs_clips",
        name="clipping_viz_large",
        env=env,
    )

    return PlayTool(sim=sim, policy_uri=policy_uri)


def play_custom(
    policy_uri: str | None = None,
    mission: str = "extractor_hub_50",
    num_cogs: int = 4,
    clip_period: int = 20,
    length_scale: float = 0.0,
    scaled_cutoff_distance: int = 3,
    start_hub_clipped: bool = True,
) -> PlayTool:
    """Play with fully customizable clipping parameters.

    Use this to experiment with different parameter combinations and validate
    the bit-shifting implementation.

    Args:
        policy_uri: Optional policy to control agents
        mission: Mission name (e.g., "extractor_hub_30", "extractor_hub_50", "extractor_hub_70")
        num_cogs: Number of agents
        clip_period: Steps between clipping events (lower = more frequent)
        length_scale: Spatial spread rate (0.0 = auto-calculate via percolation)
        scaled_cutoff_distance: Max distance in units of length_scale
        start_hub_clipped: Whether to start hub stations clipped

    Example:
        # Test with manual length_scale and aggressive clipping
        uv run ./tools/run.py experiments.recipes.cvc.clipping_viz.play_custom \\
            clip_period=5 length_scale=8.0 scaled_cutoff_distance=2
    """
    env = _make_clipping_env(
        mission=mission,
        num_cogs=num_cogs,
        clip_period=clip_period,
        length_scale=length_scale,
        scaled_cutoff_distance=scaled_cutoff_distance,
        start_hub_clipped=start_hub_clipped,
    )

    sim = SimulationConfig(
        suite="cogs_vs_clips",
        name=f"clipping_viz_custom_{mission}",
        env=env,
    )

    return PlayTool(sim=sim, policy_uri=policy_uri)


__all__ = [
    "play",
    "play_slow",
    "play_aggressive",
    "play_large_map",
    "play_custom",
]
