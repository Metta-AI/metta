#!/usr/bin/env env python
"""Example: Run machina_1 with 4 different policies controlling the 4 agents.

This demonstrates how to assign different policies to different agents in CoGames.

Usage:
    # Run with 4 random agents
    python metta/machina1_tournament/multi_policy_example.py

    # Run with specific policy checkpoints
    python metta/machina1_tournament/multi_policy_example.py \
        --policy1 stateless:./checkpoints/policy1.pt \
        --policy2 stateless:./checkpoints/policy2.pt \
        --policy3 random \
        --policy4 simple

    # Run with mix of policies
    python metta/machina1_tournament/multi_policy_example.py \
        --policy1 random \
        --policy2 random \
        --policy3 stateless:./train_dir/best.pt \
        --policy4 stateless:./train_dir/best.pt
"""

import logging
from pathlib import Path

import typer
from rich.console import Console

from mettagrid.policy.policy import PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.utils import initialize_or_load_policy
from mettagrid.simulator.rollout import Rollout

# Import cogames mission registry
try:
    from cogames.cli.mission import get_mission
except ImportError:
    # Fallback if cogames not installed as package
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "cogames" / "src"))
    from cogames.cli.mission import get_mission

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_policy_spec(policy_str: str) -> PolicySpec:
    """Parse a policy string like 'random' or 'stateless:path/to/checkpoint.pt'."""
    parts = policy_str.split(":", 1)

    if len(parts) == 1:
        # Just policy class, no data path
        policy_class = parts[0]
        policy_data = None
    else:
        policy_class, policy_data = parts

    # Map common names to full class paths
    policy_class_map = {
        "random": "mettagrid.policy.random.RandomMultiAgentPolicy",
        "simple": "cogames.policy.scripted_agent.baseline_agent.BaselinePolicy",
        "unclipping": "cogames.policy.scripted_agent.unclipping_agent.UnclippingPolicy",
        "stateless": "metta.agent.policies.fast.FastPolicy",
        "puffer": "metta.agent.policies.puffer.PufferPolicy",
    }

    policy_class_path = policy_class_map.get(policy_class, policy_class)

    return PolicySpec(
        policy_class_path=policy_class_path,
        policy_data_path=policy_data,
        name=policy_str,
        proportion=1.0,  # Not used for direct assignment
    )


def main(
    mission: str = typer.Option("machina_1", "--mission", "-m", help="Mission name"),
    policy1: str = typer.Option("random", help="Policy for agent 0 (e.g., 'random' or 'stateless:path.pt')"),
    policy2: str = typer.Option("random", help="Policy for agent 1"),
    policy3: str = typer.Option("random", help="Policy for agent 2"),
    policy4: str = typer.Option("random", help="Policy for agent 3"),
    seed: int = typer.Option(42, help="Random seed"),
    render: bool = typer.Option(True, help="Whether to render the game"),
    max_steps: int = typer.Option(1000, help="Maximum steps per episode"),
) -> None:
    """Run a mission with 4 different policies controlling the 4 agents."""
    console = Console()

    # Load mission config
    console.print(f"[cyan]Loading mission: {mission}[/cyan]")
    _, env_cfg, _ = get_mission(mission)

    # Verify this mission has exactly 4 agents
    num_agents = env_cfg.game.num_agents
    if num_agents != 4:
        console.print(
            f"[red]Error: This script is designed for 4-agent missions, but {mission} has {num_agents} agents[/red]"
        )
        raise typer.Exit(1)

    # Parse policy specifications
    policy_specs = [
        parse_policy_spec(policy1),
        parse_policy_spec(policy2),
        parse_policy_spec(policy3),
        parse_policy_spec(policy4),
    ]

    console.print("\n[bold cyan]Policy Assignments:[/bold cyan]")
    for i, spec in enumerate(policy_specs):
        console.print(f"  Agent {i}: {spec.name}")

    # Create PolicyEnvInterface for policy instantiation
    policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)

    # Load/initialize each policy
    console.print("\n[cyan]Initializing policies...[/cyan]")
    policy_instances = []
    for i, spec in enumerate(policy_specs):
        console.print(f"  Loading policy {i}: {spec.policy_class_path}")
        policy = initialize_or_load_policy(
            policy_env_info,
            spec.policy_class_path,
            spec.policy_data_path,
        )
        policy_instances.append(policy)

    # Create agent policies - one per agent
    console.print("\n[cyan]Creating agent policy assignments...[/cyan]")
    agent_policies = []
    for agent_id in range(num_agents):
        # Each agent gets its corresponding policy
        policy = policy_instances[agent_id]
        agent_policy = policy.agent_policy(agent_id)
        agent_policies.append(agent_policy)
        console.print(f"  Agent {agent_id} -> Policy {agent_id} ({policy_specs[agent_id].name})")

    # Create and run rollout
    console.print(f"\n[cyan]Starting rollout (seed={seed}, max_steps={max_steps})...[/cyan]")
    render_mode = "gui" if render else None

    rollout = Rollout(
        env_cfg,
        agent_policies,
        max_action_time_ms=10000,
        render_mode=render_mode,
        seed=seed,
        pass_sim_to_policies=True,
    )

    # Run until done or max steps
    step_count = 0
    while not rollout.is_done() and step_count < max_steps:
        rollout.step()
        step_count += 1

    # Print summary
    console.print("\n[bold green]Episode Complete![/bold green]")
    console.print(f"Steps: {rollout._sim.current_step}")
    console.print(f"Total Rewards: {rollout._sim.episode_rewards}")
    console.print(f"Final Reward Sum: {float(sum(rollout._sim.episode_rewards)):.2f}")

    # Print per-agent rewards
    console.print("\n[bold cyan]Per-Agent Rewards:[/bold cyan]")
    for agent_id, reward in enumerate(rollout._sim.episode_rewards):
        console.print(f"  Agent {agent_id} ({policy_specs[agent_id].name}): {reward:.2f}")

    # Check for timeouts
    console.print("\n[bold cyan]Action Timeouts:[/bold cyan]")
    for agent_id, timeout_count in enumerate(rollout.timeout_counts):
        if timeout_count > 0:
            console.print(f"  Agent {agent_id}: {timeout_count} timeouts")
        else:
            console.print(f"  Agent {agent_id}: No timeouts")


if __name__ == "__main__":
    typer.run(main)
