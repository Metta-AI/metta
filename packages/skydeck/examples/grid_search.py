"""Example: Create a grid search of experiments."""

import asyncio

from skydeck.database import Database
from skydeck.desired_state import DesiredStateManager
from skydeck.models import CreateExperimentRequest, DesiredState


async def main():
    """Create a grid search over different configurations."""
    # Connect to database
    db = Database("skydeck.db")
    await db.connect()

    # Create desired state manager
    desired_state_manager = DesiredStateManager(db)

    # Grid search parameters
    nodes_options = [1, 4]
    layer_options = [1, 4, 16, 64]

    print("Creating grid search experiments...")

    for nodes in nodes_options:
        for layers in layer_options:
            experiment_id = f"ca6.{nodes}x4.ppo_{layers}layer"

            request = CreateExperimentRequest(
                id=experiment_id,
                name=f"PPO {layers} Layers ({nodes} nodes)",
                flags={
                    "trainer.losses.ppo.enabled": True,
                    "policy_architecture.core_resnet_layers": layers,
                },
                base_command="lt",
                run_name=f"daveey.grid_{nodes}n_{layers}l",
                nodes=nodes,
                gpus=4,
                cloud="aws",
                spot=True,
                desired_state=DesiredState.RUNNING,  # Start immediately
                description=f"Grid search: {nodes} nodes, {layers} resnet layers",
                tags=["grid_search", "ppo", f"layers_{layers}"],
            )

            try:
                experiment = await desired_state_manager.create_experiment(request)
                print(f"✓ Created: {experiment.id}")
            except ValueError as e:
                print(f"✗ Skipped {experiment_id}: {e}")

    await db.close()
    print("\nDone! The reconciler will automatically launch all experiments.")
    print("View them at http://localhost:8000")


if __name__ == "__main__":
    asyncio.run(main())
