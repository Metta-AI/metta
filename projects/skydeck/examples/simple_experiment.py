"""Example: Create a simple experiment."""

import asyncio

from skydeck.database import Database
from skydeck.desired_state import DesiredStateManager
from skydeck.models import CreateExperimentRequest, DesiredState


async def main():
    """Create a simple experiment."""
    # Connect to database
    db = Database("skydeck.db")
    await db.connect()

    # Create desired state manager
    desired_state_manager = DesiredStateManager(db)

    # Create experiment
    request = CreateExperimentRequest(
        id="test_ppo",
        name="Test PPO Training",
        flags={
            "trainer.losses.ppo.enabled": True,
            "trainer.total_timesteps": 100000,
            "policy_architecture.core_resnet_layers": 4,
        },
        base_command="lt",
        run_name="daveey.test_ppo",
        nodes=1,
        gpus=1,
        instance_type="g4dn.xlarge",
        cloud="aws",
        spot=True,
        desired_state=DesiredState.STOPPED,  # Don't start yet
        description="Test experiment for PPO training",
        tags=["test", "ppo"],
    )

    try:
        experiment = await desired_state_manager.create_experiment(request)
        print(f"Created experiment: {experiment.id}")
        print(f"Command: {experiment.build_command()}")
        print("\nTo start it, visit http://localhost:8000 and click 'Start'")
        print("Or update desired_state to RUNNING in the database")
    except ValueError as e:
        print(f"Error: {e}")

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
