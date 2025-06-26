from typing import Optional

from metta.common.fs import get_repo_root
from metta.common.stopwatch import Stopwatch
from metta.rl.trainer_checkpoint import TrainerCheckpoint


def get_elapsed_time(
    run_name: str, filename: str = "trainer_state.pt", train_dir: str = "train_dir"
) -> Optional[float]:
    """
    Get elapsed time from the trainer state file for a training run.
    This is a standalone helper function that can be called from skypilot scripts
    to check training progress.

    Args:
        run_name: Name of the training run
        filename: Name of the trainer state file (default: "trainer_state.pt")
        train_dir: Base training directory (default: "train_dir")

    Returns:
        Elapsed time in seconds from the global timer, or None if not found
    """
    repo_root = get_repo_root()
    run_dir = repo_root / train_dir / run_name

    if not run_dir.exists():
        return None

    try:
        # Load trainer checkpoint
        trainer_checkpoint = TrainerCheckpoint.load(str(run_dir), filename)

        if trainer_checkpoint is None or trainer_checkpoint.stopwatch_state is None:
            return None

        # Create stopwatch and load the saved state
        stopwatch = Stopwatch()
        stopwatch.load_state(trainer_checkpoint.stopwatch_state, resume_running=False)

        # Get elapsed time from global timer
        elapsed_time = stopwatch.get_elapsed()

        if elapsed_time > 0:
            return elapsed_time
        else:
            return None

    except Exception as e:
        print(f"Failed to load trainer checkpoint: {e}")
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python trainer_state.py <run_name> [filename] [train_dir]")
        print("Example: python trainer_state.py my_training_run_001")
        print("Example: python trainer_state.py my_run trainer_state.pt custom_train_dir")
        sys.exit(1)

    run_name = sys.argv[1]
    filename = sys.argv[2] if len(sys.argv) > 2 else "trainer_state.pt"
    train_dir = sys.argv[3] if len(sys.argv) > 3 else "train_dir"

    print(f"Checking elapsed time for run: {run_name}")
    print(f"Trainer state file: {filename}")
    print(f"Train directory: {train_dir}")

    try:
        elapsed_time = get_elapsed_time(run_name, filename, train_dir)

        if elapsed_time is not None:
            # Format the time in a human-readable way
            if elapsed_time < 60:
                formatted = f"{elapsed_time:.1f} seconds"
            elif elapsed_time < 3600:
                formatted = f"{elapsed_time / 60:.1f} minutes"
            elif elapsed_time < 86400:
                formatted = f"{elapsed_time / 3600:.1f} hours"
            else:
                formatted = f"{elapsed_time / 86400:.1f} days"
            print(f"✓ Training has been running for {elapsed_time:.2f} seconds")
            print(f"✓ Formatted: {formatted}")
        else:
            print("✗ Could not determine elapsed time")
            print("  Possible reasons:")
            print("  - No run directory found")
            print("  - No trainer state file found")
            print("  - No stopwatch_state in trainer checkpoint")
            print("  - No elapsed time in stopwatch")

    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
