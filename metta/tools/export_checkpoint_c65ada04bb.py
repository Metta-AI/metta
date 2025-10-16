from __future__ import annotations

from pathlib import Path

from metta.tools.export_hf_policy import export_hf_policy


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_path = (
        repo_root
        / "train_dir"
        / "c65ada04bb1f477d944b84494ffa64ee86045025"
        / "checkpoints"
        / "c65ada04bb1f477d944b84494ffa64ee86045025:v40.pt"
    )

    export_dir = repo_root / "artifacts" / "hf" / "c65ada04bb1f477d944b84494ffa64ee86045025"

    export_hf_policy(
        checkpoint_path=checkpoint_path,
        export_dir=export_dir,
    )

    print(f"Exported HF artifact to {export_dir}")


if __name__ == "__main__":
    main()
