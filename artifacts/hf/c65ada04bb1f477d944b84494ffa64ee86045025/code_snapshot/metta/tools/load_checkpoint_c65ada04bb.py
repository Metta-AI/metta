from __future__ import annotations

from pathlib import Path

from metta.tools.load_hf_policy import load_policy


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_dir = repo_root / "artifacts" / "hf" / "c65ada04bb1f477d944b84494ffa64ee86045025"
    load_policy(model_dir)


if __name__ == "__main__":
    main()
