from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from hf_metta_policy.modeling_metta_policy import MettaPolicyForRL


def load_policy(model_dir: Path) -> None:
    model_dir = model_dir.expanduser().resolve()
    model = MettaPolicyForRL.from_pretrained(model_dir)
    policy = getattr(model, "policy", None)
    param_count = None
    if policy is not None and hasattr(policy, "parameters"):
        param_count = sum(param.numel() for param in policy.parameters())
    summary_parts = [f"Loaded model from {model_dir}"]
    if param_count is not None:
        summary_parts.append(f"parameters={param_count}")
    print(", ".join(summary_parts))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a locally exported Metta policy via Transformers.")
    parser.add_argument("model_dir", help="Path to the exported HF artifact directory.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    load_policy(Path(args.model_dir))


if __name__ == "__main__":
    main()
