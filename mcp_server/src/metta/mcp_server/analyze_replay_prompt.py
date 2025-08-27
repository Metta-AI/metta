#!/usr/bin/env python3
"""Generate the user prompt that would be sent to Claude for replay analysis."""

import sys
from pathlib import Path

from dotenv import load_dotenv

from metta.mcp_server.llm_client import LLMClient
from metta.mcp_server.training_utils import _analyze_replay_data, _load_replay_file

load_dotenv()

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python analyze_replay_prompt.py <replay_file.json> [policy_uri]")
        print("Example: python analyze_replay_prompt.py replay.json wandb://run/trained-model:v42")
        sys.exit(1)

    replay_file = Path(sys.argv[1])
    policy_uri = sys.argv[2] if len(sys.argv) == 3 else None

    if not replay_file.exists():
        print(f"Error: File {replay_file} does not exist")
        sys.exit(1)

    try:
        # Load and analyze the replay file
        replay_data = _load_replay_file(replay_file)
        analysis = _analyze_replay_data(replay_data, policy_uri)

        # Create LLM client and generate the user prompt
        client = LLMClient()
        user_prompt = client._create_user_prompt(analysis)

        print("=" * 80)
        print("USER PROMPT THAT WOULD BE SENT TO CLAUDE:")
        print("=" * 80)
        print(user_prompt)

    except Exception as e:
        print(f"Error analyzing replay file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
