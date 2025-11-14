"""Build Tinker-compatible training dataset with return conditioning."""

import json
from typing import Optional

from metta.llm.trajectory_collector import Episode


class TinkerDatasetBuilder:
    """Build Tinker-compatible training dataset."""

    def __init__(self, system_prompt: Optional[str] = None):
        self.system_prompt = system_prompt or self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        return """You are a MettaGrid agent trained to achieve target returns.
MettaGrid is a multi-agent gridworld game where agents:
- Navigate a grid environment
- Collect resources (ore, batteries)
- Craft items (lasers, armor)
- Can attack other agents

You will be given:
1. A target return (cumulative future reward to achieve)
2. Current observation in JSON format

Predict the action that will lead to achieving the target return.
Respond with only the action name."""

    def build_dataset(
        self,
        episodes: list[Episode],
        use_return_conditioning: bool = True,
    ) -> list[dict]:
        """Convert episodes to Tinker JSONL format with return conditioning."""
        dataset = []

        for episode in episodes:
            for step_idx, step in enumerate(episode.steps):
                if use_return_conditioning:
                    # Decision Transformer: include return-to-go
                    target_return = episode.returns_to_go[step_idx]
                    user_content = f"Target return: {target_return:.1f}\n{step.observation_text}"
                else:
                    # Behavior cloning: just observation
                    user_content = step.observation_text

                # Each step becomes a conversation
                conversation = {
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": step.action_name},
                    ]
                }
                dataset.append(conversation)

        return dataset

    def save_dataset(self, dataset: list[dict], output_path: str):
        """Save as JSONL (one JSON object per line)."""
        with open(output_path, "w") as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")
        print(f"Saved {len(dataset)} training examples to {output_path}")
