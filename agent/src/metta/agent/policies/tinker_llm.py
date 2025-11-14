"""Policy wrapper for Tinker-finetuned LLMs with return conditioning."""

from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from metta.llm.observation_encoder import ObservationEncoder
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation


class TinkerLLMPolicy(MultiAgentPolicy):
    """Policy using a Tinker fine-tuned LLM with return conditioning."""

    short_names = ["tinker_llm", "llm_dt"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        base_model: str = "meta-llama/Llama-3.2-1B",
        lora_weights_path: Optional[str] = None,
        target_return: Optional[float] = None,  # Target return for inference
        adaptive_return: bool = True,  # Dynamically adjust target return
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        super().__init__(policy_env_info, **kwargs)
        self.encoder = ObservationEncoder(policy_env_info)
        self.target_return = target_return
        self.adaptive_return = adaptive_return
        self.device = device

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
        )

        # Load LoRA weights if provided
        if lora_weights_path:
            self.model = PeftModel.from_pretrained(self.model, lora_weights_path)

        self.model.eval()

        # System prompt
        self.system_prompt = """You are a MettaGrid agent trained to achieve target returns.
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

        # Track episode statistics for adaptive return
        self.episode_return = 0.0
        self.episode_step = 0

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return _TinkerLLMAgentPolicy(self, agent_id)

    def reset(self):
        """Reset episode tracking."""
        self.episode_return = 0.0
        self.episode_step = 0


class _TinkerLLMAgentPolicy(AgentPolicy):
    """Per-agent policy wrapper with return-conditioned inference."""

    def __init__(self, parent: TinkerLLMPolicy, agent_id: int):
        super().__init__(parent.policy_env_info)
        self.parent = parent
        self.agent_id = agent_id
        self.last_reward = 0.0

    def step(self, obs: AgentObservation) -> Action:
        # Determine target return
        if self.parent.target_return is None:
            # Use high default if not specified (e.g., 90th percentile of training)
            target_return = 150.0  # TODO: Make this configurable
        elif self.parent.adaptive_return:
            # Adjust target based on progress (remaining budget)
            # This allows model to adapt if it's not meeting target
            target_return = max(0.0, self.parent.target_return - self.parent.episode_return)
        else:
            # Fixed target return
            target_return = self.parent.target_return

        # Encode observation with target return
        obs_json = self.parent.encoder.encode(obs)
        user_prompt = f"Target return: {target_return:.1f}\nObservation: {obs_json}"

        # Build messages
        messages = [
            {"role": "system", "content": self.parent.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Tokenize
        prompt = self.parent.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.parent.tokenizer(prompt, return_tensors="pt").to(self.parent.device)

        # Generate
        with torch.no_grad():
            outputs = self.parent.model.generate(
                **inputs,
                max_new_tokens=20,  # Action names are short
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.parent.tokenizer.eos_token_id,
            )

        # Decode
        response = self.parent.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()

        # Parse action
        action_name = self._parse_action(response)

        # Update episode tracking
        self.parent.episode_step += 1
        # Note: reward will be updated after step is executed

        return Action(name=action_name)

    def _parse_action(self, llm_output: str) -> str:
        """Extract action name from LLM output."""
        # Clean up output
        action = llm_output.strip().lower()

        # Try to match to valid action names
        valid_actions = [a.lower() for a in self.policy_env_info.action_names]

        # Exact match
        if action in valid_actions:
            idx = valid_actions.index(action)
            return self.policy_env_info.action_names[idx]

        # Partial match (in case LLM adds extra text)
        for i, valid_action in enumerate(valid_actions):
            if valid_action in action or action in valid_action:
                return self.policy_env_info.action_names[i]

        # Fallback to noop
        print(f"Warning: Could not parse action '{llm_output}', using noop")
        return "noop"

    def reset(self):
        """Reset per-agent state."""
        self.last_reward = 0.0
