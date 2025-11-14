# LLM Fine-tuning for MettaGrid with Decision Transformer

Train language models to play MettaGrid using return-conditioned supervised learning (Decision Transformer approach).

## Overview

This system enables fine-tuning LLMs to play MettaGrid by:
1. **Collecting diverse trajectories** from policies of varying skill levels
2. **Recording return-to-go** at each timestep
3. **Training LLM** to map (observation, target_return) → action
4. **Prompting with high target returns** at inference to achieve better performance

## Quick Start

### 1. Setup Tinker Account

```bash
# Sign up at https://thinkingmachines.ai/tinker
# Get API key from https://tinker-console.thinkingmachines.ai
export TINKER_API_KEY=your_key_here

# Install Tinker
pip install tinker
```

### 2. Collect Diverse Trajectories

```bash
# Collect from 3 different skill levels (expert, medium, weak)
uv run python scripts/llm_finetune_pipeline.py \
    --expert-policy file://./train_dir/my_run/checkpoints/my_run:v100.pt \
    --medium-policy file://./train_dir/my_run/checkpoints/my_run:v50.pt \
    --weak-policy file://./train_dir/my_run/checkpoints/my_run:v10.pt \
    --episodes-per-policy 200 \
    --output-dir ./llm_training_data

# This creates:
# - llm_training_data/train.jsonl (return-conditioned training data)
# - llm_training_data/metadata.json (dataset statistics)
```

### 3. Fine-tune on Tinker

```bash
uv run python metta/llm/finetune_with_tinker.py \
    --dataset ./llm_training_data/train.jsonl \
    --model meta-llama/Llama-3.2-1B \
    --lora-rank 32 \
    --batch-size 128 \
    --epochs 1
```

### 4. Evaluate Policy

```python
from metta.agent.policies.tinker_llm import TinkerLLMPolicy
from mettagrid.builder.envs import make_arena
from mettagrid.policy.policy_env_interface import PolicyEnvInterface

# Setup environment
arena_env = make_arena(num_agents=24)
policy_env_info = PolicyEnvInterface.from_mg_cfg(arena_env)

# Load LLM policy with return conditioning
policy = TinkerLLMPolicy(
    policy_env_info=policy_env_info,
    base_model="meta-llama/Llama-3.2-1B",
    lora_weights_path="./path/to/downloaded/lora/weights",
    target_return=150.0,  # Aim for high performance
    adaptive_return=True,  # Adjust target based on progress
)

# Use in evaluation or play
# ... (integrate with existing eval/play tools)
```

## Components

### TrajectoryCollector

Collects episodes with full reward information:
- Records observations, actions, rewards at each step
- Computes return-to-go (cumulative future rewards)
- Supports collecting from multiple policies

```python
from metta.llm import TrajectoryCollector, ObservationEncoder
from mettagrid.builder.envs import make_arena
from mettagrid.policy.policy_env_interface import PolicyEnvInterface

arena_env = make_arena(num_agents=24)
policy_env_info = PolicyEnvInterface.from_mg_cfg(arena_env)
encoder = ObservationEncoder(policy_env_info)
collector = TrajectoryCollector(arena_env, encoder)

# Collect from multiple policies
episodes = collector.collect_diverse_dataset(
    policy_uris={
        "expert": "file://./train_dir/run/checkpoints/run:v100.pt",
        "medium": "file://./train_dir/run/checkpoints/run:v50.pt",
        "weak": "file://./train_dir/run/checkpoints/run:v10.pt",
    },
    episodes_per_policy=200,
)
```

### ObservationEncoder

Converts tokenized observations to structured JSON:

```python
obs_json = encoder.encode(observation)
# Output: {"agent_state":{"health":80,"position":{"x":5,"y":7}},...}

# With return conditioning
obs_with_return = encoder.encode_with_return(observation, target_return=150.0)
# Output: "Target return: 150.0\nObservation: {...}"
```

### TinkerDatasetBuilder

Formats data for Tinker's JSONL format with return conditioning:

```python
from metta.llm import TinkerDatasetBuilder

builder = TinkerDatasetBuilder()
dataset = builder.build_dataset(episodes, use_return_conditioning=True)
builder.save_dataset(dataset, "./train.jsonl")

# Each line in JSONL:
# {"messages": [
#   {"role": "system", "content": "You are a MettaGrid agent..."},
#   {"role": "user", "content": "Target return: 150.0\nObservation: {...}"},
#   {"role": "assistant", "content": "move_north"}
# ]}
```

### TinkerLLMPolicy

Policy wrapper that uses fine-tuned LLM with return conditioning:

```python
policy = TinkerLLMPolicy(
    policy_env_info=policy_env_info,
    base_model="meta-llama/Llama-3.2-1B",
    lora_weights_path="./lora_weights",
    target_return=150.0,       # Target return for inference
    adaptive_return=True,      # Adjust based on accumulated reward
    device="cuda",
)

# The policy will:
# 1. Encode observation with target return
# 2. Query LLM for action
# 3. Parse action name and return
```

## Key Features

### Return Conditioning (Decision Transformer)

Instead of behavior cloning (copying expert), we train on diverse data with return labels:
- **Training**: Learn (obs, target_return) → action mappings across skill levels
- **Inference**: Prompt with high target returns to get high performance
- **Flexibility**: Single model can perform at different skill levels

### Adaptive Return Targeting

The policy adjusts target return based on progress:
```python
# If target_return=150 and we've earned 50 reward so far:
remaining_target = 150 - 50 = 100  # New target for rest of episode
```

This helps the model stay on track if it's underperforming.

### Data Diversity

Collecting from multiple skill levels ensures:
- Coverage of the full return spectrum
- Richer training signal
- Potential for upward distribution shift (prompting beyond training max)

## Expected Results

### Success Criteria

- ✅ Return conditioning works: higher target returns → higher actual performance
- ✅ Policy matches median training performance when prompted with median return
- 🎯 (Stretch) Policy exceeds training distribution when prompted with very high returns

### Performance Metrics

Evaluate with different target returns:
```python
for target_return in [50, 100, 150, 200]:
    policy = TinkerLLMPolicy(..., target_return=target_return)
    actual_return = evaluate(policy)
    print(f"Target: {target_return} → Actual: {actual_return}")
```

Expected: Positive correlation between target and actual returns.

## Troubleshooting

### Invalid Actions

If LLM outputs invalid actions:
- Check `_parse_action()` fallback logic
- Verify action names in training data match environment
- Consider adding action list to system prompt

### Poor Performance

- **Increase data diversity**: Collect from more policy checkpoints
- **Scale up dataset**: 600 episodes → 1000+ episodes
- **Try larger model**: Llama-3.2-1B → Llama-3.2-3B
- **Adjust return scaling**: Normalize returns to [0, 1] range

### Token Limit Exceeded

Observations too long for LLM context:
- Reduce `visible_entities` limit (currently 20)
- Summarize observations more aggressively
- Filter out low-value features

## Next Steps

1. **Run first experiment**: Collect 600 episodes, train, evaluate
2. **Analyze return conditioning**: Plot target vs. actual returns
3. **Compare to baselines**: How does it stack up against RL policies?
4. **Iterate**: Experiment with prompts, models, data scale

## References

- [Decision Transformer paper](https://arxiv.org/abs/2106.01345)
- [Tinker Documentation](https://tinker-docs.thinkingmachines.ai/)
- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
