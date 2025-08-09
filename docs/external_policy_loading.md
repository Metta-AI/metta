# Loading and Evaluating External PyTorch Policies

This guide explains how to load and evaluate external PyTorch policies (e.g., from pufferlib) in the Metta environment.

## Overview

Metta supports loading external PyTorch policies using the `pytorch://` URI scheme. This allows you to evaluate policies trained outside of Metta's training framework.

## Quick Start

### Option 1: Using the Standalone Evaluation Script

The simplest way to evaluate an external policy:

```bash
python tools/eval_external_policy.py \
    --policy-path train_dir/metta_7-23/metta.pt \
    --simulation-suite laser_tag \
    --save-replays
```

Options:
- `--policy-path`: Path to your PyTorch policy file (.pt)
- `--simulation-suite`: Choose from `navigation_single`, `laser_tag`, or `full_eval`
- `--num-episodes`: Number of episodes per task (default: 10)
- `--save-replays`: Save replay videos
- `--pytorch-config`: Path to YAML config defining the policy architecture

### Option 2: Using Your User Config

1. Update your user config (e.g., `configs/user/relh.yaml`):

```yaml
# Update the policy path
policy_uri: pytorch://train_dir/metta_7-23/metta.pt

# Define the policy architecture
pytorch:
  _target_: metta.agent.external.example.Recurrent
  input_size: 512
  hidden_size: 512
  policy:
    _target_: metta.agent.external.example.Policy
    cnn_channels: 128
    hidden_size: 512
```

2. Run evaluation:

```bash
python tools/sim.py cmd=sim user=relh
```

### Option 3: Direct Command Line

```bash
python tools/sim.py cmd=sim \
    policy_uri=pytorch://train_dir/metta_7-23/metta.pt \
    sim_job.simulation_suite.name=laser_tag
```

## Policy Architecture Configuration

The PyTorch policy loader needs to know the architecture of your policy. You can specify this in two ways:

### 1. In Your User Config

Add a `pytorch` section to your user config:

```yaml
pytorch:
  _target_: your.module.PolicyClass
  hidden_size: 512
  # ... other architecture parameters
```

### 2. Using a Separate Config File

Create a YAML file (e.g., `configs/pytorch/my_policy.yaml`):

```yaml
_target_: metta.agent.external.example.Recurrent
input_size: 512
hidden_size: 512
policy:
  _target_: metta.agent.external.example.Policy
  cnn_channels: 128
  hidden_size: 512
```

Then use it with:

```bash
python tools/eval_external_policy.py \
    --policy-path your_policy.pt \
    --pytorch-config configs/pytorch/my_policy.yaml
```

## Default Architecture

If no architecture is specified, the loader will attempt to infer parameters from the checkpoint and use the default `Recurrent` policy wrapper.

## Supported Policy Formats

The loader expects PyTorch checkpoints with state dictionaries containing:
- `policy.actor.0.weight`: Actor network weights
- `policy.actor.1.weight`: Action argument weights
- `policy.network.0.weight`: CNN feature extractor weights
- Additional LSTM weights if using recurrent policies

## Custom Policy Implementation

To use a custom policy architecture:

1. Create a new module in `metta/agent/external/`
2. Implement a policy class that follows the pufferlib interface
3. Update your config to use `_target_: metta.agent.external.your_module.YourPolicy`

## Troubleshooting

### Common Issues

1. **"Failed to load policy"**: Check that the file path is correct and the file exists
2. **Architecture mismatch**: Ensure your `pytorch` config matches the actual policy architecture
3. **Device errors**: Use `--device cpu` if you encounter CUDA issues

### Debugging Tips

- The loader will log inferred architecture parameters when loading
- Use `--save-replays` to visually verify the policy behavior
- Check the output directory for detailed evaluation results

## Example Results

After evaluation, you'll see results like:

```
Evaluation Results:
==================================================
Average simulation score: 0.842
Average category score: 0.756

Detailed scores:

laser_tag:
  simple: 0.842

Results saved to: ./eval_results/eval_metta_laser_tag/evaluation_results.yaml
```
