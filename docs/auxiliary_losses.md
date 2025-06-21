# Auxiliary Losses for Representation Learning

This document explains how to use the auxiliary losses implemented in the Metta training framework to improve representation learning.

## Overview

The auxiliary losses are designed to densify the learning signal and improve representation quality by adding additional supervision signals beyond the standard policy gradient and value function losses.

## Available Auxiliary Losses

### 1. Sensory Prediction Decoder Loss

**Purpose**: Ensures the learned representations retain sensory information by requiring the agent to reconstruct original sensory inputs from its latent representations.

**Implementation**:
- Component: `metta.agent.lib.sensory_decoder.SensoryDecoder`
- Loss: MSE between reconstructed and original sensory inputs
- Coefficient: `sensory_decoder_coef`

### 2. Latent Prediction Decoder Loss

**Purpose**: Encourages representations to encode temporal dynamics by predicting future latent states from current representations.

**Implementation**:
- Component: `metta.agent.lib.latent_decoder.LatentDecoder`
- Loss: MSE between predicted and actual next latent states
- Coefficient: `latent_decoder_coef`

### 3. Node Perturbation Loss

**Purpose**: Encourages robust representations by penalizing large deviations from mean rewards, following the formula: GAE * (r_i - r̄_i)².

**Implementation**:
- Method: `_compute_node_perturbation_loss()` in `MettaTrainer`
- Loss: Weighted reward deviation penalty
- Coefficient: `node_perturbation_coef`

### 4. Contrastive Loss

**Purpose**: Encourages similar states/actions to have similar representations and different states/actions to have different representations.

**Implementation**:
- Method: `_compute_contrastive_loss()` in `MettaTrainer`
- Loss: Contrastive learning across time sequences
- Coefficient: `contrastive_coef`

## Configuration

### Trainer Configuration

Add the auxiliary loss coefficients to your trainer configuration:

```yaml
# Auxiliary losses for representation learning
# Set to 0 to disable, >0 to enable with specified coefficient
sensory_decoder_coef: 0.1      # Sensory reconstruction loss coefficient
latent_decoder_coef: 0.1       # Latent prediction loss coefficient
node_perturbation_coef: 0.01   # Node perturbation loss coefficient
contrastive_coef: 0.1          # Contrastive learning loss coefficient
```

### Agent Configuration

To use the sensory and latent decoders, use the `fast_with_auxiliary` agent configuration or add the decoder components to your custom agent:

```yaml
# Add these components to your agent configuration
sensory_decoder:
  _target_: metta.agent.lib.sensory_decoder.SensoryDecoder
  sources:
    - name: _core_
  output_size: 128  # Should match the flattened observation size
  decoder_hidden_dim: 256

latent_decoder:
  _target_: metta.agent.lib.latent_decoder.LatentDecoder
  sources:
    - name: _core_
  output_size: 128  # Should match the core output size
  decoder_hidden_dim: 256
```

## Usage Examples

### Example 1: Enable All Auxiliary Losses

```yaml
# In your trainer config
sensory_decoder_coef: 0.1
latent_decoder_coef: 0.1
node_perturbation_coef: 0.01
contrastive_coef: 0.1

# Use the agent with decoders
agent: fast_with_auxiliary
```

### Example 2: Enable Only Contrastive Learning

```yaml
# In your trainer config
sensory_decoder_coef: 0.0      # Disabled
latent_decoder_coef: 0.0       # Disabled
node_perturbation_coef: 0.0    # Disabled
contrastive_coef: 0.1          # Enabled

# Use any agent configuration
agent: fast
```

### Example 3: Strong Sensory Reconstruction

```yaml
# In your trainer config
sensory_decoder_coef: 0.5      # Strong sensory reconstruction
latent_decoder_coef: 0.0       # Disabled
node_perturbation_coef: 0.0    # Disabled
contrastive_coef: 0.0          # Disabled

# Use the agent with decoders
agent: fast_with_auxiliary
```

## Monitoring

The auxiliary losses are automatically logged to wandb with the following keys:

- `sensory_decoder_loss`: Sensory reconstruction loss
- `latent_decoder_loss`: Latent prediction loss
- `node_perturbation_loss`: Node perturbation loss
- `contrastive_loss`: Contrastive learning loss

## Tips for Tuning

1. **Start Small**: Begin with small coefficients (0.01-0.1) and gradually increase
2. **Monitor Performance**: Watch for any degradation in main task performance
3. **Ablation Studies**: Test each loss individually to understand its impact
4. **Computational Cost**: Contrastive loss can be computationally expensive with large batch sizes
5. **Memory Usage**: Decoder components add memory overhead

## Troubleshooting

### Common Issues

1. **Shape Mismatch**: Ensure `output_size` in decoder components matches the expected dimensions
2. **Memory Errors**: Reduce batch size if using contrastive loss
3. **Training Instability**: Lower the auxiliary loss coefficients
4. **No Effect**: Check that the coefficients are > 0 and the components are properly configured

### Debugging

Enable verbose logging to see auxiliary loss values:

```yaml
verbose: true
```

Check the logs for auxiliary loss values during training to ensure they're being computed correctly.
