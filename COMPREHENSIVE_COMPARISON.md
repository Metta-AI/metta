# Comprehensive Comparison: ComponentPolicy (YAML) vs Fast (PyTorch)

## Layer-by-Layer Architecture Comparison

### 1. Observation Processing
| Layer | ComponentPolicy (YAML) | Fast (PyTorch) | Match? |
|-------|------------------------|----------------|---------|
| Input Shape | ObsTokenToBoxShaper | Token to box conversion in encode_observations | ✓ |
| Normalization | ObservationNormalizer | x / self.max_vec | ✓ |

### 2. CNN Layers
| Layer | ComponentPolicy | Fast | Match? |
|-------|-----------------|------|---------|
| cnn1 | Conv2d(25, 64, k=5, s=3) + ReLU (default) | Conv2d(25, 64, k=5, s=3) + F.relu | ✓ (after relu fix) |
| cnn2 | Conv2d(64, 64, k=3, s=1) + ReLU (default) | Conv2d(64, 64, k=3, s=1) + F.relu | ✓ (after relu fix) |

### 3. Feature Processing
| Layer | ComponentPolicy | Fast | Match? |
|-------|-----------------|------|---------|
| flatten | Flatten | flatten | ✓ |
| fc1 | Linear(64, 128) + ReLU (default) | Linear(64, 128) + F.relu | ✓ (after relu fix) |
| encoded_obs | Linear(128, 128) + ReLU (default) | Linear(128, 128) + F.relu | ✓ (after relu fix) |

### 4. LSTM Core
| Layer | ComponentPolicy | Fast | Match? |
|-------|-----------------|------|---------|
| LSTM layers | 2 | 2 | ✓ |
| Hidden size | 128 | 128 | ✓ |
| Initialization | Orthogonal, bias=1 | Orthogonal, bias=1 | ✓ |

### 5. Critic Branch
| Layer | ComponentPolicy | Fast | Match? |
|-------|-----------------|------|---------|
| critic_1 | Linear(128, 1024) + Tanh | Linear(128, 1024) + tanh | ✓ |
| value_head | Linear(1024, 1) + no activation | Linear(1024, 1) + no activation | ✓ |

### 6. Actor Branch
| Layer | ComponentPolicy | Fast | Match? |
|-------|-----------------|------|---------|
| actor_1 | Linear(128, 512) + ReLU (default) | Linear(128, 512) + F.relu | ✓ (after relu fix) |
| action_embeds | Embedding(100, 16) | Embedding(100, 16) | ✓ |
| bilinear | MettaActorSingleHead | Custom bilinear with einsum | ✓ |

## Initialization Comparison

### Weight Initialization
| Component | ComponentPolicy | Fast | Match? |
|-----------|-----------------|------|---------|
| Conv2d | Orthogonal (default from ParamLayer) | pufferlib.pytorch.layer_init (Orthogonal) | ✓ |
| Linear | Orthogonal (default from ParamLayer) | pufferlib.pytorch.layer_init (Orthogonal) | ✓ |
| LSTM | Orthogonal weights, bias=1 | Orthogonal weights, bias=1 | ✓ |
| Embeddings | Orthogonal then scale to max 0.1 | Orthogonal then scale to max 0.1 | ✓ |
| Bilinear W | Kaiming uniform | Kaiming uniform | ✓ |

## Training Mode Differences (FIXED)

### Tensor Reshaping Issues
| Issue | Before Fix | After Fix | Status |
|-------|------------|-----------|---------|
| full_log_probs shape | Used T instead of TT | Uses TT correctly | ✓ FIXED |
| value shape | Lost last dimension | Preserves shape [B, TT, 1] | ✓ FIXED |

## Remaining Potential Issues to Verify

### 1. Layer Initialization Order
- ComponentPolicy builds all components then sets them up
- Fast initializes everything in __init__
- **Potential Issue**: Initialization order might affect random seeds

### 2. Effective Rank Regularization
- ComponentPolicy critic_1 has `effective_rank: true`
- Fast doesn't implement this
- **Impact**: Minor - affects regularization during training

### 3. Weight Clipping
- ComponentPolicy has built-in weight clipping in ParamLayer
- Fast doesn't implement weight clipping
- **Impact**: Could affect training stability

### 4. L2 Init Loss
- ComponentPolicy tracks initial weights for L2-init regularization
- Fast doesn't implement this
- **Impact**: Could affect fine-tuning and catastrophic forgetting

### 5. Analyze Weights
- ComponentPolicy has weight analysis capabilities
- Fast doesn't implement this
- **Impact**: Only affects debugging/monitoring

## Summary of Fixes Applied

1. ✓ **ReLU Activations** (commit 4a4041e3b): Added missing F.relu after cnn1, cnn2, fc1, encoded_obs, actor_1
2. ✓ **BPTT Shape Mismatch** (commit 7da38bc01): Fixed full_log_probs and value tensor reshaping

## Conclusion

After the two fixes:
- **Architecture**: Now matches exactly (layers, dimensions, activations)
- **Forward Pass**: Now produces correct shapes and transformations
- **Initialization**: Matches for all major components

The main remaining differences are in auxiliary features (weight clipping, effective rank, L2-init) that might affect training dynamics but shouldn't cause catastrophic failure like the missing ReLUs and shape mismatches did.