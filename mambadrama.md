# Mamba + DRAMA Integration Plan

## Alignment with Sliding Transformer Architecture
- Mirror the `SlidingTransformerConfig`/`SlidingTransformer` pattern: each new component exposes a `ComponentConfig`, operates on `TensorDict` inputs, and manages rollout vs. training behavior (cache updates, positional state) internally.
- Store per-env cache metadata explicitly (e.g., `mamba_position`, KV states) so the trainer can follow the existing `transformer_position` conventions introduced on `richard-transformer`.
- Accept and emit the same auxiliary signals used by the sliding transformer (`last_actions`, `rewards`, `dones`, `truncateds`, `training_env_ids`) to ensure seamless integration with `CoreTrainingLoop` bookkeeping.
- Provide `get_agent_experience_spec()` entries for any new state (positions, cache tensors) similar to `SlidingTransformer.get_agent_experience_spec` so replay buffers allocate the right keys.
- Offer PolicyAutoBuilder-friendly wiring (configs + components) like `ViTSlidingTransConfig`, enabling Mamba/DRAMA policies to slot into recipes without bespoke code.

## Component Layout
- `agent/src/metta/agent/components/mamba/__init__.py`: public exports for Mamba utilities and components.
- `agent/src/metta/agent/components/mamba/config.py`: `MambaBackboneConfig` mirroring upstream `MambaConfig`, adding validation for dims, SSM/attention mix, and positional/token options aligned with sliding transformer metadata.
- `agent/src/metta/agent/components/mamba/backbone.py`: `MambaBackboneComponent` that consumes latent embeddings (`mamba_input`) plus optional auxiliary tokens, produces `mamba_output`, and manages rollout/training caches (`mamba_k_cache`, `mamba_v_cache`, `mamba_position`).
- `agent/src/metta/agent/components/mamba/model.py`: localized `MixerModel` (and optional slim `MambaLMHeadModel`) adapted to TensorDict inputs, configurable token packing (CLS/reward/action tokens), and cache helpers consistent with sliding transformer API.
- `agent/src/metta/agent/components/mamba/modules/`: copied selective state-space modules trimmed to required APIs.
  - `block.py`, `mamba.py` (from `mamba_simple`), `mamba2.py`, `mamba2_simple.py`, `mha.py`, `mlp.py`, `ssd_minimal.py` (if needed) with inference cache logic and kernel hooks intact.
  - `kernels/ops/`: Triton launchers (layer norm, selective scan, selective state update) copied from `external/mamba/mamba_ssm/ops`.
  - `kernels/csrc/`: CUDA sources from `external/mamba/csrc` for causal conv1d and fused kernels, retained for builds.
- `agent/src/metta/agent/components/drama/__init__.py`: exports for DRAMA-specific components.
- `agent/src/metta/agent/components/drama/config.py`: `DramaWorldModelConfig` aggregating structural parameters (categorical dims, hidden sizes, dropout) plus sub-configs for encoder/decoder, reward head, etc., with hooks for cache lengths and auxiliary tokens.
- `agent/src/metta/agent/components/drama/world_model.py`: `DramaWorldModelComponent` orchestrating encoder, stochastic latent heads, action-conditioned Mamba stack, reward/termination/value heads; exposes outputs keyed for `ActionProbs`, and mirrors rollout/training cache management (positions + KV) akin to `SlidingTransformer`.
- `agent/src/metta/agent/components/drama/modules/`: supporting modules split from DRAMA sources.
  - `encoder.py`, `decoder.py`, `dist_heads.py`, `reward_head.py`, `termination_head.py`, `lambda_returns.py`, `normalization.py` (from `world_models.py` and `agents.py`).
  - `mamba_wrapper.py`: renamed `MambaWrapperModel` tuned for action conditioning, sharing cache helpers with the backbone component.
  - `actor_head.py`: VecNormalize, actor/critic heads, two-hot loss decoding adapted to TensorDict outputs.
  - `kernels/`: DRAMA-specific kernel variants if they diverge from the base package (otherwise reuse shared Mamba kernels).
- `agent/src/metta/agent/components/component_config.py`: register `MambaBackboneComponentConfig` and `DramaWorldModelComponentConfig` for PolicyAutoBuilder use.
- `agent/src/metta/agent/policies/mamba_policy.py`: policy wiring the backbone component with existing action/value heads; optionally pack reward/reset/action tokens as sliding transformer does.
- `agent/src/metta/agent/policies/drama_policy.py`: policy stacking the DRAMA world model component and adapting outputs to `ActionProbs` (initially via DRAMA actor-critic wrapper) with consistent metadata handling.
- `agent/src/metta/agent/policies/mamba_sliding.py` / `drama_sliding.py` (optional): PolicyAutoBuilder configs similar to `ViTSlidingTransConfig` for turnkey usage.

## Code Relocation & Manifests
1. Copy `external/mamba/mamba_ssm/modules/{block.py,mamba_simple.py→mamba.py,mamba2.py,mamba2_simple.py,mha.py,mlp.py}` into `components/mamba/modules/`, adjusting imports to local paths and keeping kernel guards.
2. Copy `external/mamba/mamba_ssm/models/mixer_seq_simple.py` into `components/mamba/model.py`; remove HF `from_pretrained/save_pretrained` helpers, adapt to TensorDict inputs, and add token packing/caching functions parallel to `SlidingTransformer`.
3. Recreate minimal `InferenceParams`/`update_graph_cache` locally (or import from copied `generation.py`) to preserve inference cache support and align with rollout/training split.
4. Mirror upstream kernels: `external/mamba/mamba_ssm/ops` → `components/mamba/modules/kernels/ops/`; `external/mamba/csrc` → `components/mamba/modules/kernels/csrc/`. Provide build wrapper under the `agent` package for CUDA/Triton extensions.
5. Copy DRAMA’s `mamba_ssm/models/mixer_seq_simple.py` and `config_mamba.py` into `components/drama/mamba_wrapper.py` and `components/drama/config.py`, renaming classes to avoid collisions and wiring in sliding-style positional/caching options.
6. Split `external/drama/sub_models/world_models.py` into module files under `components/drama/modules/`, ensuring each class/function has a focused location with local imports and TensorDict-friendly signatures.
7. Extract relevant structures from `external/drama/agents.py` (VecNormalize, ActorCritic, two-hot loss integration) into `components/drama/modules/actor_head.py`, adapting to TensorDict outputs and our action/value keys.
8. Retain DRAMA’s kernel directory if it differs; otherwise reference the shared Mamba kernel package.
9. Update `agent/pyproject.toml` (and `uv.lock`) with dependencies: `einops`, `kornia`, `pytorch-warmup`, `line-profiler` (optional), `torchtune` (if RMSNorm import kept), Triton-compatible version pins, `causal-conv1d`.
10. Add build steps/documentation in `agent/README.md` for compiling CUDA/Triton kernels during install, including environment prerequisites (PyTorch ≥ 2.1, CUDA toolkit) and noting optional CPU fallbacks.
11. Extend `agent/tests/` with smoke tests instantiating the new components, running rollout vs. training passes (cache # shaped like sliding transformer), and verifying compatibility with `ActionProbs`.
12. Document configuration knobs and integration notes in `agent/docs/mamba_drama_integration.md`, highlighting parity with sliding transformer metadata requirements.
13. Add PolicyAutoBuilder recipes (e.g., `experiments/recipes/mamba_sliding.py`, `drama_sliding.py`) patterned after `abes_sliding_transformer.py`.

## Follow-Up Tasks
- Inventory Triton and CUDA kernels to confirm build steps and runtime flags, ensuring both Mamba and DRAMA variants share compiled artifacts where possible.
- Validate dependency additions locally (formatting, type checking, unit tests) before wiring policies into training loops.
- Revisit modularization after initial integration to expose sharable components (encoders, latent heads, backbone) for future policies, aligning with the sliding transformer component boundaries.
- Evaluate whether common positional encoding/token packing code can be abstracted across Sliding Transformer, Mamba backbone, and DRAMA world model to reduce duplication.
