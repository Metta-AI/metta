# Mamba + DRAMA Integration Plan

## Component Layout
- `agent/src/metta/agent/components/mamba/__init__.py`: public exports for Mamba utilities and components.
- `agent/src/metta/agent/components/mamba/backbone.py`: `MambaBackboneComponent` that consumes latent embeddings (`mamba_input`) and returns transformed features (`mamba_output`). Handles config parsing, cache helpers, and hooks the Mamba block stack into Metta's TensorDict flow.
- `agent/src/metta/agent/components/mamba/config.py`: defines `MambaBackboneConfig` mirroring upstream `MambaConfig` while enforcing local naming conventions and validation (model dims, attention indices, etc.).
- `agent/src/metta/agent/components/mamba/modules/`: copied selective state-space modules trimmed to required APIs.
  - `block.py`, `mamba.py` (from `mamba_simple`), `mamba2.py`, `mamba2_simple.py`, `mha.py`, `mlp.py`, `ssd_minimal.py` (if needed) with inference cache logic and kernel hooks intact.
  - `kernels/ops/`: Triton launchers (layer norm, selective scan, selective state update) copied from `external/mamba/mamba_ssm/ops`.
  - `kernels/csrc/`: CUDA sources from `external/mamba/csrc` for causal conv1d and fused kernels, retained for builds.
- `agent/src/metta/agent/components/mamba/model.py`: localized `MixerModel` (and optional slim `MambaLMHeadModel`) adapted to TensorDict inputs and our config class.
- `agent/src/metta/agent/components/drama/__init__.py`: exports for DRAMA-specific components.
- `agent/src/metta/agent/components/drama/world_model.py`: `DramaWorldModelComponent` orchestrating encoder, stochastic latent heads, action-conditioned Mamba stack, reward/termination/value heads; exposes outputs keyed for `ActionProbs` integration.
- `agent/src/metta/agent/components/drama/config.py`: `DramaWorldModelConfig` aggregating structural parameters (categorical dims, hidden sizes, dropout) plus sub-configs for encoder/decoder, reward head, etc.
- `agent/src/metta/agent/components/drama/modules/`: supporting modules split from DRAMA sources.
  - `encoder.py`, `decoder.py`, `dist_heads.py`, `reward_head.py`, `termination_head.py`, `lambda_returns.py`, `normalization.py` (from `world_models.py` and `agents.py`).
  - `mamba_wrapper.py`: renamed `MambaWrapperModel` tuned for action conditioning.
  - `actor_head.py`: VecNormalize, actor/critic heads, two-hot loss decoding adapted to TensorDict outputs.
  - `kernels/`: DRAMA-specific kernel variants if they diverge from the base package.
- `agent/src/metta/agent/components/component_config.py`: register `MambaBackboneComponentConfig` and `DramaWorldModelComponentConfig` so `PolicyAutoBuilder` can instantiate them.
- `agent/src/metta/agent/policies/mamba_policy.py`: policy wiring the backbone component with existing action/value heads.
- `agent/src/metta/agent/policies/drama_policy.py`: policy stacking the DRAMA world model component and adapting outputs to our action/value heads (initially via DRAMA actor-critic wrapper).

## Code Relocation & Manifests
1. Copy `external/mamba/mamba_ssm/modules/{block.py,mamba_simple.py→mamba.py,mamba2.py,mamba2_simple.py,mha.py,mlp.py}` into `components/mamba/modules/`, adjusting imports to local paths and keeping kernel guards.
2. Copy `external/mamba/mamba_ssm/models/mixer_seq_simple.py` into `components/mamba/model.py`; remove HF `from_pretrained/save_pretrained` helpers and adapt to TensorDict inputs.
3. Recreate minimal `InferenceParams`/`update_graph_cache` locally (or import from copied `generation.py` if needed) to preserve inference cache support.
4. Mirror upstream kernels: `external/mamba/mamba_ssm/ops` → `components/mamba/modules/kernels/ops/`; `external/mamba/csrc` → `components/mamba/modules/kernels/csrc/`. Provide build wrapper under the `agent` package for CUDA/Triton extensions.
5. Copy DRAMA’s `mamba_ssm/models/mixer_seq_simple.py` and `config_mamba.py` into `components/drama/mamba_wrapper.py` and `components/drama/config.py`, renaming classes to avoid collisions with base Mamba versions.
6. Split `external/drama/sub_models/world_models.py` into module files under `components/drama/modules/`, ensuring each class/function has a focused location with local imports.
7. Extract relevant structures from `external/drama/agents.py` (VecNormalize, ActorCritic, two-hot loss integration) into `components/drama/modules/actor_head.py`, adapting to TensorDict outputs and our action/value keys.
8. Retain DRAMA’s kernel directory if it differs; otherwise reference the shared Mamba kernel package.
9. Update `agent/pyproject.toml` (and `uv.lock`) with dependencies: `einops`, `kornia`, `pytorch-warmup`, `line-profiler` (optional), `torchtune` (if RMSNorm import kept), Triton-compatible version pins, `causal-conv1d`.
10. Add build steps/documentation in `agent/README.md` for compiling CUDA/Triton kernels during install, including environment prerequisites (PyTorch ≥ 2.1, CUDA toolkit).
11. Extend `agent/tests/` with smoke tests instantiating the new components, running dummy forward passes, and verifying compatibility with `ActionProbs`.
12. Document configuration knobs and integration notes in a new resource file (e.g., `agent/docs/mamba_drama_integration.md`) once components are in place.

## Follow-Up Tasks
- Inventory Triton and CUDA kernels to confirm build steps and runtime flags, ensuring both Mamba and DRAMA variants share compiled artifacts where possible.
- Validate dependency additions locally (formatting, type checking, unit tests) before wiring policies into training loops.
- Revisit modularization after initial integration to expose sharable components (encoders, latent heads, backbone) for future policies.
