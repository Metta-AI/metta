Transformer Stack Audit (Post-929.9 Changes)
===========================================

Context
-------
Baseline run `relh.gtrxl.929.9` (commit `73bca6e6ec`, 2025-09-29 19:04:08 PDT) delivered ~100k SPS and strong arena basic easy shaped performance. Subsequent refactors introduced a large set of changes across the transformer components, recipes, and training loop. This document inventories those additions so we can selectively reapply the wins after reverting to the baseline transformer.

Summary of Additions
--------------------

### 1. Transformer Backbone Refactor (`agent/components/transformer_core.py`)

- Introduced `TransformerBackboneVariant.backbone_defaults()` with new dimension/dropout defaults (wider 48-dim models, 2 layers, tuned memory settings).
- Added `policy_defaults()` to surface variant-specific optimizer hints (`learning_rate_hint`, `manual_init`, `strict_attr_indices`).
- Extended enum with `SLIDING` variant and wiring for sliding-window transformer implementation.
- Added optional config fields (`max_cache_size`, `pool`) for non-GTrXL variants.

### 2. Sliding Transformer Rewrite (`agent/components/sliding_transformer.py`)

- Replaced original minimal sliding transformer with a feature-rich module supporting configurable cache sizes, pooling, and head counts.
- Enables reuse of transformer scaffolding for ViT/sliding variants.

### 3. Auxiliary Token Module (`agent/components/aux_tokens.py`)

- New helper to embed reward/reset/action history into the transformer when `use_aux_tokens=True`.
- Not used by baseline ABES runs but useful for richer policies.

### 4. Transformer Policy Enhancements (`agent/policies/transformer.py`)

- Memory packing/unpacking rewritten to operate layer-by-layer; supports mixed per-layer cache lengths.
- `_cast_floating_tensors` narrowed to a whitelist, allowing bf16 storage under AMP while keeping critical heads in fp32.
- Multi-step forward path now discards returned memory to avoid leakage between disjoint sequences.
- Logging/diagnostics improved for memory norms.

### 5. Transformer Module Fixes (`agent/components/transformer_module.py`)

- Ensured positional emb/attention masks cast to correct device/dtype on-the-fly.
- Added safety around flash-attn checkpoint lambda capture.

### 6. Obs Encoder / Policy Auto Builder Updates

- Tuned default perceiver latent heads/layers to align with new transformer widths.
- Auto-builder tightened integration with variant defaults (uses `policy_defaults`).

### 7. Training Loop Upgrades (`metta/rl/*`, `metta/tools/train.py`)

- Added automatic AMP support (autocast + GradScaler) when `amp_enabled`.
- TrainTool configures AdamW, weight decay, warmup, min LR, sequence-length curriculum when a transformer policy is used.
- SDPA backend helper (`agent/util/torch_backends.py`) sets flash/mem-efficient attention priorities on modern PyTorch builds.
- Stopwatch instrumentation across rollout + optimizer steps for profiling.

### 8. Recipes and Variants (`experiments/recipes/abes/*`)

- Recipes rely on variant defaults instead of fully hard-coded dimensions, easing shared tuning across policies.
- Added dedicated sliding-transformer recipe module.

Notes for Cherry-Picking Later
------------------------------

- **Keep SDPA helper** (`agent/util/torch_backends.py`) if we want automatic flash attention toggling; low risk, easy to reapply.
- **Optimizer hints** (`policy_defaults`) help align learning rate/initialization with variant; worth reinstating once core perf is back.
- **Layer-by-layer memory copy** is more flexible but adds overhead; consider a hybrid approach (fast path for contiguous caches).
- **AMP / AdamW** provide long-term benefits but need profiling; we can reintroduce them behind explicit config flags to avoid regressing SPS.
- **Aux tokens & sliding transformer** are orthogonal features and can be cherry-picked later if needed.

Next Steps
----------

1. Revert transformer-related files to commit `73bca6e6ec` to restore the known-good architecture and training flow.
2. Rebenchmark to confirm SPS and heart.get trajectories match the baseline run.
3. Incrementally reapply the desired improvements from the list above, validating performance after each addition.

