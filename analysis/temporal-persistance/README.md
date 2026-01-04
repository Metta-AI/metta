# Temporal Persistence of Mechanistic Interpretability SAE Concepts (Proposal)

**Status:** Proposal / docs-only.

Human brains are extremely dynamic in their capacity to store memories for varying spans of time and with varying speed of onset/offset, fidelity and retrievability. 

Consider the difference in how our brains treat the following pieces of information: the location you last left your car keys, a fact read in a book or a moving comment made by a loved one.

The intelligence and efficiency in treating different types of information is likely also to be represented by our agents in some form. Inspecting this, will allow us to better understand their learning process and behaviour.

This module proposes analysis tools to **quantify the temporal persistence (“memory span”)** of Sparse Autoencoder (SAE) concepts extracted from LSTM policy states. It is designed to plug into the proposed mech interp SAEs pipeline.

## Motivation
- Generate **quantitative measures** of how long a concept stays active in memory.
- Track **signal strength over time** (how strongly it remains active while it persists), enabling comparisons like “short-but-strong” vs “long-but-weak” features.
- Enable **cross-policy comparisons** and training-time tracking.
- Inform **steering horizons**, allowing the possibility of informed clamping/attenuation of concepts.

## Data & Assumptions
- Uses existing SAE latents $( z_k(t) )$ from recorded LSTM activations.
- Benefits from **longer rollouts (≈200–1000 steps)** for stable estimates.
- Requires metadata: episode id, step idx, optional context tags (phase, spatial bins, objects).

## Proposed method
- **Event detection:** Threshold each SAE latent $(z_k(t))$ (e.g., percentile or z-score).
- **Run-length persistence:** For each activation event, measure time-to-deactivation; summarize with median and tail probabilities (KM-style survival).
- **Autocorrelation half-life:** Compute per-feature autocorrelation $(\rho_k(\Delta))$ over lags.
- **Event-aligned decay:** Average $(z_k(t_0+\Delta))$ across events.
- **(Optional) Influence over horizon:** Probe or ablate feature $(k)$ and measure action/reward impact vs horizon $(\Delta)$.

## Related Work
To my knowledge, no prior work measures per-concept temporal persistence of SAE features learned from LSTM policy states.
