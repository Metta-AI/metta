# Approach and caveat

When you’re using ReLU activations, the “dead neuron” check is simple because the condition is baked into the math: a
ReLU is either on or off. If its output is always zero, you know it’s permanently inactive — and you can detect that
just by looking at the outputs after a forward pass. That’s possible because ReLU has a literal zero-derivative region,
so a neuron that never crosses zero will never receive a gradient again. The situation with ReLU activation is unique in
having a clear definition of a dead or dormant neuron. In general, it’s not so clear — a good definition is that a
neuron doesn’t have any discriminatory power over values in the actual input distribution, and that this state is locked
in over time. To meet that definition you need a region where the output is constant and the derivative is zero, and
this simply doesn’t happen for tanh, sigmoid, or leaky ReLU. In those cases, the derivative is always nonzero at zero,
and what you’re really measuring is whether neurons are saturated, meaning they spend most of their time in the flat
tails of the activation curve. That’s a harder call and depends on the data distribution.

That's why the code for these checks looks more complex and "intrusive." To measure saturation or vanishing gradients,
we can't just look at weights or outputs after training; we need to tap into both the forward and backward passes. The
forward hooks record what each neuron is doing with the real data (activation means, variances, slopes), while the
backward hooks track whether gradients are still flowing through it.

For a [technical discussion](hook_architecture.md) of how hooks are implemented in the forward and backward passes, see
`hook_architecture.md`.

# Neuron Health Diagnostics Summary

This document summarizes diagnostic measures for detecting dead, dormant, or saturated neurons — separated by activation
type.

---

## 🟩 ReLU-family activations (ReLU, ReLU6, etc.)

| **Measure**               | **What it measures**             | **Interpretation**                          | **Needs data?** | **Implemented** | **Notes**                                          |
| ------------------------- | -------------------------------- | ------------------------------------------- | --------------- | --------------- | -------------------------------------------------- |
| Activation on-rate        | Frequency of neuron firing       | 0 → dead; low → dormant; mid → healthy      | ✅ Yes          | ✅ Yes          | Canonical dead ReLU metric                         |
| Pre-activation statistics | Mean & variance of z = W·x + b   | mean ≪ 0, var small → dead                  | ✅ Yes          | ❌ No           | Explains why it’s dead                             |
| Gradient flow (EMA)       | How often backward signal passes | 0 → neuron never updated                    | ✅ Yes          | ✅ Yes          | Equivalent to derivative magnitude × upstream grad |
| Fisher proxy              | Long-term information content    | Near-zero → unused parameter                | ✅ Yes          | ✅ Yes          | Cheap curvature estimate                           |
| Activation entropy        | Variability of outputs           | Low entropy near 0 → always off             | ✅ Yes          | ❌ No           | Detects static outputs                             |
| Ablation sensitivity      | Functional importance            | Δ≈0 → redundant/dead                        | ✅ Yes          | ❌ No           | Costly but definitive                              |
| Weight norm               | Magnitude of w                   | Very small → negligible influence           | ❌ No           | ❌ No           | Weak heuristic                                     |
| Bias negativity           | Offsetting bias                  | b ≪ −‖w‖ → likely always z<0                | ❌ No           | ❌ No           | Weight-only prior                                  |
| Positive-weight sum       | Geometry of reachable z>0 region | Σmax(wᵢ,0)+b < 0 → cannot fire if inputs ≥0 | ❌ No           | ❌ No           | Works only if prev layer ReLU                      |
| Redundancy (cosine sim)   | Duplicated neurons               | >0.99 → redundant                           | ❌ No           | ❌ No           | Weight-space check                                 |

### Implementation and formula notes

- **Activation on-rate:** `on_rate = (a > 0).float().mean()`
- **Gradient flow (EMA):** track `ema(|∂L/∂a|)` or fraction of active gradients.
- **Fisher proxy:** maintain EMA of `(∂L/∂θ)²` during backprop to measure importance.
- **Weight-only heuristics:** `Σmax(w,0)+b`, bias negativity, and weight norm can be computed at epoch end.

---

## 🟦 Smooth activations (tanh, sigmoid, GELU, Swish, etc.)

| **Measure**                   | **What it measures**          | **Interpretation**                      | **Needs data?**       | **Implemented**                      | **Notes**                          |
| ----------------------------- | ----------------------------- | --------------------------------------- | --------------------- | ------------------------------------ | ---------------------------------- | ----- | ------------------- |
| Pre-activation magnitude      | How deep neurons sit in tails | High → saturated                        | ✅ Yes                | ❌ No                                | Direct, interpretable              |
| Derivative magnitude          | Local slope                   | Small → saturated, vanishing grad       | ✅ Yes                | ✅ Yes                               | Analogue of gradient flow for ReLU |
| Average gradient norm         | Effective gradient strength   | Low → poor gradient flow                | ✅ Yes                | ❌ No                                | Core training-health metric        |
| Activation entropy / variance | Output diversity              | Low → outputs constant ⇒ saturated      | ✅ Yes                | ❌ No                                | Easy to monitor                    |
| Gradient variance per layer   | Stability of backprop         | Collapse → saturation or dead gradients | ✅ Yes                | ❌ No                                | Layer-level check                  |
| Fisher proxy                  | Parameter importance          | Low → unused parameter                  | ✅ Yes                | ✅ Yes                               | Data-driven                        |
| Ablation sensitivity          | Functional importance         | Δ≈0 → redundant                         | ✅ Yes                | ❌ No                                | Confirms functional irrelevance    |
| Bias/weight ratio heuristic   | Static saturation risk        |                                         | b                     | / (‖w‖+ε) > 2–3 → likely saturated   | ❌ No                              | ❌ No | Weight-only prior   |
| Pre-activation prior          | Prob(                         | z                                       | >T) under z~N(b,‖w‖²) | Large probability → likely saturated | ❌ No                              | ❌ No | Rough offline prior |

### Implementation and formula notes

- **Derivative magnitude:** `mean(|f'(z)|)` where `f'(z)=1−tanh²(z)` or `σ(z)(1−σ(z))`.
- **Average gradient norm:** `mean(|∂L/∂a|)` or `mean(|∂L/∂z|)` recorded during backprop.
- **Pre-activation magnitude:** `frac_saturated = (|z| > 2.5).mean()` (tanh) or `|z| > 5` (sigmoid).
- **Fisher proxy:** running EMA of squared gradients for parameters.
- **Bias/weight ratio heuristic:** offline diagnostic using `|b| / (‖w‖+ε)`.

---

## 🧭 Conceptual summary

| **Category**                  | **ReLU** (piecewise linear)          | **Smooth (tanh/sigmoid/GELU)**                     |
| ----------------------------- | ------------------------------------ | -------------------------------------------------- | ----- | ---------------- | --- | ------------- |
| Kind of inactivity            | _Structural_ (hard zero region)      | _Statistical_ (data-dependent saturation)          |
| Derivative behavior           | Exactly zero on one side             | Small but nonzero in tails                         |
| Can infer from weights alone? | Partially (geometry & bias)          | No — must use data                                 |
| Best metrics                  | On-rate, gradient flow, Fisher proxy |                                                    | f′(z) | , gradient norm, | z   | tail fraction |
| Functional test               | Ablation sensitivity                 | Same                                               |
| Interpretation                | Permanently zero output, no gradient | Output nearly constant, tiny gradient, recoverable |

---

### ✅ TL;DR

- **ReLU:** Deadness is _binary and structural._ Diagnose via _activation on-rate_, _gradient flow_, or _Fisher proxy_;
  sometimes weights/biases suffice.
- **Smooth activations:** Saturation is _continuous and data-dependent._ Measure _derivative magnitude_, _average
  gradient norm_, or _activation variance_ **in place**.
- **Universal:** Functional tests (_ablation sensitivity_, _Fisher proxy_) work for both.
