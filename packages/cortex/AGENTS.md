# Cortex Agent Playbook

These instructions extend the repository-wide guidance in the root `AGENTS.md`.
Always read both documents before making changes inside `packages/cortex`.

## 1. Establish the PyTorch Ground Truth

- Every Triton kernel must have a numerically correct PyTorch reference in
  `src/cortex/kernels`. Extract the minimal computation needed and land that
  first if it does not already exist.
- Keep signatures aligned between the PyTorch and Triton paths so they can be
  swapped drop-in; do not introduce behaviour that only exists on the Triton
  side.
- Validate new PyTorch code with unit tests before starting the Triton port.

## 2. Implement the Triton Kernel

- Build the Triton implementation in `src/cortex/kernels/*_triton/`, covering
  both forward and backward passes. Match tensor shapes, dtype expectations,
  and semantics of the PyTorch ground truth exactly.
- Do **not** change the PyTorch implementation to call into Triton, short-circuit
  logic, or otherwise mask correctness issues. The PyTorch path is the sole
  source of truth.
- Prefer small, well-named helper utilities to keep the Triton kernels readable;
  annotate inputs and constexpr parameters when it aids comprehension.

## 3. Testing Requirements

- Add or extend tests in `packages/cortex/tests/` that:
  - Compare the Triton output against the PyTorch reference across a range of
    shapes, dtypes, and reset patterns.
  - Check gradient parity (forward + backward) using autograd when applicable.
  - Exercise error handling for unsupported arguments.
- Run relevant slices locally before requesting review:
  - `pytest packages/cortex/tests/test_<kernel>.py`
  - `uv run pytest` when touching multiple areas
- Capture failure cases with informative asserts (e.g. tolerances, shapes) to
  make future debugging easier.

### Evaluations policy (do not edit for testing)

- Do not modify anything under `packages/cortex/evaluations/` to validate kernels
  or fixes. Always add or extend unit tests under `packages/cortex/tests/` instead.
- Avoid adding new presets, flags, or CLI changes in the evaluation harness for
  ad‑hoc sanity checks. Express such checks as pytest tests so they run in CI and
  serve as regressions.
- If an evaluation change is absolutely necessary (e.g., to expose a tested feature
  in a demo), get explicit maintainer approval and document the rationale in the PR.

## 4. Iterate Until Tests Pass

- Use the PyTorch reference to debug numerical drift; only relax tolerances when
  you can justify the precision loss.
- Keep the PyTorch implementation unchanged while iterating—no shortcuts such
  as dispatching back into Triton or mutating global state.
- If the Triton kernel requires additional metadata (e.g., segmentation masks or
  state buffers), plumb that data explicitly through the Triton wrapper rather
  than altering the reference implementation.

## 5. Wire Up Callers

- When integrating a new kernel with higher-level cells or layers inside
  `src/cortex/cells/`, ensure both the PyTorch and Triton paths remain available
  and gated in the appropriate feature flags or device checks.
- Document any new environment requirements (e.g., minimum compute capability) in
  the module docstring or a nearby README.

## 6. Required Quality Checks

- Format and lint touched Python files: `ruff format` then `ruff check`.
- Run mypy on the modified modules (or the narrowest package that contains them):
  `uv run mypy packages/cortex/src/cortex/...`.
- Re-run the targeted pytest command after linting to guard against regressions.

##  Triton Kernel Notes

- Triton prefers dot operands with tile dimensions ≥16. For kernels that must
  handle smaller tiles (e.g., batch padding of size 8), emulate matmul with
  explicit reductions instead of relying on `tl.dot`.
- Keep shared memory budgets in mind—large tiles or extra staging buffers can
  exceed 100 KB on common GPUs. Profile both forward and backward kernels after
  changing tile sizes or adding scratch space.
- Numerical parity: default tolerances for Triton vs. PyTorch parity checks are
  `rtol=1e-3, atol=1e-2` for forward/last-state comparisons and `rtol=1e-3,
  atol=1e-1` for gradient checks. Tighten only when justified and update the
  tests accordingly.
- Regression tests should cover both forward correctness and autograd parity.
  Re-run the dedicated `test_<kernel>_reset_forward_backward_match_backends`
  style tests (or add one if missing) whenever kernel math changes.
- Unknown coverage: multi-layer wiring, projection heads, and non power-of-two
  hidden sizes are still PyTorch-only in several cells. When extending Triton
  support, document the new limits and keep the PyTorch path as reference.

### Triton Kernel Safety & Debugging (sLSTM/mLSTM)

- Bounds safety on tail tiles
  - When the grid tiles B×DH (or B×L, etc.), use `boundary_check=(0, 1)` on every
    `tl.load`/`tl.store` that operates on those block pointers. Do not assume `B % siz_B == 0`.
  - If autotuning tries multiple `siz_B` (e.g., 16 and 32), padding B to a single
    multiple is insufficient. Prefer boundary checks over relying on divisibility.

- Compute sensitive math in float32, cast for storage
  - Accumulate gate preactivations (Ī, F̄, Z̄, Ō) and intermediate states in `tl.float32`.
  - Apply `log`, `exp`, `sigmoid`, and `tanh` in float32, then cast the final outputs
    back to the kernel `DTYPE` only at store sites.

- Stable nonlinearity patterns
  - Tanh: avoid `(1 - exp(-2x)) / (1 + exp(-2x))` (inf/inf → NaN). Use a stable form:
    `tanh(x) = sign(x) * (1 - 2 / (1 + exp(2*abs(x))))` in float32.
  - Stabilized m_next rule (sLSTM): `m_next = is_first ? Ī : max(Ī, m + log_sigmoid(F̄))`.
  - Always add a small epsilon (e.g., `1e-6`) to denominators that can approach 0.

- Resets and segmentation
  - Respect per‑timestep resets in forward and backward. Zero carry-over across reset
    boundaries (e.g., inter‑chunk contributions) to match PyTorch step semantics.
  - Prefer representing resets as explicit masks passed to kernels rather than implicit
    assumptions in the caller.

- Autotuning and shared memory
  - Be mindful of SMEM budgets; accumulators are float32. Keep tile sizes within the
    `CORTEX_TRITON_SMEM_SOFT_LIMIT` (defaults used in wrappers) or adjust grid/block sizes.

- Repro and fallback toggles
  - `CORTEX_DISABLE_TRITON=1` (or `CORTEX_FORCE_PYTORCH=1`) forces the PyTorch reference for
    quick triage. If PyTorch is stable and Triton isn’t, focus debug on kernel math/tiling.

- Minimal parity harness
  - Build a small test that runs both backends on randomized shapes/dtypes (fp16/bf16/fp32),
    non‑multiple batch sizes (e.g., 17/31/33), and reset patterns. Assert `no NaN/Inf` and
    numeric closeness within agreed tolerances.

- Upstream call‑site guardrails (useful while debugging)
  - Add non‑finite checks around logits/states in the high‑level modules and skip optimizer
    steps when any grad is non‑finite to avoid corrupting weights during kernel bring‑up.
