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
