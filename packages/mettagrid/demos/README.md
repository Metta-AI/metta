# MettaGrid Demos README

This folder contains pure, framework-specific demos that prove our adapters work without relying on Metta's internal
training code. Each script is intentionally minimal and self-contained so external researchers (or CI) can copy/paste
and run them with only the target library installed.

---

## Goals of these demos

1. **Prove Compatibility**
   - Show each adapter (PettingZoo, Gymnasium/SB3, PufferLib) can reset/step/close cleanly
   - Run a short end-to-end "training" loop in the target ecosystem to exercise the whole API

2. **Keep Boundaries Clean**
   - No calls into Metta training utilities (`train.py`, simulation loops, etc.)
   - Only import the adapter (`MettaGridPettingZooEnv`, `MettaGridGymEnv`, `MettaGridPufferEnv`) + curriculum

3. **Be CI-Friendly**
   - Hard cap on steps/timesteps (≤~300) so they finish fast
   - Deterministic seeds and simple asserts to catch regressions

4. **Serve as Reference Code**
   - Researchers can see exactly how to wire MettaGrid into their pipelines
   - Internal devs can sanity-check changes against real external APIs

---

## Directory Structure

```text
./demos/
  ├── demo_train_pettingzoo.py   # PettingZoo Parallel API demo
  ├── demo_train_gym.py          # Gymnasium + Stable-Baselines3 (single-agent) demo
  ├── demo_train_puffer.py       # PufferLib demo
  └── README.md                  # (this file)
```

---

## How to run (locally or in CI)

All scripts are **PEP 723 compliant** with inline dependency specifications. From repo root:

```bash
# Run individual demos
uv run python demos/demo_train_pettingzoo.py
uv run python demos/demo_train_gym.py
uv run python demos/demo_train_puffer.py

# Or test all demos via GitHub Actions workflow
gh workflow run test-demo-environments.yml
```

> **Note**: If `uv` isn't available, you can extract dependencies from the `# /// script` headers and install with pip,
> or run inside the project's dev environment.

---

## What "success" looks like in each demo

### PettingZoo (`demo_train_pettingzoo.py`)

- Passes `pettingzoo.test.parallel_api_test`
- Executes random rollout + toy policy update loop (~300 steps)
- Asserts non-NaN rewards, sensible shapes, and episode cycling
- Multi-agent action spaces work correctly

### Gymnasium / SB3 (`demo_train_gym.py`)

- Single-agent wrapper satisfies SB3's expectations
- Trains a PPO agent for ~256 steps without errors
- Rollout verification shows trained model can act
- Vectorized `DummyVecEnv` test confirms parallelism

### PufferLib (`demo_train_puffer.py`)

- Environment constructs and steps with correct action data types
- Short random rollout + preference-based learning loop
- Compatible with PufferLib's high-throughput expectations
- Handles multi-agent Box action spaces correctly

---

## Framework Compatibility

We ensure MettaGrid plays nicely with the PyTorch RL stacks that game/multi-agent researchers use:

### **Tier 1** (actively tested in demos + CI)

- **Stable-Baselines3 (SB3)** – most common baseline suite for Gym environments
- **PettingZoo** – standard multi-agent RL API with extensive ecosystem
- **PufferLib** – vectorized, throughput-focused env/training toolkit

### **Tier 2** (compatible by design, tested in unit tests)

- **Tianshou** – works through PettingZoo ParallelEnv API (test_pettingzoo_env.py)
- **CleanRL** – works through Gymnasium/PettingZoo APIs (test_gym_env.py, test_pettingzoo_env.py)
- **MARLlib** – built on PettingZoo ParallelEnv (test_pettingzoo_env.py)

### **Tier 3** (should work, not regularly tested)

- **TorchRL (Meta/Facebook)** – official PyTorch RL lib with env abstractions
- **SampleFactory v2** – scalable on-policy RL for games
- **RLlib** – distributed RL framework (Ray); PettingZoo compatible

---

## Related Projects / Baselines We Care About

We primarily want to slot in next to popular open-source MARL / game RL stacks:

- **PettingZoo "classic" envs** (MAgent, SISL, Overcooked) – API parity, benchmarkability
- **MiniGrid / BabyAI** – curriculum + gridworld baselines (single & multi-agent)
- **SMAC / SMACv2** – cooperative multi-agent control benchmarks
- **MELTINGPOT** – social dilemma / generalization focus (JAX, but PettingZoo-compatible adapters exist)
- **SampleFactory v2 / ENVPOOL** – high-throughput env runners

These demos serve as proof that MettaGrid can drop into the same pipelines as the above. Future work: add benchmark
scripts that mirror their training configs for apples-to-apples comparisons.

---

## CI Integration

Demos are automatically tested via GitHub Actions (`.github/workflows/test-demo-environments.yml`):

```yaml
strategy:
  matrix:
    demo: [demo_train_pettingzoo, demo_train_puffer, demo_train_gym]
```

Each demo runs with:

- **45-second timeout** (expected completion ~5s)
- **Deterministic execution** for consistent CI results
- **Failure detection** for import errors, API changes, or hangs

---

## Future TODOs / Nice-to-haves

- **SuperSuit examples**: show how to wrap our PettingZoo env in common pre-processing stacks
- **TorchRL/Tianshou snippets**: add 20-line demos to reassure compatibility
- **Multi-agent Gym wrapper**: shared-reward or stacked-obs Gym env for SB3 if demand arises
- **Performance benchmark**: quick throughput comparison vs reference envs
- **Cleaner config injection**: CLI flags to tweak curriculum/env params from demos
- **Notebook versions**: Jupyter notebooks for interactive exploration

---

## Contributing / Updating the demos

- **Keep them short** (< ~2 minutes total runtime in CI)
- **Prefer deterministic seeds**, explicit asserts, and clear prints over heavy logging
- **If you change an adapter's API**, update the corresponding demo and tests in the same PR
- **Use these scripts as acceptance tests** when refactoring core env APIs
- **Follow PEP 723** for dependency specifications in script headers

---

## Troubleshooting

### Common Issues

**ImportErrors**: The demos only import the specific adapter + curriculum. If something else slips in, that's a red
flag.

**PettingZoo test closes env**: Don't try to reuse the same env instance after `parallel_api_test`; create a new one.

**Action data types**: Make sure actions are `int32` scalars for the discrete action space.

**SB3 shape complaints**: Verify you're using the single-agent wrapper and that obs/action spaces match SB3
expectations.

### Gymnasium vector envs (AsyncVectorEnv)

Use the Gymnasium wrapper so the vector worker can read `metadata`:

```python
from gymnasium.vector import AsyncVectorEnv
from pufferlib.emulation import GymnasiumPufferEnv
from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
from mettagrid.simulator import Simulator
from mettagrid.config.mettagrid_config import MettaGridConfig

def make_env():
    return GymnasiumPufferEnv(
        env_creator=MettaGridPufferEnv,
        env_kwargs={"simulator": Simulator(), "cfg": MettaGridConfig(...)},
    )

vec_env = AsyncVectorEnv([make_env] * 4)
```

Note: Gymnasium’s vector API is single-agent; the wrapper flattens joint obs/actions into one Gym space, so your policy
must already handle that shape.

### Debug Commands

```bash
# Check demo dependencies
uv run python -c "import sys; print(sys.path)"

# Test specific adapter import
uv run python -c "from mettagrid import MettaGridPettingZooEnv; print('Import works')"

# Verbose demo run
uv run python demos/demo_train_gym.py --verbose  # (if supported)
```

---

## Architecture Context

These demos exercise the **external compatibility layer** of MettaGrid's architecture:

```text
┌─────────────────────────────────────────────────────────────┐
│ EXTERNAL RESEARCH FRAMEWORKS                                │
├─────────────────────────────────────────────────────────────┤
│ Stable-Baselines3 │ PettingZoo/MARL │ PufferLib/CleanRL    │
│        ↓          │        ↓        │         ↓            │
│ SingleAgentGymEnv │ PettingZooEnv   │ PufferEnv            │
└─────────────────────┬─────────────────┬─────────────────────┘
                      │                 │
                      ▼                 ▼
                ┌─────────────────────────────┐
                │     MettaGridCore (C++)     │
                └─────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│ SOFTMAX STUDIO TRAINING (Primary)                          │
├─────────────────────────────────────────────────────────────┤
│          MettaTrainer → PufferMettaGridEnv                  │
└─────────────────────────────────────────────────────────────┘
```

The demos ensure the **top layer** works correctly without involving the **bottom layer** (Softmax Studio training).
