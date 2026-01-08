# Recipe Validation

### Two-Tier Testing System

**CI Suite** (`ci_suite.py`) - Fast smoke tests

- **Purpose**: Verify recipes don't crash, not performance
- **Runs**: On every commit (will run on GitHub pre-merge soon)
- **Characteristics**: 10k timesteps, 5-minute timeouts, runs locally
- **Run with**: `metta ci --stage recipe-tests`

**Stable Suite** (`stable_suite.py`) - Performance validation

- **Purpose**: Track end-to-end performance (SPS, learning outcomes)
- **Runs**: During releases on remote infrastructure
- **Characteristics**: 100M-2B timesteps, multi-GPU (1-16), acceptance criteria for metrics
- **Run with**: Part of release automation (or manually via job runner tools)

### When Adding a Prod Recipe

Add **both** test types to your recipe:

1. **CI test** in `ci_suite.py`: Minimal smoke test (just verify it runs)
2. **Stable test** in `stable_suite.py`: Full training run with performance criteria

### Quick Commands

```bash
metta ci --stage recipe-tests

# Run stable suite validation (performance tests on remote infrastructure)
python devops/stable/stable.py validate

# Run stable validation with job filtering
python devops/stable/stable.py validate --job "arena*"
```
