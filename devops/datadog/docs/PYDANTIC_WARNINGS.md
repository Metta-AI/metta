# Pydantic Warnings in Dashboard Collectors

## Issue

When running the dashboard collectors, you may see warnings like:

```
UnsupportedFieldAttributeWarning: The 'repr' attribute with value False was provided to the `Field()` function, which has no effect in the context it was used.
UnsupportedFieldAttributeWarning: The 'frozen' attribute with value True was provided to the `Field()` function, which has no effect in the context it was used.
```

## Status: HARMLESS - No Action Required

These warnings are **cosmetic only** and do NOT prevent metrics from being collected or pushed to Datadog.

## Root Cause

The warnings come from OmegaConf's internal use of Pydantic Field() with attributes that Pydantic v2 no longer supports in the same way. Specifically:

1. OmegaConf internally uses `Field(repr=False, frozen=True)` for its configuration nodes
2. Pydantic v2 changed how these attributes work - they must now be attached via `Annotated` types
3. This is an **upstream issue in OmegaConf**, not in our code

## Test Results

### Local Test (2025-10-24)
- ✅ GitHub collector: 24 metrics pushed to Datadog
- ✅ Kubernetes collector: 15 metrics pushed to Datadog
- ✅ EC2 collector: 19 metrics pushed to Datadog
- ✅ Skypilot collector: 30 metrics pushed to Datadog

### Cluster Test (2025-10-24)
- ✅ GitHub collector: 28 metrics pushed to Datadog
- ✅ Kubernetes collector: 15 metrics pushed to Datadog
- ✅ EC2 collector: 19 metrics pushed to Datadog
- Job failed on later collectors (wandb, asana) - this is a separate pre-existing issue

## Why We Don't Fix This

1. **Upstream Issue**: The warnings come from OmegaConf's code, not ours
2. **No Impact**: Metrics are successfully collected and pushed despite warnings
3. **Complexity**: Suppressing warnings requires environment variables or code changes that add maintenance burden
4. **Version Dependency**: Will likely be fixed when OmegaConf releases Pydantic v2 compatibility updates

## If You Want to Suppress Warnings

You can suppress these specific warnings by adding to your Python script:

```python
import warnings
from pydantic._internal._generate_schema import UnsupportedFieldAttributeWarning

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
```

Or via environment variable before running:

```bash
export PYTHONWARNINGS="ignore::pydantic._internal._generate_schema.UnsupportedFieldAttributeWarning"
```

However, this is **not recommended** as it may hide other legitimate warnings.

## Related Issues

- Pydantic v2 migration guide: https://docs.pydantic.dev/latest/migration/
- OmegaConf Pydantic v2 support: https://github.com/omry/omegaconf/issues/1059

## Summary

**Leave the warnings alone.** They don't break anything, and suppressing them adds complexity without benefit.
