# URI Handling Comparison: Main vs Richard-Policy-Cull

## Overview

This document compares the URI handling implementations between the `main` branch (using PolicyStore/PolicyRecord) and the `richard-policy-cull` branch (using CheckpointManager) to ensure feature parity and identify areas for alignment.

## URI Format Support

Both branches support the same URI formats:

### 1. File URIs: `file://`
- **Format**: `file:///path/to/checkpoint.pt` or `file:///path/to/checkpoints/`
- **Usage**: Local filesystem access to checkpoint files or directories
- **Both branches**: ✅ Full support

### 2. Wandb URIs: `wandb://`
- **Short Format**: `wandb://run/<run_name>[:<version>]` 
- **Short Format**: `wandb://sweep/<sweep_name>[:<version>]`
- **Full Format**: `wandb://<entity>/<project>/<artifact_type>/<name>[:<version>]`
- **Usage**: Access to Weights & Biases artifacts
- **Both branches**: ✅ Full support

### 3. S3 URIs: `s3://`
- **Format**: `s3://bucket/path/to/file.pt`
- **Usage**: AWS S3 object storage access
- **Both branches**: ✅ Full support

### 4. PyTorch URIs: `pytorch://` 
- **Format**: `pytorch://path/to/model`
- **Usage**: Loading PyTorch Hub models or specific formats
- **Main branch**: ✅ Full support (PolicyStore)
- **Richard-policy-cull**: ❌ Not implemented

## Core Components Comparison

### Main Branch (PolicyStore/PolicyRecord)

**Key Classes:**
- `PolicyStore`: Central management for policy loading, caching, and storage
- `PolicyRecord`: Lightweight wrapper containing policy metadata and lazy-loading
- `PolicyMetadata`: Structured metadata storage

**URI Handling Flow:**
```
policy_uri → PolicyStore.policy_records() → PolicyRecord[] → PolicyRecord.policy
```

**Key Methods:**
- `PolicyStore.policy_records(uri, selector_type, n, metric)`: Returns list of PolicyRecord objects
- `PolicyStore.load_from_uri(uri)`: Loads single PolicyRecord from URI
- `PolicyRecord.policy`: Lazy-loads actual policy agent
- `PolicyStore._prs_from_wandb()`: Wandb artifact collection handling
- `PolicyStore._prs_from_path()`: Local file system handling
- `PolicyStore._prs_from_pytorch()`: PyTorch hub integration

**Selection Strategies:**
- `"all"`: Return all found policies
- `"latest"`: Return most recent policy
- `"top"`: Return top N by metric score
- `"rand"`: Return random policy

### Richard-Policy-Cull Branch (CheckpointManager)

**Key Classes:**
- `CheckpointManager`: Simple checkpoint management with filename-embedded metadata
- Direct policy loading without intermediate wrapper objects

**URI Handling Flow:**
```
policy_uri → CheckpointManager.load_from_uri() → PolicyAgent
```

**Key Methods:**
- `CheckpointManager.load_from_uri(uri)`: Direct policy loading
- `CheckpointManager.get_policy_metadata(uri)`: Extract metadata from URI/filename
- `discover_policy_uris(base_uri, strategy, count, metric)`: Policy discovery with selection
- `key_and_version(uri)`: Extract run_name and epoch from URI

**Selection Strategies:**
- `"latest"`: Most recent by epoch
- `"best_score"`: Highest score from filename metadata
- `"all"`: All matching policies

## Detailed Feature Comparison

### 1. URI Parsing and Validation

**Main Branch (PolicyStore):**
```python
def _load_policy_records_from_uri(self, uri: str) -> list[PolicyRecord]:
    if uri.startswith("wandb://"):
        return self._prs_from_wandb(uri)
    elif uri.startswith("file://"):
        return self._prs_from_path(uri[len("file://"):])
    elif uri.startswith("pytorch://"):
        return self._prs_from_pytorch(uri[len("pytorch://"):])
    else:
        return self._prs_from_path(uri)  # Fallback to local path
```

**Richard-Policy-Cull (CheckpointManager):**
```python
@staticmethod
def load_from_uri(uri: str):
    if uri.startswith("file://"):
        path_str = uri[7:]  # Remove "file://" prefix
        # ... handle file loading
    elif uri.startswith("s3://"):
        with local_copy(uri) as local_path:
            return torch.load(local_path, weights_only=False)
    elif uri.startswith("wandb://"):
        return load_policy_from_wandb_uri(uri, device="cpu")
    else:
        raise ValueError(f"Unsupported URI format: {uri}")
```

### 2. Wandb Integration

**Main Branch - Rich Artifact Collection Support:**
- Supports artifact collections with version filtering
- Handles both short and full wandb URI formats
- `wandb://run/<name>` → `<entity>/<project>/model/<name>`
- `wandb://sweep/<name>` → `<entity>/<project>/sweep_model/<name>`
- Metadata from wandb artifact metadata

**Richard-Policy-Cull - Direct Artifact Loading:**
- Uses existing `load_policy_from_wandb_uri()` function
- Supports standard wandb URI formats
- Metadata extracted via `get_wandb_checkpoint_metadata()`

### 3. Metadata Handling

**Main Branch (PolicyMetadata):**
```python
class PolicyMetadata:
    agent_step: int = 0
    epoch: int = 0
    generation: int = 0
    train_time: float = 0.0
    action_names: list[str] = []
    # ... extensible structure
```

**Richard-Policy-Cull (Embedded Metadata):**
```python
# Filename format: {run_name}.e{epoch}.s{agent_step}.t{total_time}.sc{score}.pt
def parse_checkpoint_filename(filename: str) -> tuple[str, int, int, int, float]:
    # Extracts run_name, epoch, agent_step, total_time, score
```

### 4. Policy Selection and Discovery

**Main Branch:**
- Selection happens at PolicyStore level
- Metrics can be from metadata, eval_scores, or remote stats server
- Rich filtering and scoring capabilities
- Threshold checking (80% valid scores requirement)

**Richard-Policy-Cull:**
- Selection happens via `discover_policy_uris()` 
- Metrics primarily from filename-embedded data
- Simpler selection logic focused on epoch and score

### 5. Caching

**Main Branch:**
- `PolicyCache` with LRU eviction
- Caches at PolicyRecord level
- Configurable cache size per PolicyStore

**Richard-Policy-Cull:**
- Simple OrderedDict cache in CheckpointManager
- Caches actual checkpoint files
- Fixed cache size per manager instance

## Key Differences

### 1. Architecture Philosophy

**Main Branch:** 
- Object-oriented with clear separation of concerns
- PolicyRecord acts as a lazy-loading proxy
- Rich metadata and query capabilities
- Designed for complex policy management scenarios

**Richard-Policy-Cull:**
- Functional approach with static methods
- Direct policy loading without wrappers
- Filename-embedded metadata for simplicity
- Designed for straightforward checkpoint management

### 2. Metadata Storage

**Main Branch:**
- Metadata stored in PolicyRecord objects
- Extensible PolicyMetadata structure
- Can incorporate external metadata sources

**Richard-Policy-Cull:**
- Metadata embedded in filenames
- Fixed schema: `{name}.e{epoch}.s{step}.t{time}.sc{score}.pt`
- Simple parsing with regex validation

### 3. Policy Selection

**Main Branch:**
- Complex selection with metric evaluation
- Stats server integration for external scores
- Fallback mechanisms for missing metrics

**Richard-Policy-Cull:**
- Filename-based selection
- Limited to epoch and score metrics
- Simple strategy mapping

### 4. Error Handling

**Main Branch:**
- Graceful degradation with warnings
- Multiple fallback mechanisms
- Rich error context

**Richard-Policy-Cull:**
- Direct exceptions for invalid formats
- Simpler error paths
- Clear failure modes

## Missing Features Analysis

### PyTorch URI Support in Richard-Policy-Cull

The richard-policy-cull branch lacks `pytorch://` URI support that exists in main:

**Main Branch Implementation:**
```python
def _prs_from_pytorch(self, path: str) -> list[PolicyRecord]:
    # Creates PolicyRecord with pytorch policy loading
    pr._cached_policy = load_pytorch_policy(path, device, pytorch_cfg)
```

**Missing in Richard-Policy-Cull:** No equivalent functionality

### Advanced Wandb Features in Richard-Policy-Cull

**Main Branch Features:**
- Artifact collection browsing
- Version filtering across collections
- Short URI format expansion with entity/project context

**Richard-Policy-Cull Status:**
- Basic wandb loading works
- May not support all URI variations from main branch

### Rich Metadata Querying in Richard-Policy-Cull

**Main Branch:**
- External metadata injection
- Query-based policy selection
- Integration with stats servers

**Richard-Policy-Cull:**
- Limited to filename metadata
- No external metadata integration

## Tool Integration Patterns

### Main Branch Tools

**SimTool:**
```python
policy_store = PolicyStore.create(device, wandb_config, data_dir)
policy_records = policy_store.policy_records(
    uri_or_config=policy_uri,
    selector_type=self.selector_type,
    n=1,
    metric=metric
)
```

**PlayTool/ReplayTool:**
```python
policy_store = PolicyStore.create(...)
simulation = Simulation.create(policy_store=policy_store, ...)
```

### Richard-Policy-Cull Tools

**SimTool:**
```python
discovered_uris = discover_policy_uris(
    policy_uri, strategy=strategy, count=count, metric=metric
)
agent = CheckpointManager.load_from_uri(policy_uri_path)
metadata = CheckpointManager.get_policy_metadata(policy_uri_path)
```

**PlayTool/ReplayTool:**
```python
simulation = Simulation.create(policy_uri=policy_uri, ...)
# CheckpointManager.load_from_uri() called internally
```

## Current Status

### Main Branch Advantages:
- ✅ Complete PyTorch URI support
- ✅ Rich metadata and querying capabilities
- ✅ Complex policy selection strategies
- ✅ External metadata integration
- ✅ Robust caching and error handling

### Richard-Policy-Cull Advantages:
- ✅ Simpler, more direct API
- ✅ Embedded metadata in filenames
- ✅ Static methods for easy testing
- ✅ Clear separation from training pipeline
- ✅ Filename-based policy identification

### Gaps to Address:

1. **PyTorch URI Support**: Richard-policy-cull needs `pytorch://` implementation
2. **Wandb URI Variants**: Ensure all wandb URI formats from main work in richard-policy-cull
3. **Metadata Extensibility**: Consider if filename-only metadata is sufficient
4. **Selection Strategy Parity**: Map remaining selection strategies between branches
5. **Error Handling**: Align error handling patterns and messages

## Focused Recommendations for Wandb URI Compatibility and Testing

Based on the comprehensive analysis above and the requirement to focus on wandb URI compatibility and comprehensive testing (excluding pytorch URI changes), here are the specific recommendations:

### 1. Enhance Wandb URI Support Compatibility ⭐ HIGH PRIORITY

**Priority: High**  
**Effort: Medium**

Ensure all wandb URI formats from main branch work in richard-policy-cull:

**Test these formats:**
- `wandb://run/<run_name>`
- `wandb://run/<run_name>:<version>`
- `wandb://sweep/<sweep_name>`
- `wandb://sweep/<sweep_name>:<version>`
- `wandb://<entity>/<project>/<artifact_type>/<name>:<version>`

**Key Issue Identified:** The main branch PolicyStore expands short wandb URI formats like `wandb://run/<name>` to full format using configured entity/project context. The richard-policy-cull branch passes URIs directly to `load_policy_from_wandb_uri()`, which may not handle short formats.

**Implementation Approach:**
- Verify if `load_policy_from_wandb_uri()` in `metta/rl/wandb.py` can handle short URI formats
- If not, add URI expansion logic similar to main branch PolicyStore
- Ensure `get_wandb_checkpoint_metadata()` works with all URI variants

**Files to examine/modify:**
- `metta/rl/wandb.py`: Check `load_policy_from_wandb_uri()` implementation
- `metta/rl/checkpoint_manager.py`: May need URI expansion before calling wandb functions

### 2. Create Comprehensive URI Tests ⭐ HIGH PRIORITY  

**Priority: High**  
**Effort: Medium**

**COMPLETED ✅** - Created comprehensive test suite in `/private/tmp/metta/tests/rl/test_uri_handling_comprehensive.py` covering:

- **File URIs**: Single files, directories, run directories with checkpoint navigation
- **S3 URIs**: Basic loading and metadata extraction
- **Wandb URIs**: Full format, short run format, short sweep format, metadata extraction, error conditions
- **URI Normalization**: Path-to-URI conversion consistency
- **Edge Cases**: Malformed URIs, empty directories, network errors
- **Integration Tests**: Cross-format compatibility, metadata consistency, fallback behavior

**Test Coverage:**
- 25+ test methods across 4 test classes
- Comprehensive mocking of external dependencies (wandb, s3)
- Focus on wandb URI compatibility as requested
- Error handling and edge case coverage

### 3. Improve Policy Selection Strategy Mapping

**Priority: Medium**  
**Effort: Low**

Update `discover_policy_uris()` to support all selection strategies from main branch:

```python
def discover_policy_uris(base_uri: str, strategy: str = "latest", count: int = 1, metric: str = "epoch") -> List[str]:
    """Discover policy URIs from a base URI using CheckpointManager."""
    # Add support for main branch strategies:
    # "all", "latest", "top", "rand" -> "all", "latest", "best_score", "random"
    
    strategy_map = {
        "top": "best_score",
        "latest": "latest", 
        "best_score": "best_score",
        "all": "all",
        "rand": "random"  # Add this
    }
    
    mapped_strategy = strategy_map.get(strategy, "latest")
    # ... rest of implementation
```

**Files to modify:**
- `metta/rl/policy_management.py`: Add "rand" strategy support
- `metta/rl/checkpoint_manager.py`: Add random selection to `select_checkpoints()`

### 4. Standardize Error Handling and Messages

**Priority: Medium**  
**Effort: Low**

Align error handling patterns between branches:

```python
# CheckpointManager should use similar error handling patterns as PolicyStore
@staticmethod
def load_from_uri(uri: str):
    try:
        # ... loading logic
    except Exception as e:
        logger.warning(f"Failed to load policy from {uri}: {e}")
        return None  # Instead of raising immediately
```

**Approach:**
- Make CheckpointManager more tolerant of failures
- Add warning logs before returning None
- Provide fallback mechanisms similar to PolicyStore

### 5. URI Normalization Consistency

**Priority: Low**  
**Effort: Very Low**

Ensure both branches handle path-to-URI conversion consistently:

```python
# Both branches should have equivalent normalization
CheckpointManager.normalize_uri("/path/to/file") == PolicyStore.normalize_uri("/path/to/file")
# Should both return: "file:///absolute/path/to/file"
```

### 6. Metadata Schema Alignment (Optional)

**Priority: Low (Optional)**  
**Effort: High**

Consider extending CheckpointManager metadata to match PolicyMetadata richness:

**Recommendation:** Keep filename-only approach. The filename-embedded metadata approach in richard-policy-cull is simpler and adequate for current use cases.

### Implementation Priority (Focused Scope)

**Phase 1 (Required for Wandb URI Compatibility):**
1. Enhance Wandb URI support (#1) ⭐
2. Comprehensive URI tests (#2) ✅ COMPLETED

**Phase 2 (Enhanced Compatibility):**
3. Improve selection strategies (#3)
4. Standardize error handling (#4)
5. URI normalization consistency (#5)

**Phase 3 (Optional):**
6. Metadata schema alignment (#6)

### Migration Path

Since the richard-policy-cull branch is intended to replace the main branch's policy handling:

1. **Implement Phase 1 recommendations** to ensure basic feature parity
2. **Run extensive testing** with existing policy URIs used in production
3. **Update documentation** to reflect any API changes
4. **Provide migration guide** for users transitioning from PolicyStore to CheckpointManager

This approach ensures that the richard-policy-cull branch can be a drop-in replacement for policy handling while maintaining the benefits of its simpler, more direct architecture.