# Metta AI Policy System Audit Report

## Executive Summary

The current policy saving/loading system in Metta AI is **complex and fragmented**, with multiple layers of abstraction, backward compatibility hacks, and overlapping responsibilities. The system has grown organically and shows clear signs of technical debt that would benefit from a complete redesign.

**Key Issues:**
- Complex agent initialization flow with multiple abstraction layers
- Fragmented policy management across multiple classes
- Heavy backward compatibility burden affecting clean design
- URI-based policy resolution system that adds complexity
- Multiple policy formats and loading mechanisms

## System Architecture Overview

### Entry Point Flow

The system entry point is `./tools/run.py` which uses a recipe-based configuration system:
1. `experiments.recipes.arena_basic_easy_shaped.train` → creates `TrainTool`
2. `TrainTool.invoke()` → calls `train()` function
3. Training system creates `PolicyStore` and manages `MettaAgent` lifecycle

### Core Classes and Responsibilities

#### 1. PolicyStore (`agent/src/metta/agent/policy_store.py`)
**Purpose:** Central policy management hub
**Complexity:** HIGH ⚠️

- **Lines of Code:** 493 lines - Very large for a single responsibility class
- **Responsibilities:**
  - Loading policies from multiple URI schemes (`wandb://`, `file://`, `pytorch://`)
  - Policy selection with multiple strategies (`top`, `latest`, `rand`, `all`)
  - Caching with `PolicyCache` 
  - WandB artifact management
  - Backward compatibility handling
  - Policy metadata extraction and scoring

**Key Methods:**
- `policy_record()` - Get single policy
- `policy_records()` - Get multiple policies with selection strategy
- `save()` - Save policy to disk
- `load_from_uri()` - Load from various URI schemes
- Multiple private loading methods for different sources

#### 2. PolicyRecord (`agent/src/metta/agent/policy_record.py`)
**Purpose:** Container for policy + metadata
**Complexity:** MEDIUM

- **Lines of Code:** 252 lines
- **Responsibilities:**
  - Lazy loading of policies
  - Metadata management with backward compatibility
  - Policy URI handling
  - Detailed policy introspection and `__repr__`

**Issues:**
- Complex backward compatibility in metadata handling
- Circular reference potential with `_policy_store`
- Heavy `__repr__` method (lines 142-251) doing policy analysis

#### 3. MettaAgent (`agent/src/metta/agent/metta_agent.py`) 
**Purpose:** Main agent wrapper
**Complexity:** VERY HIGH ⚠️⚠️⚠️

- **Lines of Code:** 419 lines
- **Responsibilities:**
  - Agent initialization
  - Policy creation and management
  - Environment setup and feature mapping
  - Action space handling
  - Memory management
  - **MASSIVE** backward compatibility in `__setstate__` (lines 304-416)

**Major Issues:**
- `__setstate__` method spans 112 lines handling legacy checkpoint conversion
- Complex feature remapping system
- Mixed responsibilities (agent logic + policy management + environment setup)

#### 4. AgentConfig + Registry (`agent/src/metta/agent/agent_config.py`)
**Purpose:** Agent architecture selection
**Complexity:** MEDIUM

- Dual policy systems: ComponentPolicy vs PyTorch implementations
- Registry pattern with `AGENT_REGISTRY` 
- Factory method `create_agent()`

#### 5. CheckpointManager (`metta/rl/checkpoint_manager.py`)
**Purpose:** Checkpoint and policy saving coordination
**Complexity:** HIGH

- **Lines of Code:** 303 lines
- Policy lifecycle management during training
- Distributed training coordination
- WandB artifact uploading
- Metadata extraction and storage

## Policy Lifecycle Flows

### 1. Training Flow: Policy Creation → Saving
```
TrainTool.invoke() 
→ PolicyStore.create() 
→ CheckpointManager.load_or_create_policy()
  → MettaAgent.__init__() [new policy] 
    → create_agent() from AgentConfig
→ Training Loop
→ CheckpointManager.save_policy() [periodic]
  → PolicyRecord creation
  → PolicyStore.save() 
    → torch.save() to disk
```

### 2. Evaluation Flow: Policy Loading
```
SimTool.invoke() 
→ PolicyStore.policy_records()
  → _select_policy_records() 
    → _load_policy_records_from_uri()
      → _prs_from_path() | _prs_from_wandb() | _prs_from_pytorch()
        → PolicyRecord.policy [lazy loading]
          → PolicyStore.load_from_uri()
            → _load_from_file() → torch.load()
```

### 3. URI Resolution System
The system supports multiple URI schemes:
- `file://path/to/checkpoint` - Local files
- `wandb://entity/project/artifact:version` - WandB artifacts  
- `pytorch://path/to/model` - PyTorch model files
- Plain paths (converted to file://)

**Selection Strategies:**
- `top` - Best performing by metric
- `latest` - Most recent checkpoint
- `rand` - Random selection
- `all` - Return all policies

## Major Pain Points

### 1. Backward Compatibility Burden
- **112-line `__setstate__` method** in MettaAgent handling legacy checkpoints
- Codebase aliasing in PolicyStore (`agent` → `metta.agent`)
- Complex metadata migration logic in PolicyRecord
- Multiple checkpoint formats supported

### 2. Complex Agent Initialization 
- 6 different parameters needed for agent creation
- Dual creation paths (ComponentPolicy vs PyTorch)
- Environment initialization tightly coupled with agent creation
- Feature mapping and remapping logic mixed in

### 3. Fragmented Responsibilities
- PolicyStore does too much (loading, caching, selection, metadata, WandB)
- MettaAgent mixes agent logic with policy management
- CheckpointManager handles both trainer state and policy state
- No clear separation between policy storage and policy logic

### 4. URI Resolution Complexity
- Multiple URI schemes with different semantics
- Complex parsing and validation logic
- WandB artifact collection handling
- Version resolution and artifact downloading

### 5. Memory and Performance Issues
- Large objects being pickled/unpickled frequently
- Complex caching system with LRU eviction
- Heavy policy introspection in `__repr__` methods
- Potential circular reference issues

## Current Policy Formats

### 1. Modern Format (Current)
```python
PolicyRecord(
    policy_store=PolicyStore,
    run_name=str,
    uri=str,
    metadata=PolicyMetadata
)
# Contains:
# - _cached_policy: MettaAgent | DistributedMettaAgent  
# - Serialized via torch.save()
```

### 2. Legacy Checkpoint Format (Backward Compatible)
- Component-based structure with direct component access
- Handled via complex `__setstate__` conversion in MettaAgent
- Automatically converted to ComponentPolicy structure on load

## Recommendations for Redesign

### 1. Simplify Core Architecture
- **Split PolicyStore into focused classes:**
  - `PolicyLoader` - Loading from different sources
  - `PolicyCache` - Memory management (already exists)
  - `PolicySelector` - Selection strategies
  - `PolicyMetadataManager` - Metadata handling

### 2. Clean Agent Initialization
- **Single agent factory with clear interface**
- **Remove environment coupling from agent creation**  
- **Standardize initialization parameters**

### 3. Remove Backward Compatibility Burden
- **Define cutoff point for legacy support**
- **Create migration tool for old checkpoints**
- **Clean up `__setstate__` methods**

### 4. Simplify URI System
- **Reduce supported URI schemes to essential ones**
- **Standardize policy resolution logic**
- **Remove complex WandB artifact handling**

### 5. Policy Format Standardization
- **Single, clean checkpoint format**
- **Separate model weights from metadata** 
- **Version policy format with migration path**

### 6. Reduce Coupling
- **Separate policy serialization from agent logic**
- **Remove policy management from training loop**
- **Standardize policy interface across implementations**

## Complexity Metrics

| Component | Lines of Code | Complexity Level | Priority for Refactor |
|-----------|--------------|------------------|----------------------|
| PolicyStore | 493 | Very High ⚠️⚠️⚠️ | **Critical** |
| MettaAgent | 419 | Very High ⚠️⚠️⚠️ | **Critical** | 
| CheckpointManager | 303 | High ⚠️⚠️ | High |
| PolicyRecord | 252 | Medium ⚠️ | Medium |

**Total LOC in core policy system: ~1,467 lines**

This system would benefit greatly from a ground-up redesign focused on simplicity, clear separation of concerns, and removal of technical debt.