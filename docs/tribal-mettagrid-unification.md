# Tribal-MettaGrid Unification Plan

## Executive Summary

This document outlines a plan to unify the Nim tribal environment with MettaGrid's high-performance PufferLib integration, combining the best of both architectures: tribal's expressive game mechanics with MettaGrid's zero-copy performance.

## Current Architecture Analysis

### MettaGrid Strengths
- **Zero-copy numpy integration** via pybind11
- **Direct PufferLib inheritance** (`MettaGridEnv(MettaGridCore, PufferEnv)`)
- **Vectorized C++ operations** with persistent buffers
- **Mature training infrastructure** with stats/replay writing
- **Configurable game mechanics** and flexible observation/action spaces

### Tribal Strengths  
- **Expressive domain logic** in 1700+ lines of Nim
- **Rich game mechanics**: village cooperation, resource chains, tower defense
- **Efficient compilation** from Nim to C with manual memory management
- **Self-contained implementation** easier to modify and extend
- **Domain-specific optimizations** (sparse heatmap processing, tribal colors)

### Current Integration Issues
- **Conversion overhead**: ctypes requires manual numpy ↔ SeqInt conversion
- **Memory copies**: No zero-copy operations due to ctypes limitations
- **Adapter complexity**: Generic `EnvConfig` pattern adds abstraction layers
- **Dependency conflicts**: PufferLib[metta] tries to install conflicting packages

## Unification Strategy

### Phase 1: Direct Pybind11 Integration 

**Goal**: Replace genny/ctypes bindings with direct pybind11 bindings for zero-copy performance.

#### Implementation Plan

1. **Create pybind11 wrapper** for tribal environment
   ```cpp
   // tribal_pybind.cpp
   #include <pybind11/pybind11.h>
   #include <pybind11/numpy.h>
   #include "tribal_binding.h"  // Generated from Nim
   
   PYBIND11_MODULE(tribal_core, m) {
       py::class_<TribalEnv>(m, "TribalEnv")
           .def(py::init<int>(), "max_steps"_a)
           .def("reset", &TribalEnv::reset, py::return_value_policy::reference_internal)
           .def("step", &TribalEnv::step, py::return_value_policy::reference_internal);
   }
   ```

2. **Modify Nim to export C-compatible interface**
   ```nim
   # tribal_c_api.nim
   proc tribal_create_env(max_steps: cint): ptr Environment {.exportc, cdecl.} =
     return cast[ptr Environment](newEnvironment())
   
   proc tribal_step(env: ptr Environment, actions: ptr array[MAP_AGENTS*2, uint8], 
                   obs: ptr array[OBS_SIZE, uint8], rewards: ptr array[MAP_AGENTS, float32],
                   terminals: ptr array[MAP_AGENTS, bool]) {.exportc, cdecl.} =
     # Direct array manipulation - zero copy
     copyMem(obs, env.observations.addr, OBS_SIZE * sizeof(uint8))
   ```

3. **Create zero-copy numpy views**
   ```python
   class TribalGridEnvDirect:
       def step(self, actions):
           # Direct numpy array views - no copying
           return self._core.step_inplace(actions, self._obs_view, self._rewards_view, 
                                         self._terminals_view, self._truncations_view)
   ```

#### Benefits
- **Performance**: Eliminates conversion overhead between Python and Nim
- **Memory efficiency**: Direct numpy array views, no copies
- **API consistency**: Same interface as MettaGrid for drop-in replacement

### Phase 2: PufferLib Integration

**Goal**: Make tribal environment inherit directly from PufferEnv like MettaGrid.

#### Implementation Plan

1. **Create TribalPufferBase** following MettaGrid pattern
   ```python
   class TribalPufferBase(TribalCore, PufferEnv):
       """Direct PufferLib integration for tribal environment."""
       
       def __init__(self, tribal_config: TribalConfig, **kwargs):
           TribalCore.__init__(self, tribal_config)
           PufferEnv.__init__(self, buf=kwargs.get('buf'))
           
       @property
       def single_observation_space(self):
           return self._observation_space
           
       @property  
       def emulated(self) -> bool:
           return False  # Native environment
   ```

2. **Create TribalEnv** with training features
   ```python
   class TribalEnv(TribalPufferBase):
       """Training-ready tribal environment with stats and replay writing."""
       
       def __init__(self, tribal_config, stats_writer=None, replay_writer=None):
           super().__init__(tribal_config)
           self._stats_writer = stats_writer
           self._replay_writer = replay_writer
   ```

3. **Eliminate EnvConfig adapter layer**
   ```python
   # Before (adapter pattern)
   env_config = TribalEnvConfig(...)
   env = env_config.create_environment()
   
   # After (direct instantiation like MettaGrid)
   env = TribalEnv(TribalConfig(...))
   ```

#### Benefits
- **Direct PufferLib compatibility**: No adapter patterns needed
- **Feature parity**: Stats writing, replay recording, episode tracking
- **Training integration**: Works seamlessly with existing trainer infrastructure

### Phase 3: Build System Unification

**Goal**: Integrate tribal builds into the main metta build system.

#### Implementation Plan

1. **Add Nim compilation to mettagrid build**
   ```cmake
   # mettagrid/CMakeLists.txt
   find_program(NIM_COMPILER nim)
   if(NIM_COMPILER)
       add_custom_command(
           OUTPUT tribal_binding.o
           COMMAND ${NIM_COMPILER} c --app:staticlib --noMain tribal_c_api.nim
           DEPENDS tribal_c_api.nim
       )
       target_sources(mettagrid PRIVATE tribal_binding.o)
   endif()
   ```

2. **Create unified Python package**
   ```python
   # mettagrid/src/metta/mettagrid/__init__.py
   from .mettagrid_env import MettaGridEnv
   try:
       from .tribal_env import TribalEnv
   except ImportError:
       TribalEnv = None  # Optional tribal support
   ```

3. **Simplify installation**
   ```bash
   # Single install command with optional tribal support
   pip install -e ./mettagrid[tribal]
   ```

#### Benefits
- **Unified build system**: One setup.py handles both environments
- **Optional dependencies**: Tribal support is optional, won't break without Nim
- **Consistent installation**: Same process for all environments

## Migration Path

### Step 1: Proof of Concept (2-3 days)
- [ ] Create minimal pybind11 wrapper for tribal environment
- [ ] Implement zero-copy step function
- [ ] Benchmark performance vs current ctypes implementation

### Step 2: Full Pybind11 Integration (1 week)  
- [ ] Complete pybind11 API coverage
- [ ] Add PufferLib inheritance hierarchy
- [ ] Update tribal_basic recipe to use direct instantiation
- [ ] Verify training performance matches or exceeds current implementation

### Step 3: Build System Integration (3-5 days)
- [ ] Add Nim compilation to mettagrid CMake
- [ ] Create unified Python package with optional tribal support
- [ ] Update installation documentation
- [ ] Test on multiple platforms (Linux, macOS)

### Step 4: Feature Parity (1 week)
- [ ] Add stats writing and replay recording to tribal environment
- [ ] Implement episode tracking and timing information
- [ ] Ensure all training features work identically to MettaGrid

## Performance Expectations

### Current Performance (ctypes)
- **Step time**: ~5-10ms (includes numpy conversion overhead)
- **Memory usage**: 2x due to Python↔Nim copies
- **Training throughput**: ~1000 steps/sec

### Expected Performance (pybind11)
- **Step time**: ~2-5ms (zero-copy operations) 
- **Memory usage**: 1x (direct numpy views)
- **Training throughput**: ~2000-5000 steps/sec (2-5x improvement)

## Technical Considerations

### Nim-to-C Interface Design
- Use `{.exportc, cdecl.}` for C-compatible functions
- Design for direct array manipulation to avoid copies
- Handle Nim's garbage collector carefully with long-lived C pointers

### Memory Management
- Nim environment must outlive Python numpy views
- Consider using Nim's `--gc:arc` for deterministic cleanup
- Document lifetime management for safe C API usage

### Cross-Platform Compatibility  
- Test Nim compilation on Linux, macOS, Windows
- Handle platform-specific linking requirements
- Consider static vs dynamic linking for distribution

### Backward Compatibility
- Keep current ctypes interface as fallback during transition
- Maintain `TribalEnvConfig` for compatibility with existing code
- Provide migration guide for users

## Success Metrics

### Performance Metrics
- [ ] Step time reduced by 50%+ 
- [ ] Memory usage reduced by 50%+
- [ ] Training throughput increased by 2x+

### Integration Metrics
- [ ] Tribal environment can be used interchangeably with MettaGrid in training scripts
- [ ] All PufferLib features work identically (vectorization, auto-reset, etc.)
- [ ] Build system works reliably across platforms

### Developer Experience Metrics
- [ ] Installation reduced to single command
- [ ] No dependency conflicts or version issues
- [ ] Documentation and examples updated

## Conclusion

This unification plan leverages the strengths of both architectures: tribal's rich game mechanics and MettaGrid's high-performance infrastructure. The key insight is replacing the ctypes bridge with direct pybind11 integration to achieve zero-copy performance while maintaining the expressive power of Nim for game logic.

The phased approach minimizes risk by validating performance improvements early and maintaining backward compatibility throughout the transition. The result will be a unified, high-performance environment system that makes it easy to develop new game mechanics while maintaining optimal training performance.