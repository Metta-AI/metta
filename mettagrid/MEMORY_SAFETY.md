# Memory Safety Improvements

This document outlines the memory safety improvements implemented in the mettagrid codebase to prevent memory leaks, buffer overflows, and segmentation faults.

## Issues Fixed

### 1. Unsafe Static Casts (HIGH PRIORITY)
**Files**: `action_handler.hpp`, `actions/attack.hpp`, `actions/attack_nearest.hpp`

**Problem**: Code used `static_cast<Agent*>()` without type validation, which could cause undefined behavior if the object wasn't actually an Agent.

**Solution**: Replaced with `dynamic_cast<Agent*>()` and added null checks:
```cpp
// Before (unsafe)
Agent* actor = static_cast<Agent*>(_grid->object(actor_object_id));

// After (safe)
GridObject* obj = _grid->object(actor_object_id);
Agent* actor = dynamic_cast<Agent*>(obj);
if (!actor) {
    return false; // Invalid object type
}
```

### 2. Buffer Access Bounds Checking (HIGH PRIORITY)
**Files**: `mettagrid_c.cpp`

**Problem**: No bounds checking on agent indices before accessing observation buffers.

**Solution**: Added validation:
```cpp
if (agent_idx >= _agents.size()) {
    // Invalid agent index - this should not happen in normal operation
    return;
}
```

Also added validation for token buffer overflow and actions array size mismatch.

### 3. Raw Pointer Ownership Issues (MEDIUM PRIORITY)
**Files**: `mettagrid_c.cpp`

**Problem**: Raw pointers created with `new` but if `add_object()` fails, memory leaks occur.

**Solution**: Use RAII with `std::make_unique`:
```cpp
// Before (leak prone)
Wall* wall = new Wall(r, c, wall_cfg);
_grid->add_object(wall);

// After (safe)
auto wall = std::make_unique<Wall>(r, c, wall_cfg);
if (!_grid->add_object(wall.release())) {
    throw std::runtime_error("Failed to add wall to grid");
}
```

### 4. Grid Bounds Validation (MEDIUM PRIORITY)
**Files**: `grid.hpp`

**Problem**: `remove_object()` and `object()` functions lacked bounds checking.

**Solution**: Added comprehensive validation:
```cpp
inline GridObject* object(GridObjectId obj_id) {
    if (obj_id >= objects.size()) {
        return nullptr;
    }
    return objects[obj_id].get();
}
```

### 5. Python Exception Handling (MEDIUM PRIORITY)
**Files**: `mettagrid_c.cpp`

**Problem**: Python object access could throw exceptions that weren't handled.

**Solution**: Added try-catch blocks:
```cpp
try {
    num_agents = cfg["num_agents"].cast<int>();
    // ... other config access
} catch (const py::cast_error& e) {
    throw std::runtime_error("Invalid config parameter: " + std::string(e.what()));
}
```

## Compiler Improvements

### Enhanced Warning Flags
**Files**: `CMakeLists.txt`

Added comprehensive compiler warnings to catch potential issues:
- `-Wall -Wextra -Wpedantic`
- `-Wconversion -Wsign-conversion`
- `-Wunused -Wuninitialized`
- `-Wmissing-declarations`
- `-Wformat=2 -Wcast-qual -Wcast-align`
- `-Wshadow -Wpointer-arith`

### AddressSanitizer Integration
Automatic AddressSanitizer integration in Debug builds:
```cmake
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mettagrid_obj PUBLIC "-fsanitize=address" "-fno-omit-frame-pointer")
    target_link_options(mettagrid_obj PUBLIC "-fsanitize=address")
  endif()
endif()
```

## Enhanced Testing

### Improved Memory Leak Tests
**Files**: `tests/test_leaks.py`

- Increased test iterations from 20 to 50
- Added stress testing with rapid creation/destruction
- Added peak memory usage tracking
- Early failure detection for excessive memory growth

### C++ Memory Safety Tests
**Files**: `tests/test_cpp_memory_safety.cpp`

New comprehensive C++ tests covering:
- Grid bounds checking
- Object ownership transfer
- Type safety with dynamic_cast
- Safe object destruction

### Enhanced Test Scripts
**Files**: `scripts/test_leaks.sh`

Multi-method leak detection:
- Python memory usage monitoring
- AddressSanitizer integration
- Valgrind support (Linux)
- Automated tool availability detection

## Usage

### Running Memory Safety Tests

```bash
# Run Python memory leak tests
python -m pytest tests/test_leaks.py -v

# Run comprehensive leak detection suite
./scripts/test_leaks.sh

# Build and run C++ memory safety tests
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
make
ctest --label-regex test
```

### Debug Builds with AddressSanitizer

```bash
mkdir build_debug && cd build_debug
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
make
# AddressSanitizer is automatically enabled
```

### Valgrind Analysis (Linux)

```bash
valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all python your_script.py
```

## Best Practices

1. **Always use RAII**: Prefer `std::unique_ptr` and `std::make_unique` over raw `new`
2. **Validate inputs**: Check bounds and null pointers before use
3. **Use dynamic_cast**: For safe downcasting in polymorphic hierarchies
4. **Exception safety**: Handle Python exceptions in pybind11 code
5. **Test thoroughly**: Run memory leak tests regularly
6. **Enable warnings**: Use compiler warning flags in development

## Monitoring

Regular memory safety monitoring should include:
- Running `./scripts/test_leaks.sh` in CI/CD
- Building with Debug mode for AddressSanitizer
- Periodic Valgrind analysis on Linux systems
- Monitoring peak memory usage in production

## Future Improvements

Potential future enhancements:
- Integration with clang-static-analyzer
- Custom memory allocators for better tracking
- Memory usage benchmarks
- Automated leak detection in CI/CD
- Thread safety analysis for concurrent environments