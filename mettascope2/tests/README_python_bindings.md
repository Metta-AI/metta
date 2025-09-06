# Tribal Python Bindings Tests

This directory contains tests for the Tribal Environment Python bindings generated using [genny](https://github.com/treeform/genny).

## Running Tests

### Prerequisites
1. Build the bindings:
   ```bash
   nimble bindings
   ```

2. Activate the uv environment:
   ```bash
   source /home/relh/Code/metta/.venv/bin/activate
   ```

### Run Tests
```bash
# Run the comprehensive test suite
python tests/test_python_bindings.py

# Run with unittest for more detailed output
python -m unittest tests.test_python_bindings -v
```

## Test Coverage

The test suite (`test_python_bindings.py`) covers:

- **Constants**: Environment dimensions, agent count, observation shape
- **Environment Management**: Creation, reset, stepping, termination
- **Actions**: All 6 action types (NOOP, MOVE, ATTACK, GET, SWAP, PUT)
- **Observations**: 15 agents × 19 layers × 11×11 grid = 34,485 elements
- **Rewards**: Multi-agent reward collection and validation
- **Sequences**: Genny sequence operations (SeqInt, SeqFloat, SeqBool)
- **Error Handling**: Global error system with check_error() and take_error()
- **Memory Management**: Object creation/destruction without leaks
- **Multi-Environment**: Concurrent environment instances
- **RL Integration**: Realistic training scenario simulation

## Expected Results

All 14 tests should pass:
```
🎉 All tests passed! Tribal Python bindings are fully functional.
```

If tests fail, ensure:
1. The bindings were built with `nimble bindings`
2. The shared library (`libtribal.so`) exists in `bindings/generated/`
3. You're using the correct Python environment

## Performance Notes

- **Zero-copy observations**: Direct memory access between Nim and Python
- **Reward generation**: Environment produces ~0.001 rewards for GET actions
- **Step timing**: Each step processes 15 agents across 100×50 map
- **Memory efficient**: Automatic cleanup via genny's reference counting