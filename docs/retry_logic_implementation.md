# Retry Logic Implementation

## Task Details
- **Task**: Add retry logic to mettagrid environment initialization
- **GID**: 1205987654321
- **Priority**: High
- **Tags**: bug, reliability

## Implementation

This implementation adds retry logic with exponential backoff for mettagrid environment initialization to handle race conditions when multiple agents spawn simultaneously.

### Files Created

1. **`metta/utils/retry.py`** - Retry decorator implementation
   - Generic `exponential_backoff_retry` decorator
   - Specialized `env_init_retry` for environment initialization
   - Configurable max attempts, delays, and exception types

2. **`tests/util/test_retry.py`** - Comprehensive unit tests
   - Tests for successful retries
   - Tests for failure scenarios
   - Exponential backoff timing verification
   - Exception handling tests
   - Logging verification

### Usage Example

```python
# Example of applying the retry decorator to environment initialization:

# Before:
def initialize_environment(config):
    """Initialize the mettagrid environment."""
    # ... initialization code that might fail ...
    return env

# After:
@env_init_retry
def initialize_environment(config):
    """Initialize the mettagrid environment with retry logic."""
    # ... initialization code that might fail ...
    return env

# The decorator will automatically retry on:
# - RuntimeError
# - ConnectionError  
# - TimeoutError
# With exponential backoff starting at 0.1s up to 2s, max 3 attempts

```

### Configuration

The retry decorator is configurable:
- `max_attempts`: Maximum retry attempts (default: 3)
- `initial_delay`: Starting delay in seconds (default: 0.1)
- `max_delay`: Maximum delay cap (default: 2.0)
- `backoff_factor`: Delay multiplier (default: 2.0)
- `exceptions`: Tuple of exceptions to retry (default: all)

### Testing

Run the tests:
```bash
uv run pytest tests/util/test_retry.py -v
```

## Integration

To use in mettagrid environment initialization:

1. Import the decorator:
   ```python
   from metta.utils.retry import env_init_retry
   ```

2. Apply to initialization functions:
   ```python
   @env_init_retry
   def initialize_environment(config):
       # initialization code
   ```

The retry logic will automatically handle transient failures during environment setup.
