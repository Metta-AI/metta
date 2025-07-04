# Testing Philosophy

Follow generic best practices. With help from our friend Claude, here are some generic points:
* Tests should be independent and idempotent. Tests shouldn't pass or fail based on other tests.
* Tests should be focused. Test one thing per test.
* Tests should be fast and efficient.
* Tests should cover edge cases and boundary conditions, not just the happy path.
* External dependencies should be appropriately mocked, so your test can be focused on what it's testing.
* Tests should be readable by humans.

In addition to these generic points, here are some other thoughts:
* If you need to fix a bug or a regression, you should first try to reproduce the bug as a test, and then fix it. Second best is to fix it and then write a test that would have caught it.
* Unit tests are better because they're faster. Integration tests are better because they're more directly connected to usage (e.g., actually training).
* LLMs should be writing most of your tests, but you should review them to make sure they're covering what they should be.

# Test Structure

## Integration Tests (Repository Root `tests/`)
* **Location**: Integration tests live in the `tests/` directory at the repository root
* **Purpose**: Test interactions between multiple components, end-to-end workflows, and system-wide functionality
* **Structure**:
  - Tests should start with `test_`. Other files should not start with `test_`.
  - The test directory should mirror the directory structure of the project, including subdirectories
  - Example: tests for `package_name/your/favorite/file.py` should be in `tests/your/favorite/test_file.py`

## Unit Tests (Subpackage `tests/`)
* **Location**: Each subpackage should have its own `tests/` directory
* **Purpose**: Test individual components, functions, and classes in isolation
* **Structure**: Follow the same naming conventions and mirroring structure as integration tests, but scoped to the subpackage

## Example Structure
```
project-root/
├── tests/                    # Integration tests
│   ├── __init__.py
│   ├── test_end_to_end.py
│   └── submodule/
│       └── test_integration.py
├── package_name/
│   ├── __init__.py
│   ├── module.py
│   └── tests/               # Unit tests for package_name
│       ├── __init__.py
│       └── test_module.py
└── another_package/
    ├── __init__.py
    ├── component.py
    └── tests/               # Unit tests for another_package
        ├── __init__.py
        └── test_component.py
```

If tests are not structured this way, we should migrate them.
