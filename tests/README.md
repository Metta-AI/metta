# Testing Philosophy

Follow generic best practices. With help from our friend Claude, here are some generic points

- Tests should be independent and idempotent. Tests shouldn't pass or fail based on other tests.
- Tests should be focused. Test one thing per test.
- Tests should be fast and efficient.
- Tests should cover edge cases and boundary conditions, not just the happy path.
- External dependencies should be appropriately mocked, so your test can be focused on what it's testing.
- Tests should be readable by humans.

In addition to these generic points, here are some other thoughts:

- If you need to fix a bug or a regression, you should first try to reproduce the bug as a test, and then fix it. Second
  best is to fix it and then write a test that would have caught it.
- Unit tests are better because they're faster. Integration tests are better because they're more directly connected to
  usage (e.g., actually training).
- LLMs should be writing most of your tests, but you should review them to make sure they're covering what they should
  be.

# Testing Structure

- Tests should be in `tests/`, with there being a single such directory at the root of the project.
- Tests should start with `test_`. Other files should not start with `test_`.
- The test directory should mirror the directory structure of the project, including subdirectories. So, for example,
  tests for `package_name/your/favorite/file.py` should be in `tests/your/favorite/test_file.py`.

If tests are not structured this way, we should migrate them.
