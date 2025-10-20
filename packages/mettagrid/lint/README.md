# Clang-Tidy Integration with Bazel

This directory contains a Bazel-native integration for clang-tidy, providing automatic C++ linting without the need for
external scripts or manual compile_commands.json generation.

## Features

- **Native Bazel Integration**: Works directly with Bazel's build graph using aspects
- **Automatic Configuration**: Uses Bazel's compilation context for accurate include paths and defines
- **No Manual Setup**: No need to generate compile_commands.json manually
- **Incremental**: Only runs on changed files
- **Test Integration**: Can fail builds when issues are found

## Usage

### Generate a Report (doesn't fail)

```bash
bazel build //lint:clang_tidy_report
cat bazel-bin/lint/clang_tidy_report_summary.txt
```

### Run as a Test (fails on issues)

```bash
bazel test //lint:clang_tidy_test
```

### Run via Alias

```bash
bazel test //lint:lint  # Alias for clang_tidy_test
```

## Configuration

The linter uses the `.clang-tidy` configuration file in the project root. Modify this file to customize which checks are
enabled.

## Implementation Details

The integration consists of:

- `clang_tidy.bzl`: Bazel aspect and rules for running clang-tidy
- `clang_tidy_wrapper.sh`: Simple wrapper script for the system clang-tidy
- `BUILD`: Defines the lint targets

The aspect automatically:

1. Extracts compilation flags from C++ targets
2. Runs clang-tidy on each source file
3. Collects reports for analysis
4. Can fail tests if issues are found

## Advantages Over Previous Approach

- **No Manual Scripts**: Replaces `clang-tidy.sh` and `generate_compile_commands.py`
- **Better Integration**: Uses Bazel's actual compilation context
- **More Reliable**: No need to parse Bazel output or maintain separate scripts
- **Incremental**: Only processes changed files
- **Cacheable**: Results are cached by Bazel

## Requirements

- System clang-tidy installed (`/usr/bin/clang-tidy`)
- Bazel with Bzlmod support
- C++20 compatible clang-tidy version
