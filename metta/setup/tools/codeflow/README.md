# codeflow

**codeflow** is a tool for copying a codebase (or parts of it) for use with LLM prompts.

## How it works

codeflow is a spiritual descendant of [files-to-prompt](https://github.com/simonw/files-to-prompt),
a tool for specifying and formatting a subset of a codebase to submit as context to an LLM.

codeflow builds on files-to-prompt with additional features:

- Automatic detection and prioritization of parent READMEs
- Token profiling to understand context window usage
- Flame graph visualization of token distribution

## Usage

```bash
# Copy current directory to clipboard (macOS):
metta code -p

# Copy specific paths (space-separated, relative to current directory):
metta code src tests

# Filter to specific file types:
metta code -e .py -e .js src

# Generate raw output instead of XML-wrapped:
metta code -r src tests

# Profile token usage:
metta code --profile src

# Generate interactive flame graph of token distribution:
metta code --flamegraph src
```
