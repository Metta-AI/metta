# Check Build Quality Action

This action checks build output for errors, warnings, and runtime issues.

## Inputs

- `errors` (required): Number of build errors
- `warnings` (required): Number of build warnings
- `runtime_issues` (optional): Number of runtime issues (default: '0')
- `max_warnings` (optional): Maximum allowed warnings (default: '50')
- `build_type` (optional): Type of build for messages (default: 'Build')
- `full_output` (optional): Full build output for debugging
- `print_full_output` (optional): Whether to print full output on failure (default: 'true')

## Usage Example

```yaml
- name: Check C++ build quality
  uses: ./.github/actions/check-cpp-build
  with:
    errors: ${{ steps.build.outputs.total_errors }}
    warnings: ${{ steps.build.outputs.total_warnings }}
    runtime_issues: ${{ steps.build.outputs.runtime_issues }}
    max_warnings: '100'
    build_type: 'MettaGrid C++'
    full_output: ${{ steps.build.outputs.full_output }}
```
