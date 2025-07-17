# External Services in GitHub Actions

This guide explains how to handle external service credentials (Wandb, Codecov, Asana, etc.) in a way that allows workflows to run successfully even when credentials aren't available (e.g., on forks or external PRs).

## General Principles

1. **Never fail due to missing optional credentials** - External services should enhance CI, not block it
2. **Provide clear feedback** - Use GitHub notices to indicate when services are skipped
3. **Be consistent** - Use the same patterns across all workflows

## Patterns for Different Services

### 1. Services with Config Flags (Preferred)
For services that support being disabled via configuration:

```yaml
# Example: Wandb
- name: Run tests
  run: uv run tools/train.py wandb=off
```

### 2. Conditional Steps
For services that require credentials to run:

```yaml
# Example: Codecov
- name: Upload coverage
  if: ${{ secrets.CODECOV_TOKEN != '' }}
  env:
    CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }}
  run: ./upload_codecov.py

- name: Skip notification
  if: ${{ secrets.CODECOV_TOKEN == '' }}
  run: echo "::notice::Skipping Codecov - credentials not available"
```

### 3. Placeholder Values
For services that just need a value to be set:

```yaml
# Example: Some APIs
- name: Run with API
  env:
    API_KEY: ${{ secrets.API_KEY || 'placeholder_not_used' }}
  run: ./script_that_checks_api_key.py
```

### 4. Using the Check Credential Action
For a consistent approach across workflows:

```yaml
- name: Check Wandb availability
  id: wandb_check
  uses: ./.github/actions/check-credential
  with:
    secret_value: ${{ secrets.WANDB_API_KEY }}
    service_name: "Wandb"

- name: Run training
  env:
    WANDB_API_KEY: ${{ steps.wandb_check.outputs.value_or_default }}
  run: |
    if [ "${{ steps.wandb_check.outputs.available }}" = "true" ]; then
      uv run tools/train.py
    else
      uv run tools/train.py wandb=off
    fi
```

## Adding a New External Service

When adding a new external service:

1. **Make it optional by default** - The workflow should succeed without credentials
2. **Add conditional logic** - Use one of the patterns above
3. **Document in the workflow** - Add comments explaining the credential requirements
4. **Test on a fork** - Ensure the workflow works without secrets

## Security Notes

- Never log credential values
- Use `${{ secrets.NAME != '' }}` to check availability (doesn't expose the value)
- Prefer service-specific disable flags over dummy credentials when possible
