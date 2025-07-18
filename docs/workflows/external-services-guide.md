# External Services in GitHub Actions

This guide explains how to handle external service credentials (Wandb, Codecov, Asana, etc.) in a way that allows workflows to run successfully even when credentials aren't available (e.g., on forks or external PRs).

## Quick Start

Add this single step after checkout to automatically handle external PRs:

```yaml
- name: Detect PR context
  uses: ./.github/actions/detect-pr-context
  with:
    has_secrets: ${{ secrets.CODECOV_TOKEN != '' }}
```

This will:
- ✅ Detect if running on a fork, Dependabot PR, or external PR
- ✅ Display a clear banner showing the context
- ✅ Set `EXTERNAL_PR_MODE` environment variable
- ✅ Configure service placeholders automatically

## General Principles

1. **Never fail due to missing optional credentials** - External services should enhance CI, not block it
2. **Provide clear feedback** - Display banners showing what mode CI is running in
3. **Be consistent** - Use the same `EXTERNAL_PR_MODE` pattern everywhere

## Usage Patterns

### 1. Skip External Services on External PRs

```yaml
- name: Upload to Codecov
  if: env.EXTERNAL_PR_MODE != 'true'
  env:
    CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }}
  run: ./upload_codecov.py
```

### 2. Use Config Flags Based on Context

```yaml
- name: Run tests
  run: |
    if [ "$EXTERNAL_PR_MODE" = "true" ]; then
      uv run tools/train.py wandb=off
    else
      uv run tools/train.py
    fi
```

### 3. Automatic Environment Variables

When `EXTERNAL_PR_MODE=true`, these are automatically set:
- `WANDB_MODE=offline` - Wandb runs in offline mode
- `CODECOV_TOKEN=skip` - Codecov skips upload
- `ASANA_API_KEY=skip` - Asana integration skipped

## What You'll See

On external PRs, you'll see banners like:
- `🔓 Running on fork PR - external services disabled`
- `🔓 Running on Dependabot PR - external services disabled`
- `🔓 Running on external PR (no secrets) - external services disabled`

On internal PRs with secrets:
- `🔒 Running on internal PR - all services enabled`

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
