# Disk Cleanup Action

Free up ~16GB on GitHub Actions runners by removing Android SDK and swap.

## Usage

```yaml
- name: Free disk space
  uses: ./.github/actions/disk-cleanup
```

That's it! No configuration needed.

## What it does

- Removes Android SDK (~12GB)
- Removes swap file (~4GB)
- Cleans APT cache
- **Takes ~5 seconds**

## Example workflow

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # Free up space before heavy operations
      - name: Free disk space
        uses: ./.github/actions/disk-cleanup

      # Now you have ~16GB more space
      - name: Build large Docker image
        run: docker build -t myapp .
```

## Outputs

- `space-before` - Free space before cleanup (GB)
- `space-after` - Free space after cleanup (GB)
- `space-saved` - Space saved (GB)

## Notes

- Android SDK is pre-installed on all GitHub Ubuntu runners
- Most workflows don't need Android SDK or swap
- This is safe for 99% of workflows
- If you need Android SDK, don't use this action
