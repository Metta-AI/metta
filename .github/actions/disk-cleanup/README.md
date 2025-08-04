# Disk Cleanup Action

Free up disk space on GitHub Actions runners by removing unnecessary pre-installed software.

## Quick Start

### 1. Diagnostic Mode (default - no files deleted)

```yaml
- name: Check disk usage
  uses: ./.github/actions/disk-cleanup
```

### 2. Clean up disk space

```yaml
- name: Free up disk space
  uses: ./.github/actions/disk-cleanup
  with:
    dry-run: 'false'
```

### 3. Selective cleanup

```yaml
- name: Free up disk space
  uses: ./.github/actions/disk-cleanup
  with:
    dry-run: 'false'
    android: 'false' # Keep Android SDK
    docker-images: 'false' # Keep Docker images
```

## Inputs

| Input            | Description                               | Default | Space Saved |
| ---------------- | ----------------------------------------- | ------- | ----------- |
| `dry-run`        | If true, only shows what would be deleted | `true`  | -           |
| `android`        | Remove Android SDK and tools              | `true`  | ~8-12GB     |
| `dotnet`         | Remove .NET SDK and runtime               | `true`  | ~2-5GB      |
| `haskell`        | Remove Haskell GHC                        | `true`  | ~2-5GB      |
| `large-packages` | Remove Chrome, Firefox, Cloud SDKs, etc.  | `true`  | ~2-4GB      |
| `docker-images`  | Remove all Docker images                  | `true`  | ~5-20GB     |
| `swap-storage`   | Remove swap file                          | `true`  | ~4GB        |
| `tool-cache`     | Remove old tool versions                  | `true`  | ~5-10GB     |

**Total potential savings: 25-50GB** (varies by runner)

## Outputs

| Output         | Description                    |
| -------------- | ------------------------------ |
| `space-before` | Free space before cleanup (GB) |
| `space-after`  | Free space after cleanup (GB)  |
| `space-saved`  | Space saved (GB)               |

## Examples

### Example 1: Diagnostic first, then cleanup

```yaml
steps:
  - uses: actions/checkout@v4

  # First, see what would be deleted
  - name: Check what can be cleaned
    uses: ./.github/actions/disk-cleanup

  # Then actually clean up
  - name: Free up disk space
    id: cleanup
    uses: ./.github/actions/disk-cleanup
    with:
      dry-run: 'false'

  - name: Show results
    run: |
      echo "Freed up ${{ steps.cleanup.outputs.space-saved }}GB of disk space"
      echo "Now have ${{ steps.cleanup.outputs.space-after }}GB free"
```

### Example 2: Use in your test workflow

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # Free up space before Docker builds
      - name: Free up disk space
        uses: ./.github/actions/disk-cleanup
        with:
          dry-run: 'false'
          # Keep tools you need
          docker-images: 'false' # We need Docker

      # Now run your tests with plenty of space
      - name: Run tests
        run: |
          docker build -t myapp .
          docker run myapp test
```

### Example 3: Fail if not enough space

```yaml
- name: Free up disk space
  id: cleanup
  uses: ./.github/actions/disk-cleanup
  with:
    dry-run: 'false'

- name: Check we have enough space
  run: |
    if [ "${{ steps.cleanup.outputs.space-after }}" -lt "20" ]; then
      echo "Error: Only ${{ steps.cleanup.outputs.space-after }}GB free, need at least 20GB"
      exit 1
    fi
```

## What gets removed?

### Android (`android: true`)

- Android SDK at `/usr/local/lib/android`
- Android environment directories

### .NET (`dotnet: true`)

- .NET SDK at `/usr/share/dotnet`
- .NET runtime files

### Haskell (`haskell: true`)

- GHC at `/opt/ghc`
- Cabal and GHCup

### Large Packages (`large-packages: true`)

- Google Chrome
- Firefox
- Microsoft Edge
- Google Cloud SDK
- AWS CLI
- Azure CLI
- Miniconda

### Docker (`docker-images: true`)

- All Docker images
- All stopped containers
- All unused volumes
- All unused networks

### Tool Cache (`tool-cache: true`)

- Old Go versions (keeps latest)
- Old Node.js versions (keeps latest LTS)
- Python 2.x
- PyPy
- Ruby
- CodeQL
- Boost libraries
- PowerShell

### Always cleaned

- APT package cache
- pip cache
- npm cache

## Tips

1. **Always run dry-run first** to see what will be deleted
2. **Keep what you need** - set specific inputs to `false`
3. **Run early** in your workflow, right after checkout
4. **Check the space** - use the outputs to verify you have enough

## Troubleshooting

**Q: The action shows 0GB saved**

- You might be running in dry-run mode (check `dry-run: false`)
- The directories might already be cleaned

**Q: I need more space**

- Enable all cleanup options
- Consider using a larger runner
- Check if your Docker images or build artifacts are too large

**Q: A tool I need was removed**

- Set the corresponding input to `false`
- Or reinstall it after cleanup
