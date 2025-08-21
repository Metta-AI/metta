# Disk Cleanup Action

Free up ~8GB on GitHub Actions runners by removing Android SDK in the background.

## Usage

```yaml
- name: Free disk space
  uses: ./.github/actions/disk-cleanup
```

That's it! No configuration needed.

## What it does

- Removes Android SDK in background
- **Returns immediately** (<2 seconds)
- Space is freed while your job continues running

## Example workflow

```yaml
jobs:
  build:
    runs-on: default
    steps:
      - uses: actions/checkout@v4

      # Start cleanup - returns immediately
      - name: Free disk space
        uses: ./.github/actions/disk-cleanup

      # Continue with your workflow while cleanup happens in background
      - name: Setup environment
        run: |
          # Disk space is being freed up in parallel
          npm install

      - name: Build large Docker image
        run: |
          # By now, you have ~8GB more space available
          docker build -t myapp .
```

## How it works

1. Checks what can be cleaned (Android SDK, swap)
2. Starts deletion in background
3. Returns immediately so your workflow continues
4. Space becomes available as deletions complete

## Output

```
Starting cleanup of Android SDK...
✓ Cleaning in background: Android SDK
  ~8GB will be freed up as your job continues
```

## Notes

- Android SDK is pre-installed on all GitHub Ubuntu runners but rarely needed
- The cleanup happens in parallel with your next steps
- No options or configuration - designed to just work
- Safe for 99% of workflows (if you need Android SDK, don't use this action)
