# devops/setup-git-hooks.sh
#!/bin/bash
set -e

HOOKS_DIR="$(git rev-parse --show-toplevel)/devops/git-hooks"
GIT_HOOKS_DIR="$(git rev-parse --git-dir)/hooks"

for hook in "$HOOKS_DIR"/*; do
  if [ -f "$hook" ]; then
    ln -sf "$hook" "$GIT_HOOKS_DIR/$(basename "$hook")"
  fi
done

echo "Git hooks installed successfully"
