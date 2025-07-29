#!/usr/bin/env bash
set -euo pipefail

# Maintain the md5sums of the src/*.tsx files in a .md5sums file. If the md5sums
# have changed, rebuild the widget. The reason for this script: `metta install`
# needs a script that can return whether or not the widget needs to be rebuilt,
# and `npm run build` needs a script that can do that and also build the widget.
# Exit codes:
#   0 = no rebuild needed
#   1 = build needed and performed
#   2 = script error
#   3 = build failed

# Configuration
readonly MD5SUMS_FILE=".md5sums"
readonly COMPILED_JS="heatmap_widget/static/index.js"
readonly SOURCE_PATTERN="src/*.tsx"

# Get command-line arguments
readonly SHOULD_BUILD="${1:-0}"

# Validate arguments
if [[ "$SHOULD_BUILD" != "0" && "$SHOULD_BUILD" != "1" ]]; then
    echo "Error: First argument must be 0 (check if build is needed only) or 1 (check and build if needed)" >&2
    exit 2
fi

# Detect platform and set appropriate md5 command
if command -v md5sum >/dev/null 2>&1; then
    readonly MD5_CMD="md5sum"
elif command -v md5 >/dev/null 2>&1; then
    readonly MD5_CMD="md5 -r"
else
    echo "Error: Neither md5sum nor md5 command found" >&2
    exit 2
fi

readonly CURRENT_MD5SUMS=$($MD5_CMD $SOURCE_PATTERN 2>/dev/null | sort)

# Build the project and save the md5sums
build_and_save() {
    local msg="$1"
    if [[ "$SHOULD_BUILD" == "1" ]]; then
        echo "$msg"

        # Build and save md5sums on success
        if npx vite build; then
            echo "$CURRENT_MD5SUMS" > "$MD5SUMS_FILE"
            echo "Build completed and md5sums saved"
        else
            echo "Error: Build failed" >&2
            exit 3
        fi
    else
        echo "$msg (build skipped - check mode)"
    fi
}

# Main logic
main() {
    # If no compiled JS file exists, we need to build
    if [[ ! -f "$COMPILED_JS" ]]; then
        build_and_save "No compiled JS file found. Build needed."
        exit 0
    fi

    # If no md5sums file exists, we need to build
    if [[ ! -f "$MD5SUMS_FILE" ]]; then
        build_and_save "No .md5sums file found. Build needed."
        exit 0
    fi

    # Compare current md5sums with saved ones
    local saved_md5sums=$(cat "$MD5SUMS_FILE")
    if [[ "$saved_md5sums" == "$CURRENT_MD5SUMS" ]]; then
        echo "Source files unchanged. No build needed."
        if [[ "$SHOULD_BUILD" == "1" ]]; then
            # So that `npm run build` doesn't have a error exit code on successfully skipping the build.
            exit 0
        else
            # Only 'check if we need to build'-mode should have an error exit code.
            exit 1
        fi
    else
        echo "Source files have changed:"
        diff -d --color=always $MD5SUMS_FILE <(echo "$CURRENT_MD5SUMS") || true
        build_and_save "Source files changed. Build needed."
        exit 0
    fi
}

# Run main function
main "$@"
