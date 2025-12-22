#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

nimble bindings
nim r tools/gen_atlas.nim
nim r tools/gen_ui_atlas.nim
