#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export NIMBLE_DIR="${NIMBLE_DIR:-$SCRIPT_DIR/.nimble}"

mkdir -p "$NIMBLE_DIR" bindings/generated
nimble install -y
nimble c -y -d:release --app:lib --gc:arc -d:fidgetUseCached=true --tlsEmulation:off --out:libmettascope2.dylib --outdir:bindings/generated bindings/bindings.nim
