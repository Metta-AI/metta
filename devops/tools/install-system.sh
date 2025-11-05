#!/bin/bash

err() {
  echo "ERROR: $1" >&2
  exit 1
}

check_cmd() {
  command -v "$1" > /dev/null 2>&1
  return $?
}

detect_package_manager() {
  if check_cmd apt-get; then
    echo "apt"
  elif check_cmd yum; then
    echo "yum"
  elif check_cmd apk; then
    echo "apk"
  elif check_cmd brew; then
    echo "brew"
  elif check_cmd dnf; then
    echo "dnf"
  elif check_cmd pacman; then
    echo "pacman"
  else
    echo "unknown"
  fi
}

install_package() {
  local pkg_manager="$1"
  shift
  local packages="$@"

  case "$pkg_manager" in
    apt)
      apt-get update && apt-get install -y $packages
      ;;
    yum)
      yum install -y $packages
      ;;
    dnf)
      dnf install -y $packages
      ;;
    apk)
      apk add --no-cache $packages
      ;;
    pacman)
      pacman -Sy --noconfirm $packages
      ;;
    brew)
      brew install $packages
      ;;
    *)
      return 1
      ;;
  esac
}

get_package_name() {
  local tool="$1"
  local pkg_manager="$2"

  case "$tool" in
    g++)
      case "$pkg_manager" in
        apt) echo "build-essential" ;;
        yum | dnf) echo "gcc-c++ make" ;;
        apk) echo "build-base" ;;
        pacman) echo "base-devel" ;;
        brew) echo "gcc" ;;
        *) echo "" ;;
      esac
      ;;
    nimble)
      get_package_name "nim" "$pkg_manager"
      ;;
    *)
      echo "$tool"
      ;;
  esac
}

ensure_tool() {
  local tool="$1"

  ensure_paths

  if [ "$tool" = "nim" ] || [ "$tool" = "nimble" ]; then
    if ensure_nim_via_nimby "$REQUIRED_NIM_VERSION"; then
      ensure_paths
      return 0
    fi

    err "Failed to install Nim via nimby"
  fi

  # Special handling for Bazel via Bazelisk
  if [ "$tool" = "bazel" ]; then
    if ensure_bazel_setup; then
      ensure_paths
      return 0
    fi
    err "Failed to install Bazel via bazelisk"
  fi

  if check_cmd "$tool"; then
    return 0
  fi

  echo "$tool not found. Installing $tool..."

  local pkg_manager=$(detect_package_manager)

  if [ "$pkg_manager" = "unknown" ]; then
    err "$tool is required but no package manager found. Please install it manually"
  fi

  local packages=$(get_package_name "$tool" "$pkg_manager")

  if ! install_package "$pkg_manager" "$packages"; then
    err "Failed to install $tool"
  fi

  if ! check_cmd "$tool"; then
    err "Installed $tool but it is not available"
  fi
}

# Required tool versions
REQUIRED_NIM_VERSION="2.2.6"
REQUIRED_NIMBY_VERSION="0.1.6"
REQUIRED_BAZEL_VERSION="7.0.0"

# Common install directories in order of preference
COMMON_INSTALL_DIRS="/usr/local/bin /usr/bin /opt/bin $HOME/.local/bin $HOME/bin $HOME/.nimby/nim/bin $HOME/.nimble/bin $HOME/.cargo/bin /opt/homebrew/bin"

# Add common directories to PATH if not already present
ensure_paths() {
  for dir in $COMMON_INSTALL_DIRS; do
    if [ -d "$dir" ]; then
      case ":$PATH:" in
        *":$dir:"*) ;; # Already in PATH
        *) export PATH="$dir:$PATH" ;;
      esac
    fi
  done

  # Source env files if they exist
  [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
  [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
}

# Return first dir in COMMON_INSTALL_DIRS that is in PATH and writable
get_install_dir() {
  for dir in $COMMON_INSTALL_DIRS; do
    case ":$PATH:" in
      *":$dir:"*)
        if [ -w "$dir" ] || [ "$EUID" -eq 0 ]; then
          echo "$dir"
          return 0
        fi
        ;;
    esac
  done
  return 1
}

# Ensure uv is in PATH, installed, and uv project environment associated with this repo
ensure_uv_setup() {
  if ! check_cmd uv; then
    echo "uv is not installed. Installing uv..."

    local install_dir=$(get_install_dir)
    if [ -n "$install_dir" ]; then
      export UV_INSTALL_DIR="$install_dir"
      echo "Setting UV_INSTALL_DIR=$UV_INSTALL_DIR (detected from PATH)"
    else
      echo "Using default uv installation location"
    fi

    curl -LsSf https://astral.sh/uv/install.sh | sh

    echo "After installation, checking for uv:"
    which uv && echo "uv installed successfully at $(which uv)" || echo "uv not found in PATH"

    # ensure_paths

    if ! check_cmd uv; then
      err "Failed to install uv. Please install it manually from https://github.com/astral-sh/uv"
    fi
  fi
}

ensure_bazel_setup() {
  if ! check_cmd bazel; then
    echo "Bazel is not installed. Installing bazelisk..."

    local install_dir=$(get_install_dir)
    if [ -n "$install_dir" ]; then
      local dest="$install_dir/bazel"
      echo "Installing bazel to $dest (detected from PATH)"
    else
      local dest="$HOME/.local/bin/bazel"
      echo "Installing bazel to default location: $dest"
    fi

    local url=$(get_bazelisk_url)

    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
      dest="${dest}.exe"
    fi

    mkdir -p "$(dirname "$dest")"
    echo "Installing bazelisk to $dest"
    curl -fsSL "$url" -o "$dest"
    chmod +x "$dest"

    #  ensure_paths

    if ! check_cmd bazel; then
      err "Failed to install bazelisk. Please install it manually from https://github.com/bazelbuild/bazelisk"
    fi
  fi

  # Check version
  local current_version
  if ! current_version=$(get_bazel_version); then
    err "Failed to determine bazel version"
  fi

  if ! version_ge "$current_version" "$REQUIRED_BAZEL_VERSION"; then
    echo "ERROR: Bazel version $current_version is too old" >&2
    echo "ERROR: Minimum required version is $REQUIRED_BAZEL_VERSION" >&2
    echo "ERROR: Please upgrade bazel:" >&2
    echo "  - If using bazelisk: it should auto-upgrade (try removing ~/.bazelisk/downloads)" >&2
    echo "  - If using system bazel: uninstall and let install-system.sh install bazelisk" >&2
    echo "  - Manual install: https://github.com/bazelbuild/bazelisk" >&2
    exit 1
  fi
}

version_ge() {
  local current="$1"
  local required="$2"

  # Prefer version-aware sort if available (GNU sort -V)
  if sort -V < /dev/null > /dev/null 2>&1; then
    if [ "$(printf '%s\n%s\n' "$required" "$current" | sort -V | tail -n1)" = "$current" ]; then
      return 0
    else
      return 1
    fi
  fi

  # Fallback: numeric segment compare (handles X.Y.Z)
  local -a current_parts
  local -a required_parts

  IFS='.' read -r -a current_parts <<< "$current"
  IFS='.' read -r -a required_parts <<< "$required"

  local length="${#current_parts[@]}"
  if [ "${#required_parts[@]}" -gt "$length" ]; then
    length="${#required_parts[@]}"
  fi

  for ((i = 0; i < length; i++)); do
    local current_segment="${current_parts[i]:-0}"
    local required_segment="${required_parts[i]:-0}"

    if ((current_segment > required_segment)); then
      return 0
    fi

    if ((current_segment < required_segment)); then
      return 1
    fi
  done

  return 0
}

get_nim_version() {
  if ! check_cmd nim; then
    return 1
  fi

  local version_line
  if ! version_line=$(nim --version 2> /dev/null | head -n1); then
    return 1
  fi

  # Expected format: "Nim Compiler Version X.Y.Z [os: arch]"
  local version
  version=$(echo "$version_line" | awk '{print $4}')

  if [ -z "$version" ]; then
    return 1
  fi

  echo "$version"
  return 0
}

get_bazel_version() {
  if ! check_cmd bazel; then
    return 1
  fi

  local version_line
  if ! version_line=$(bazel --version 2> /dev/null); then
    return 1
  fi

  # Extract second field and strip any non-numeric suffix (e.g., '-homebrew', '+build', etc.)
  local version_raw
  version_raw=$(echo "$version_line" | awk '{print $2}')
  local version
  version=$(echo "$version_raw" | sed -E 's/[^0-9.].*$//')

  if [ -z "$version" ]; then
    return 1
  fi

  echo "$version"
  return 0
}

ensure_nim_via_nimby() {
  local required_version="$1"

  ensure_paths

  local current_version=""
  if check_cmd nim; then
    current_version=$(get_nim_version 2> /dev/null || echo "")
  fi

  if [ -n "$current_version" ] && version_ge "$current_version" "$required_version" && check_cmd nimble; then
    return 0
  fi

  if [ -n "$current_version" ]; then
    echo "Found Nim $current_version but require >= $required_version. Installing via Nimby..."
  else
    echo "Nim not found. Installing via Nimby..."
  fi

  if ! install_nim_via_nimby "$required_version"; then
    return 1
  fi

  ensure_paths
  current_version=$(get_nim_version 2> /dev/null || echo "")

  if [ -n "$current_version" ] && version_ge "$current_version" "$required_version" && check_cmd nimble; then
    return 0
  fi

  echo "Nim install finished but requirements are still not satisfied." >&2
  return 1
}

install_nim_via_nimby() {
  local target_version="$1"

  local detected_system
  detected_system=$(uname -s)
  local detected_arch
  detected_arch=$(uname -m)
  echo "Detected host: $detected_system ($detected_arch)"

  local url
  if ! url=$(get_nimby_url "$REQUIRED_NIMBY_VERSION"); then
    return 1
  fi

  local tmp_dir
  tmp_dir=$(mktemp -d 2> /dev/null || mktemp -d -t nimby)
  if [ -z "$tmp_dir" ] || [ ! -d "$tmp_dir" ]; then
    echo "Failed to create temporary directory for Nimby install" >&2
    return 1
  fi

  local nimby_home="$HOME/.nimby"
  if ! mkdir -p "$nimby_home"; then
    echo "Failed to create $nimby_home" >&2
    rm -rf "$tmp_dir"
    return 1
  fi

  (
    set -e
    cd "$tmp_dir"
    echo "Downloading Nimby from $url"
    curl -fsSL "$url" -o nimby
    chmod +x nimby
    ./nimby use "$target_version"
    mv -f nimby "$nimby_home/nimby"
  )
  local status=$?
  rm -rf "$tmp_dir"

  if [ "$status" -ne 0 ]; then
    echo "Failed to install Nim $target_version via Nimby" >&2
    return 1
  fi

  local nim_path="$nimby_home/nim/bin"
  link_nimby_bins "$nim_path"

  return 0
}

link_nimby_bins() {
  local nim_path="$1"

  if [ -z "$nim_path" ] || [ ! -d "$nim_path" ]; then
    return 0
  fi

  local install_dir
  install_dir=$(get_install_dir)

  if [ -z "$install_dir" ]; then
    echo "Nim is installed in $nim_path. Add it to your PATH (e.g. export PATH=\"$nim_path:\$PATH\")."
    return 0
  fi

  if [ ! -d "$install_dir" ]; then
    if ! mkdir -p "$install_dir"; then
      echo "Unable to create $install_dir. Nim binaries remain in $nim_path." >&2
      return 1
    fi
  fi

  local linked_any=0
  for tool in nim nimble; do
    local src="$nim_path/$tool"
    local dest="$install_dir/$tool"

    if [ -x "$src" ]; then
      if ln -sf "$src" "$dest"; then
        linked_any=1
      fi
    fi
  done

  if [ "$linked_any" -eq 1 ]; then
    echo "Linked Nim binaries into $install_dir. Ensure this directory is in your PATH."
  else
    echo "Could not link Nim binaries into $install_dir. Binaries remain in $nim_path." >&2
  fi

  return 0
}

get_nimby_url() {
  local version="$1"
  local system
  system=$(uname -s)
  local machine
  machine=$(uname -m)

  case "$system" in
    Linux)
      case "$machine" in
        x86_64|amd64)
          echo "https://github.com/treeform/nimby/releases/download/${version}/nimby-Linux-X64"
          ;;
        *)
          echo "Unsupported Linux architecture for Nimby: $machine" >&2
          return 1
          ;;
      esac
      ;;
    Darwin)
      case "$machine" in
        arm64|aarch64)
          echo "https://github.com/treeform/nimby/releases/download/${version}/nimby-macOS-ARM64"
          ;;
        *)
          echo "Unsupported macOS architecture for Nimby: $machine" >&2
          return 1
          ;;
      esac
      ;;
    *)
      echo "Unsupported operating system for Nimby: $system" >&2
      return 1
      ;;
  esac
}

get_bazelisk_url() {
  local version="${1:-v1.19.0}"
  local system=$(uname -s | tr '[:upper:]' '[:lower:]')
  local machine=$(uname -m | tr '[:upper:]' '[:lower:]')

  local base="https://github.com/bazelbuild/bazelisk/releases/download/${version}/"

  if [[ "$system" == "linux" ]]; then
    if [[ "$machine" == "aarch64" ]] || [[ "$machine" == "arm64" ]]; then
      echo "${base}bazelisk-linux-arm64"
    else
      echo "${base}bazelisk-linux-amd64"
    fi
  elif [[ "$system" == "darwin" ]]; then
    if [[ "$machine" == "arm64" ]]; then
      echo "${base}bazelisk-darwin-arm64"
    else
      echo "${base}bazelisk-darwin-amd64"
    fi
  elif [[ "$system" == "mingw"* ]] || [[ "$system" == "msys"* ]]; then
    echo "${base}bazelisk-windows-amd64.exe"
  fi
}

ensure_tool "curl"
ensure_tool "g++"
ensure_tool "git"
ensure_tool "nim"
ensure_tool "nimble"
ensure_tool "bazel"
ensure_uv_setup
