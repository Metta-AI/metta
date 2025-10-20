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

  if [ "$(uname -s)" = "Linux" ] && { [ "$tool" = "nim" ] || [ "$tool" = "nimble" ]; }; then
    if ensure_linux_nim_version "$REQUIRED_NIM_VERSION"; then
      ensure_paths
      return 0
    fi

    err "Failed to install Nim via choosenim"
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
REQUIRED_NIM_VERSION="2.2.4"

# Common install directories in order of preference
COMMON_INSTALL_DIRS="/usr/local/bin /usr/bin /opt/bin $HOME/.local/bin $HOME/bin $HOME/.nimble/bin $HOME/.cargo/bin /opt/homebrew/bin"

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
}

version_ge() {
  local current="$1"
  local required="$2"

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

ensure_linux_nim_version() {
  local required_version="$1"

  ensure_paths

  local current_version=""
  if check_cmd nim; then
    current_version=$(get_nim_version 2> /dev/null || echo "")
  fi

  if [ -n "$current_version" ] && version_ge "$current_version" "$required_version" && check_cmd nimble; then
    link_nim_bins
    return 0
  fi

  if [ -n "$current_version" ]; then
    echo "Found Nim $current_version but require >= $required_version. Upgrading via choosenim..."
  else
    echo "Nim not found. Installing via choosenim..."
  fi

  if ! install_nim_via_choosenim "$required_version"; then
    return 1
  fi

  ensure_paths
  current_version=$(get_nim_version 2> /dev/null || echo "")

  if [ -n "$current_version" ] && version_ge "$current_version" "$required_version" && check_cmd nimble; then
    echo "Nim $current_version with nimble found."
    link_nim_bins
    return 0
  fi

  echo "Nim install finished but requirements are still not satisfied." >&2
  return 1
}

link_nim_bins() {
  local nimble_dir="$HOME/.nimble/bin"

  if [ ! -d "$nimble_dir" ]; then
    return 0
  fi

  local install_dir=$(get_install_dir)

  if [ -z "$install_dir" ]; then
    echo "Nim is installed in $nimble_dir. Add it to your PATH (e.g. export PATH=\"$nimble_dir:\$PATH\")."
    return 0
  fi

  if [ ! -d "$install_dir" ]; then
    if ! mkdir -p "$install_dir"; then
      echo "Unable to create $install_dir. Nim binaries remain in $nimble_dir." >&2
      return 1
    fi
  fi

  local linked_any=0
  for tool in nim nimble; do
    local src="$nimble_dir/$tool"
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
    echo "Could not link Nim binaries into $install_dir. Binaries remain in $nimble_dir." >&2
  fi
}

install_nim_via_choosenim() {
  if [ "$(uname -s)" != "Linux" ]; then
    return 1
  fi

  local target_version="${1:-}"
  local choosenim_args="-y"
  if [ -n "$target_version" ]; then
    choosenim_args="$choosenim_args $target_version"
  fi

  echo "Installing Nim via choosenim${target_version:+ (target $target_version)}..."

  if ! env CHOOSENIM_NO_ANALYTICS=1 CHOOSENIM_NO_COLOR=1 bash -lc "curl https://nim-lang.org/choosenim/init.sh -sSf | sh -s -- $choosenim_args"; then
    echo "Failed to run choosenim installer" >&2
    return 1
  fi

  ensure_paths

  if check_cmd nim && check_cmd nimble; then
    echo "Nim installed successfully via choosenim."
    return 0
  fi

  echo "Nim install finished but binaries are still missing. Ensure ~/.nimble/bin is in your PATH or install manually." >&2
  return 1
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
ensure_bazel_setup
ensure_uv_setup
