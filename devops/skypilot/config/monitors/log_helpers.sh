#!/usr/bin/env bash

# Enhanced logging function with millisecond precision
log() {
    local level="$1"
    local message="$2"

    # Get current time with nanoseconds and convert to milliseconds
    local timestamp
    if command -v gdate &> /dev/null; then
        # macOS with GNU coreutils
        timestamp=$(gdate '+%H:%M:%S.%3N')
    elif date --version &> /dev/null 2>&1; then
        # GNU date (Linux)
        timestamp=$(date '+%H:%M:%S.%3N')
    else
        # Fallback to seconds only
        timestamp=$(date '+%H:%M:%S')
    fi

    # Color codes for different log levels (optional)
    local color_reset='\033[0m'
    local color=""

    # Enable colors only if output is to a terminal
    if [ -t 1 ]; then
        case "$level" in
            ERROR)
                color='\033[0;31m'  # Red
                ;;
            WARN)
                color='\033[0;33m'  # Yellow
                ;;
            INFO)
                color='\033[0;32m'  # Green
                ;;
            DEBUG)
                color='\033[0;36m'  # Cyan
                ;;
        esac
    fi

    # Format: [HH:MM:SS.mmm] LEVEL    message
    printf "[%s] ${color}%-8s${color_reset} %s\n" "$timestamp" "[$level]" "$message"
}

# Convenience functions
log_info() {
    log "INFO" "$1"
}

log_error() {
    log "ERROR" "$1"
}

log_warn() {
    log "WARN" "$1"
}

log_debug() {
    if [ -n "${DEBUG:-}" ]; then
        log "DEBUG" "$1"
    fi
}
