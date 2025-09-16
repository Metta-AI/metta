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

print_job_debug_report() {
    echo "[DEBUG] ========== JOB DEBUG REPORT =========="
    echo "[DEBUG] Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "[DEBUG] Script PID: $$"
    echo "[DEBUG] CMD_PID: ${CMD_PID:-'not set'}"
    echo "[DEBUG] CMD_PGID: ${CMD_PGID:-'not set'}"

    # Check current job status
    echo "[DEBUG] Current job table:"
    jobs -l 2>&1 || echo "  No jobs found"

    # Check all background jobs
    echo "[DEBUG] Background job PIDs:"
    jobs -p 2>&1 || echo "  No background jobs"

    # Check process group if we have PGID
    if [[ -n "${CMD_PGID:-}" ]]; then
        echo "[DEBUG] Process group ${CMD_PGID} members:"
        ps -eo pid,ppid,pgid,state,etime,cmd | grep "^\s*[0-9]\+\s\+[0-9]\+\s\+${CMD_PGID}" 2>&1 || echo "  No processes found in group"
    fi

    # Check for any child processes of this script
    echo "[DEBUG] Child processes of script (PID $$):"
    ps --ppid $$ -o pid,ppid,pgid,state,etime,cmd 2>&1 || echo "  No child processes found"

    # Check for zombie processes
    echo "[DEBUG] Zombie processes:"
    ps aux | grep " <defunct>" | grep -v grep || echo "  No zombie processes found"

    # Check for processes matching the module name
    if [[ -n "${METTA_MODULE_PATH:-}" ]]; then
        local module_name=$(basename "${METTA_MODULE_PATH}")
        echo "[DEBUG] Processes matching module '$module_name':"
        ps aux | grep "$module_name" | grep -v grep || echo "  No matching processes found"
    fi

    # Check monitor processes if they exist
    echo "[DEBUG] Monitor processes:"
    ps aux | grep -E "(monitor_gpu|monitor_memory|cluster_stop_monitor)" | grep -v grep || echo "  No monitor processes found"

    # System resource state
    echo "[DEBUG] System state:"
    echo "  Load average: $(uptime | awk -F'load average:' '{print $2}')"
    echo "  Memory usage:"
    free -h | sed 's/^/    /'

    # Check for any processes that might be holding resources
    echo "[DEBUG] Top 5 CPU consuming processes:"
    ps aux --sort=-%cpu | head -6 | tail -5 | sed 's/^/  /'

    echo "[DEBUG] ========== END DEBUG REPORT =========="
}
