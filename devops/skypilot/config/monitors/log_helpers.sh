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
    local report=""

    report+="[DEBUG] ========== JOB DEBUG REPORT ==========\n"
    report+="[DEBUG] Timestamp: $(date '+%Y-%m-%d %H:%M:%S')\n"
    report+="[DEBUG] Script PID: $$\n"
    report+="[DEBUG] CMD_PID: ${CMD_PID:-'not set'}\n"

    # Check current job status
    report+="[DEBUG] Current job table:\n"
    report+="$(jobs -l 2>&1 || echo '  No jobs found')\n"

    # Check all background jobs
    report+="[DEBUG] Background job PIDs:\n"
    report+="$(jobs -p 2>&1 || echo '  No background jobs')\n"

    # Check process group (python -m mode has PGID = CMD_PID)
    report+="[DEBUG] Process group ${CMD_PID} members:\n"
    report+="$(ps -eo pid,ppid,pgid,state,etime,cmd | grep "^\s*[0-9]\+\s\+[0-9]\+\s\+${CMD_PID}" 2>&1 || echo '  No processes found in group')\n"

    # Check for any child processes of this script
    report+="[DEBUG] Child processes of script (PID $$):\n"
    report+="$(ps --ppid $$ -o pid,ppid,pgid,state,etime,cmd 2>&1 || echo '  No child processes found')\n"

    # Check for zombie processes
    report+="[DEBUG] Zombie processes:\n"
    report+="$(ps aux | grep ' <defunct>' | grep -v grep || echo '  No zombie processes found')\n"

    # Check for processes matching the module name
    if [[ -n "${METTA_MODULE_PATH:-}" ]]; then
        local module_name=$(basename "${METTA_MODULE_PATH}")
        report+="[DEBUG] Processes matching module '$module_name':\n"
        report+="$(ps aux | grep "$module_name" | grep -v grep || echo '  No matching processes found')\n"
    fi

    # Check monitor processes if they exist
    report+="[DEBUG] Monitor processes:\n"
    report+="$(ps aux | grep -E '(monitor_gpu|monitor_memory|cluster_stop_monitor)' | grep -v grep || echo '  No monitor processes found')\n"

    # System resource state
    report+="[DEBUG] System state:\n"
    report+="  Load average: $(uptime | awk -F'load average:' '{print $2}')\n"
    report+="  Memory usage:\n"
    report+="$(free -h | sed 's/^/    /')\n"

    # Check for any processes that might be holding resources
    report+="[DEBUG] Top 5 CPU consuming processes:\n"
    report+="$(ps aux --sort=-%cpu | head -6 | tail -5 | sed 's/^/  /')\n"

    report+="[DEBUG] ========== END DEBUG REPORT ==========\n"

    # Output everything at once
    echo -ne "$report"
}
