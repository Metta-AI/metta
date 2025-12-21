#!/bin/bash
# Check cogames submissions and alert when scored

set -e

POLICY=""
VERSION=""
CHECK_INTERVAL=30
QUIET=false
ONCE=false

# Colors (disabled if not a terminal)
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NC='\033[0m' # No Color
else
    RED='' GREEN='' YELLOW='' BLUE='' BOLD='' NC=''
fi

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] [POLICY_NAME] [VERSION]

Monitor a cogames submission and alert when it receives a score.

Arguments:
  POLICY_NAME           Full policy name (e.g., harvest_policy_v2.3)
  VERSION               Optional version number (e.g., 3)

Options:
  -p, --policy NAME     Policy base name (e.g., harvest_policy)
  -v, --version VER     Version suffix (e.g., v2.3)
  -i, --interval SECS   Check interval in seconds (default: 30)
  -q, --quiet           Don't play sound alerts
  -1, --once            Check once and exit (don't loop)
  -h, --help            Show this help message

Examples:
  $(basename "$0") harvest_policy_v2.3
  $(basename "$0") harvest_policy_v2 3               # policy + version
  $(basename "$0") --policy harvest_policy_v2 --version 3
  $(basename "$0") harvest_policy                    # auto-detect latest
  $(basename "$0") harvest_policy_v2.3 --interval 60
  $(basename "$0") harvest_policy_v2.3 --once       # check once, no loop

If only --policy is provided without --version, the script will
auto-detect and watch the latest version on the leaderboard.
EOF
    exit 0
}

error() {
    echo -e "${RED}Error:${NC} $1" >&2
    echo "Try '$(basename "$0") --help' for usage information." >&2
    exit 1
}

# Clean exit on Ctrl+C
cleanup() {
    echo ""
    echo -e "${BLUE}Stopped watching.${NC}"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Function to play sound
play_alert() {
    if command -v afplay &> /dev/null; then
        afplay /System/Library/Sounds/Glass.aiff 2>/dev/null
    elif command -v paplay &> /dev/null; then
        paplay /usr/share/sounds/freedesktop/stereo/complete.oga 2>/dev/null || \
        paplay /usr/share/sounds/freedesktop/stereo/message.oga 2>/dev/null
    else
        echo -e "\a"
    fi
}

# Function to send desktop notification
send_notification() {
    local title="$1"
    local message="$2"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        osascript -e "display notification \"$message\" with title \"$title\"" 2>/dev/null || true
    elif command -v notify-send &> /dev/null; then
        # Linux with libnotify
        notify-send "$title" "$message" 2>/dev/null || true
    fi
}

# Function to get rank from leaderboard
get_rank() {
    local policy_name="$1"
    local rank=0
    while IFS= read -r line; do
        rank=$((rank + 1))
        if echo "$line" | grep -qF "$policy_name"; then
            echo "$rank"
            return
        fi
    done < <(.venv/bin/cogames leaderboard 2>/dev/null | tail -n +2)
    echo "?"
}

# Function to display score banner
show_score() {
    local policy="$1"
    local score="$2"
    local rank="$3"

    echo ""
    echo -e "${GREEN}${BOLD}==========================================${NC}"
    echo -e "${GREEN}${BOLD}  POLICY: ${NC}${policy}"
    echo -e "${GREEN}${BOLD}  SCORE:  ${NC}${score}"
    echo -e "${GREEN}${BOLD}  RANK:   ${NC}#${rank}"
    echo -e "${GREEN}${BOLD}==========================================${NC}"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -p|--policy)
            [[ -z "${2:-}" ]] && error "--policy requires an argument"
            POLICY="$2"
            shift 2
            ;;
        -v|--version)
            [[ -z "${2:-}" ]] && error "--version requires an argument"
            VERSION="$2"
            shift 2
            ;;
        -i|--interval)
            [[ -z "${2:-}" ]] && error "--interval requires an argument"
            if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                error "--interval must be a positive integer"
            fi
            CHECK_INTERVAL="$2"
            shift 2
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -1|--once)
            ONCE=true
            shift
            ;;
        -*)
            error "Unknown option: $1"
            ;;
        *)
            # Positional arguments: POLICY [VERSION]
            if [ -z "$POLICY" ]; then
                POLICY="$1"
            elif [ -z "$VERSION" ]; then
                VERSION="$1"
            else
                error "Unexpected argument: $1"
            fi
            shift
            ;;
    esac
done

if [ -z "$POLICY" ]; then
    error "No policy specified"
fi

# If no version specified, find the latest one
# Only skip auto-detect if version looks complete (e.g., v2.3, not just v2)
if [ -z "$VERSION" ] && [[ ! "$POLICY" =~ v[0-9]+\.[0-9]+ ]]; then
    echo -e "${BLUE}Finding latest version for:${NC} $POLICY"
    # Sort numerically by version number (v2, v2.1, v2.2, etc.)
    LEADERBOARD_LINE=$(.venv/bin/cogames leaderboard 2>/dev/null | grep -F "${POLICY}." | sort -V | tail -1)
    LATEST=$(echo "$LEADERBOARD_LINE" | awk '{print $1}')
    if [ -n "$LATEST" ]; then
        FULL_NAME="$LATEST"
    else
        error "No versions found for '$POLICY' on leaderboard"
    fi
elif [ -n "$VERSION" ]; then
    FULL_NAME="${POLICY}.${VERSION}"
else
    FULL_NAME="$POLICY"
fi

# Check if score already exists - if so, just report and exit
LINE=$(.venv/bin/cogames leaderboard 2>/dev/null | grep -F "${FULL_NAME}" | head -1)
if [ -n "$LINE" ]; then
    SCORE=$(echo "$LINE" | awk '{print $4}')
    if [[ "$SCORE" =~ ^[0-9]+\.[0-9]+$ ]]; then
        RANK=$(get_rank "$FULL_NAME")
        show_score "$FULL_NAME" "$SCORE" "$RANK"
        [[ "$QUIET" = false ]] && { play_alert; play_alert; play_alert; }
        send_notification "Score Ready!" "$FULL_NAME: $SCORE (Rank #$RANK)"
        exit 0
    fi
fi

# If --once flag, exit here if no score yet
if [[ "$ONCE" = true ]]; then
    echo -e "${YELLOW}No score yet for:${NC} $FULL_NAME"
    exit 1
fi

echo -e "${BLUE}Watching for score on:${NC} $FULL_NAME"
echo -e "${BLUE}Checking every ${CHECK_INTERVAL}s...${NC} (Ctrl+C to stop)"
echo ""

while true; do
    # Get exact match only
    LINE=$(.venv/bin/cogames leaderboard 2>/dev/null | grep -F "${FULL_NAME}" | head -1)

    if [ -n "$LINE" ]; then
        SCORE=$(echo "$LINE" | awk '{print $4}')

        if [[ "$SCORE" =~ ^[0-9]+\.[0-9]+$ ]]; then
            RANK=$(get_rank "$FULL_NAME")
            show_score "$FULL_NAME" "$SCORE" "$RANK"
            [[ "$QUIET" = false ]] && { play_alert; play_alert; play_alert; }
            send_notification "Score Ready!" "$FULL_NAME: $SCORE (Rank #$RANK)"
            exit 0
        else
            echo -e "${YELLOW}[$(date '+%H:%M:%S')]${NC} ${FULL_NAME}: Pending..."
        fi
    else
        echo -e "${YELLOW}[$(date '+%H:%M:%S')]${NC} ${FULL_NAME}: Not found"
    fi

    sleep $CHECK_INTERVAL
done
