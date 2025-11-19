#!/usr/bin/env bash
set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Default values
DEFAULT_GPUS="1"
DEFAULT_NUM_WORKERS="1"
DEFAULT_POLL_INTERVAL="10"
DEFAULT_IDLE_TIMEOUT="600"
DEFAULT_HEARTBEAT_INTERVAL="3600"

# Help function
show_help() {
    cat << EOF
Usage: $(basename "$0") COMMAND [OPTIONS]

Commands:
    init        Initialize database tables
    launch      Launch worker pool(s) on SkyPilot
    status      Show status of running workers and queue
    stop        Stop worker pool(s)
    test        Launch a test sweep with worker pool
    help        Show this help message

Launch Options:
    --num-workers NUM      Number of workers to launch (default: $DEFAULT_NUM_WORKERS)
    --gpus NUM            Number of GPUs per worker (default: $DEFAULT_GPUS)
    --group NAME          Group name for workers (optional, if not set workers pull any job)
    --db-url URL          PostgreSQL connection URL (or set POSTGRES_URL env var)
    --poll-interval SEC   Seconds between polls (default: $DEFAULT_POLL_INTERVAL)
    --idle-timeout SEC    Seconds before idle worker shuts down (default: $DEFAULT_IDLE_TIMEOUT, 0=never)

Examples:
    # Initialize database
    $(basename "$0") init

    # Launch workers for specific group
    $(basename "$0") launch --group experiment1 --num-workers 3 --gpus 2

    # Launch workers for any job (no group filter)
    $(basename "$0") launch --num-workers 4 --gpus 1

    # Check status
    $(basename "$0") status
    $(basename "$0") status --group experiment1

    # Stop workers
    $(basename "$0") stop --group experiment1
    $(basename "$0") stop --all

    # Launch test sweep
    $(basename "$0") test

Environment Variables:
    POSTGRES_URL     Required: PostgreSQL connection string

EOF
}

# Check environment
check_env() {
    if [ -z "${POSTGRES_URL:-}" ]; then
        echo -e "${RED}Error: POSTGRES_URL environment variable is not set${NC}"
        echo "Please export POSTGRES_URL=postgresql://user:password@host:port/dbname"
        return 1
    fi
    echo -e "${GREEN}✓ POSTGRES_URL is set${NC}"
    return 0
}

# Parse launch arguments
parse_launch_args() {
    local num_workers="$DEFAULT_NUM_WORKERS"
    local gpus="$DEFAULT_GPUS"
    local group=""
    local db_url="${POSTGRES_URL:-}"
    local poll_interval="$DEFAULT_POLL_INTERVAL"
    local idle_timeout="$DEFAULT_IDLE_TIMEOUT"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --num-workers|--num_workers)
                num_workers="$2"
                shift 2
                ;;
            --gpus)
                gpus="$2"
                shift 2
                ;;
            --group)
                group="$2"
                shift 2
                ;;
            --db-url|--db_url)
                db_url="$2"
                shift 2
                ;;
            --poll-interval|--poll_interval)
                poll_interval="$2"
                shift 2
                ;;
            --idle-timeout|--idle_timeout)
                idle_timeout="$2"
                shift 2
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                echo -e "${RED}Error: Unknown option $1${NC}" >&2
                show_help
                exit 1
                ;;
        esac
    done

    # Validate required parameters
    if [[ -z "$db_url" ]]; then
        echo -e "${RED}Error: Database URL required. Set POSTGRES_URL or use --db-url${NC}" >&2
        exit 1
    fi

    # Export for use in launch function
    export LAUNCH_NUM_WORKERS="$num_workers"
    export LAUNCH_GPUS="$gpus"
    export LAUNCH_GROUP="$group"
    export LAUNCH_DB_URL="$db_url"
    export LAUNCH_POLL_INTERVAL="$poll_interval"
    export LAUNCH_IDLE_TIMEOUT="$idle_timeout"
}

# Initialize database tables
init_db() {
    echo -e "${YELLOW}Initializing database tables...${NC}"

    # Look for setup script in multiple locations
    local setup_script=""
    for path in "$SCRIPT_DIR/../database/setup.py" "$PROJECT_ROOT/metta/sweep/database/setup.py"; do
        if [ -f "$path" ]; then
            setup_script="$path"
            break
        fi
    done

    if [ -z "$setup_script" ]; then
        echo -e "${YELLOW}Setup script not found, creating tables directly...${NC}"

        # Create tables using psql if available, otherwise skip
        if command -v psql &> /dev/null; then
            psql "$POSTGRES_URL" << EOF 2>/dev/null || true
                -- Create job_queue table
                CREATE TABLE IF NOT EXISTS job_queue (
                    id SERIAL PRIMARY KEY,
                    job_id VARCHAR(255) UNIQUE NOT NULL,
                    job_definition JSONB NOT NULL,
                    status VARCHAR(50) DEFAULT 'pending',
                    worker_id VARCHAR(255),
                    group_id VARCHAR(255),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    claimed_at TIMESTAMP WITH TIME ZONE,
                    started_at TIMESTAMP WITH TIME ZONE,
                    completed_at TIMESTAMP WITH TIME ZONE,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    metadata JSONB
                );

                -- Create worker_status table
                CREATE TABLE IF NOT EXISTS worker_status (
                    worker_id VARCHAR(255) PRIMARY KEY,
                    hostname VARCHAR(255),
                    group_id VARCHAR(255),
                    last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    status VARCHAR(50) DEFAULT 'idle',
                    current_job_id VARCHAR(255),
                    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    metadata JSONB
                );

                -- Create job_signals table for multi-node coordination
                CREATE TABLE IF NOT EXISTS job_signals (
                    id SERIAL PRIMARY KEY,
                    job_id VARCHAR(255) NOT NULL,
                    job_definition JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    expires_at TIMESTAMP DEFAULT (NOW() + INTERVAL '5 minutes'),
                    CONSTRAINT unique_job_signal UNIQUE (job_id)
                );

                -- Create indexes for performance
                CREATE INDEX IF NOT EXISTS idx_job_queue_status ON job_queue(status);
                CREATE INDEX IF NOT EXISTS idx_job_queue_group ON job_queue(group_id);
                CREATE INDEX IF NOT EXISTS idx_worker_status_heartbeat ON worker_status(last_heartbeat);
EOF
            echo -e "${GREEN}✓ Database tables created/verified${NC}"
        else
            echo -e "${YELLOW}psql not found, assuming tables exist${NC}"
        fi
    else
        # Run the Python setup script
        uv run python "$setup_script"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Database tables created/verified${NC}"
        else
            echo -e "${RED}Failed to initialize database${NC}"
            exit 1
        fi
    fi
}

# Launch worker pool
launch_workers() {
    local num_workers="$LAUNCH_NUM_WORKERS"
    local gpus="$LAUNCH_GPUS"
    local group="$LAUNCH_GROUP"
    local db_url="$LAUNCH_DB_URL"
    local poll_interval="$LAUNCH_POLL_INTERVAL"
    local idle_timeout="$LAUNCH_IDLE_TIMEOUT"

    if [[ -n "$group" ]]; then
        echo -e "${BLUE}Launching $num_workers worker(s) for group: $group${NC}"
    else
        echo -e "${BLUE}Launching $num_workers worker(s) for any job (no group filter)${NC}"
    fi
    echo -e "${BLUE}Configuration: GPUs=$gpus per worker, poll_interval=${poll_interval}s, idle_timeout=${idle_timeout}s${NC}"
    echo ""

    local success_count=0
    local failed_count=0

    for ((i=0; i<num_workers; i++)); do
        local worker_id
        if [[ -n "$group" ]]; then
            worker_id="${group}.worker_${i}"
        else
            worker_id="worker_${i}_$(date +%s)"
        fi

        echo -e "${YELLOW}Launching worker $((i+1))/$num_workers: $worker_id${NC}"

        # Build the command using the correct launch pattern
        local cmd="uv run $PROJECT_ROOT/devops/skypilot/launch.py \
            metta.sweep.tools.worker.WorkerTool"

        # Add required parameters
        cmd="$cmd db_url=\"$db_url\""
        cmd="$cmd worker_id=$worker_id"

        # Add optional group parameter
        if [[ -n "$group" ]]; then
            cmd="$cmd group=$group"
        fi

        # Add additional parameters
        cmd="$cmd poll_interval=$poll_interval"
        cmd="$cmd idle_timeout=$idle_timeout"

        # Add SkyPilot options
        cmd="$cmd -hb $DEFAULT_HEARTBEAT_INTERVAL --gpus=$gpus --no-torch"

        # Execute and capture output
        if output=$(cd "$PROJECT_ROOT" && eval "$cmd" 2>&1); then
            echo -e "${GREEN}✓ Successfully launched $worker_id${NC}"

            # Extract cluster name from output if available
            if cluster_name=$(echo "$output" | grep -oE "Cluster '.*' is up" | sed "s/Cluster '\(.*\)' is up/\1/"); then
                echo -e "  Cluster: $cluster_name"
            fi

            ((success_count++))
        else
            echo -e "${RED}✗ Failed to launch $worker_id${NC}"
            echo -e "${RED}  Error: $output${NC}"
            ((failed_count++))
        fi

        echo ""
    done

    # Summary
    echo -e "${BLUE}=== Launch Summary ===${NC}"
    echo -e "${GREEN}Successfully launched: $success_count worker(s)${NC}"
    if [[ $failed_count -gt 0 ]]; then
        echo -e "${RED}Failed to launch: $failed_count worker(s)${NC}"
        return 1
    fi

    echo ""
    echo -e "${BLUE}To check worker status, run:${NC}"
    echo "  $(basename "$0") status"
    if [[ -n "$group" ]]; then
        echo "  $(basename "$0") status --group $group"
    fi
}

# Show status of workers and queue
show_status() {
    local group="${1:-}"
    local show_sky=true
    local show_db=true

    # Parse status options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --group)
                group="$2"
                shift 2
                ;;
            --sky-only)
                show_db=false
                shift
                ;;
            --db-only)
                show_sky=false
                shift
                ;;
            *)
                shift
                ;;
        esac
    done

    # Show database status
    if [ "$show_db" = true ] && [ -n "${POSTGRES_URL:-}" ]; then
        echo -e "${YELLOW}=== Database Worker Status ===${NC}"

        if command -v psql &> /dev/null; then
            # Worker status query
            local group_filter=""
            if [[ -n "$group" ]]; then
                group_filter="AND group_id = '$group'"
            fi

            psql "$POSTGRES_URL" -c "
                SELECT
                    worker_id,
                    group_id,
                    status,
                    current_job_id,
                    last_heartbeat,
                    EXTRACT(EPOCH FROM (NOW() - last_heartbeat))::INT as seconds_since_heartbeat
                FROM worker_status
                WHERE last_heartbeat > NOW() - INTERVAL '10 minutes'
                    $group_filter
                ORDER BY group_id, worker_id;
            " 2>/dev/null || echo -e "${YELLOW}No database connection available${NC}"

            # Queue status
            echo -e "\n${YELLOW}=== Queue Status ===${NC}"
            psql "$POSTGRES_URL" -c "
                SELECT
                    status,
                    group_id,
                    COUNT(*) as count
                FROM job_queue
                WHERE created_at > NOW() - INTERVAL '24 hours'
                GROUP BY status, group_id
                ORDER BY group_id, status;
            " 2>/dev/null || true
        else
            echo -e "${YELLOW}psql not available, skipping database status${NC}"
        fi
    fi

    # Show SkyPilot status
    if [ "$show_sky" = true ]; then
        echo -e "\n${YELLOW}=== SkyPilot Clusters ===${NC}"

        if [[ -n "$group" ]]; then
            echo -e "${BLUE}Filtering for group: $group${NC}"
            sky status --refresh | grep -E "${group}\.worker_[0-9]+" || {
                echo -e "${YELLOW}No SkyPilot clusters found for group: $group${NC}"
            }
        else
            sky status --refresh | grep -E "worker" || {
                echo -e "${YELLOW}No worker clusters found${NC}"
            }
        fi
    fi
}

# Stop workers
stop_workers() {
    local group=""
    local stop_all=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --group)
                group="$2"
                shift 2
                ;;
            --all)
                stop_all=true
                shift
                ;;
            *)
                shift
                ;;
        esac
    done

    if [[ "$stop_all" = false ]] && [[ -z "$group" ]]; then
        echo -e "${RED}Error: Specify --group NAME or --all to stop workers${NC}" >&2
        exit 1
    fi

    if [[ "$stop_all" = true ]]; then
        echo -e "${BLUE}Stopping all workers...${NC}"
        local clusters=$(sky status --refresh | grep -oE "worker[^ ]*" | sort -u)
    else
        echo -e "${BLUE}Stopping workers for group: $group${NC}"
        local clusters=$(sky status --refresh | grep -oE "${group}\.worker_[0-9]+" | sort -u)
    fi

    if [[ -z "$clusters" ]]; then
        echo -e "${YELLOW}No workers found${NC}"
        return 0
    fi

    local count=0
    for cluster in $clusters; do
        echo -e "${YELLOW}Stopping cluster: $cluster${NC}"
        if sky down "$cluster" -y; then
            echo -e "${GREEN}✓ Successfully stopped $cluster${NC}"
            ((count++))
        else
            echo -e "${RED}✗ Failed to stop $cluster${NC}"
        fi
        echo ""
    done

    echo -e "${GREEN}Stopped $count worker(s)${NC}"
}

# Launch a test sweep with worker pool
test_sweep() {
    echo -e "${YELLOW}Launching test sweep with worker pool...${NC}"

    check_env || return 1

    # Launch sweep controller using correct syntax for sweep tool
    local cmd="uv run $PROJECT_ROOT/devops/skypilot/launch.py \
        metta.sweep.tools.sweep.SweepTool \
        experiment=worker_pool_test \
        dispatcher_type=remote_queue \
        db_url=\"$POSTGRES_URL\" \
        max_trials=10 \
        protein_config.metric=evaluator/eval_arena/score \
        --local"

    echo -e "${BLUE}Running: $cmd${NC}"

    if cd "$PROJECT_ROOT" && eval "$cmd"; then
        echo -e "${GREEN}✓ Test sweep launched${NC}"
    else
        echo -e "${RED}Failed to launch test sweep${NC}"
        return 1
    fi
}

# Main command handler
main() {
    if [[ $# -eq 0 ]]; then
        show_help
        exit 1
    fi

    local command="$1"
    shift

    case "$command" in
        init)
            check_env || exit 1
            init_db
            ;;
        launch)
            parse_launch_args "$@"
            launch_workers
            ;;
        status)
            show_status "$@"
            ;;
        stop)
            stop_workers "$@"
            ;;
        test)
            test_sweep
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo -e "${RED}Unknown command: $command${NC}" >&2
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"