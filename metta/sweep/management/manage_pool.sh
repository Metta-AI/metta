#!/bin/bash
# Worker pool management script for distributed sweep execution

set -e

# Default configuration
DEFAULT_NUM_WORKERS=4
DEFAULT_GPU_TYPE="L4:1"
DEFAULT_POLL_INTERVAL=10

# Color output for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if required environment variables are set
check_env() {
    if [ -z "$POSTGRES_URL" ]; then
        echo -e "${RED}Error: POSTGRES_URL environment variable is not set${NC}"
        echo "Please export POSTGRES_URL=postgresql://user:password@host:port/dbname"
        exit 1
    fi
    echo -e "${GREEN}✓ POSTGRES_URL is set${NC}"
}

# Launch workers using launch.py
launch_workers() {
    NUM_WORKERS=${1:-$DEFAULT_NUM_WORKERS}
    GPU_TYPE=${2:-$DEFAULT_GPU_TYPE}

    echo -e "${YELLOW}Launching $NUM_WORKERS workers with GPU type $GPU_TYPE...${NC}"

    for i in $(seq 1 $NUM_WORKERS); do
        WORKER_ID="worker-$(hostname)-$i"
        echo -e "${GREEN}Launching $WORKER_ID...${NC}"

        # Use launch.py to start the worker tool via run.py
        ./devops/skypilot/launch.py worker \
            db_url="$POSTGRES_URL" \
            worker_id="$WORKER_ID" \
            poll_interval=$DEFAULT_POLL_INTERVAL \
            --gpus="$GPU_TYPE" \
            --detach \
            --name="$WORKER_ID" &

        # Small delay between launches
        sleep 2
    done

    echo -e "${GREEN}✓ Launched $NUM_WORKERS workers${NC}"
    echo "Workers will appear in 'sky status' once provisioned"
}

# Terminate all workers
terminate_workers() {
    echo -e "${YELLOW}Terminating all workers...${NC}"

    # Get list of running workers from sky status
    WORKERS=$(sky status --refresh | grep "worker-" | awk '{print $1}' || true)

    if [ -z "$WORKERS" ]; then
        echo -e "${YELLOW}No workers found${NC}"
        return
    fi

    for WORKER in $WORKERS; do
        echo -e "${RED}Terminating $WORKER...${NC}"
        sky down "$WORKER" --yes || true
    done

    echo -e "${GREEN}✓ All workers terminated${NC}"
}

# Show worker status from database
show_status() {
    echo -e "${YELLOW}Fetching worker status from database...${NC}"

    # Use psql to query the worker_status table
    psql "$POSTGRES_URL" -c "
        SELECT
            worker_id,
            status,
            current_job_id,
            last_heartbeat,
            EXTRACT(EPOCH FROM (NOW() - last_heartbeat)) as seconds_since_heartbeat
        FROM worker_status
        WHERE last_heartbeat > NOW() - INTERVAL '5 minutes'
        ORDER BY last_heartbeat DESC;
    " 2>/dev/null || echo -e "${RED}Failed to query database. Is PostgreSQL running?${NC}"

    # Also show queue status
    echo -e "\n${YELLOW}Queue Status:${NC}"
    psql "$POSTGRES_URL" -c "
        SELECT
            status,
            COUNT(*) as count
        FROM job_queue
        WHERE created_at > NOW() - INTERVAL '24 hours'
        GROUP BY status
        ORDER BY status;
    " 2>/dev/null || true

    # Show Sky status
    echo -e "\n${YELLOW}Sky Status:${NC}"
    sky status --refresh | grep "worker-" || echo "No workers in sky status"
}

# Initialize database tables
init_db() {
    echo -e "${YELLOW}Initializing database tables...${NC}"

    # Use Python script for database initialization (no psql required)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SETUP_SCRIPT="$SCRIPT_DIR/../database/setup.py"

    if [ ! -f "$SETUP_SCRIPT" ]; then
        echo -e "${RED}Error: Setup script not found at $SETUP_SCRIPT${NC}"
        exit 1
    fi

    # Run the Python script with the POSTGRES_URL
    uv run python "$SETUP_SCRIPT"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Database tables created/verified${NC}"
    else
        echo -e "${RED}Failed to initialize database${NC}"
        echo "Please check if your Supabase project is paused at https://app.supabase.com"
        exit 1
    fi
}

# Launch a test sweep with worker pool
test_sweep() {
    echo -e "${YELLOW}Launching test sweep with worker pool...${NC}"

    # Launch controller locally or on cloud
    ./devops/skypilot/launch.py sweep \
        experiment="worker_pool_test" \
        dispatcher_type="remote_queue" \
        db_url="$POSTGRES_URL" \
        max_trials=10 \
        protein_config.metric="evaluator/eval_arena/score" \
        --local  # Run controller locally for testing

    echo -e "${GREEN}✓ Test sweep launched${NC}"
}

# Main menu
show_help() {
    echo "Worker Pool Management Script"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  init              Initialize database tables"
    echo "  launch [N] [GPU]  Launch N workers (default: 4) with GPU type (default: L4:1)"
    echo "  terminate         Terminate all workers"
    echo "  status            Show worker and queue status"
    echo "  test              Launch a test sweep with worker pool"
    echo "  help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 init                    # Initialize database"
    echo "  $0 launch                  # Launch 4 workers with L4 GPUs"
    echo "  $0 launch 8 A100:1         # Launch 8 workers with A100 GPUs"
    echo "  $0 status                  # Check worker status"
    echo "  $0 terminate               # Stop all workers"
    echo ""
    echo "Environment Variables:"
    echo "  POSTGRES_URL     Required: PostgreSQL connection string"
}

# Parse command line arguments
case "${1:-help}" in
    init)
        check_env
        init_db
        ;;
    launch)
        check_env
        launch_workers "$2" "$3"
        ;;
    terminate)
        terminate_workers
        ;;
    status)
        check_env
        show_status
        ;;
    test)
        check_env
        test_sweep
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac