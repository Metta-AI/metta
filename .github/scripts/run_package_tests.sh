#!/bin/bash
# Local reproduction of the GitHub Actions run_package_tests job

set -e # Exit on error

# Keep single-threaded during tests to avoid flaky crashes in xdist workers.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-1}
export PYTORCH_NUM_THREADS=${PYTORCH_NUM_THREADS:-1}

# Colors for output
RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
MAGENTA='\033[1;35m'
CYAN='\033[1;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Local setup (only run when not in CI)
if [ "${CI}" != "true" ]; then
  echo -e "${CYAN}============================================${NC}"
  echo -e "${CYAN}Local reproduction of run_package_tests job${NC}"
  echo -e "${CYAN}============================================${NC}"

  # Check if uv is installed
  if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ 'uv' is not installed. Please install it first:${NC}"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
  fi

  # Check if we're in the metta repository
  if [ ! -f "pyproject.toml" ] || [ ! -d "packages/mettagrid" ]; then
    echo -e "${RED}❌ This script must be run from the metta repository root${NC}"
    exit 1
  fi

  # Setup virtual environment with testing dependencies
  echo -e "\n${YELLOW}📦 Setting up Python environment...${NC}"
  if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
  fi

  echo "Installing testing dependencies..."
  uv sync --no-dev --group testing
fi

# Create directories for test results
echo -e "\n${YELLOW}📁 Creating test result directories...${NC}"
mkdir -p test-results coverage-reports

# Determine which package suites should run (defaults to true)
RUN_APP_BACKEND_TESTS=${RUN_APP_BACKEND_TESTS:-true}

# Define the test runner function
run_package_tests() {
  local package=$1
  local color=$2

  # Extract just the package name from paths like "packages/mettagrid"
  local package_name=$(basename "$package")

  # Determine the relative path prefix based on package depth
  local path_prefix="../"
  if [[ "$package" == packages/* ]]; then
    path_prefix="../../"
  fi

  # Pytest arguments matching CI
  PYTEST_BASE_ARGS="-n 4 --timeout=100 --timeout-method=thread --cov --cov-branch --benchmark-skip --maxfail=1 --disable-warnings --durations=10 -v"

  # Save raw output for duration parsing
  local raw_output="test-results/${package_name}_raw.log"

  echo -e "${color}[${package_name}]${NC} Starting tests..."

  # Skip packages when requested (currently only app_backend is gated)
  if [[ "$package_name" == "app_backend" && "$RUN_APP_BACKEND_TESTS" != "true" ]]; then
    echo -e "${color}[${package_name}]${NC} Skipping tests (no app_backend changes detected)"
    echo 0 > "test-results/${package_name}.exit"
    return
  fi

  # Run tests and prefix each line with package name and color
  if [ "$package" == "core" ]; then
    (
      uv run pytest $PYTEST_BASE_ARGS \
        --cov-report=xml:coverage-reports/coverage-${package_name}.xml \
        2>&1
      echo $? > test-results/${package_name}.exit
    ) | tee "$raw_output" | while IFS= read -r line; do
      echo -e "${color}[${package_name}]${NC} $line"
    done
  else
    (
      cd "$package" && uv run pytest $PYTEST_BASE_ARGS \
        --cov-report=xml:${path_prefix}coverage-reports/coverage-${package_name}.xml \
        2>&1
      echo $? > ${path_prefix}test-results/${package_name}.exit
    ) | tee "$raw_output" | while IFS= read -r line; do
      echo -e "${color}[${package_name}]${NC} $line"
    done
  fi

  # Extract duration info for later summary
  grep -E "^[0-9]+\.[0-9]+s " "$raw_output" > "test-results/${package_name}_durations.txt" || true
}

# Function to extract and display failed test stacktraces
print_failed_test_stacktraces() {
  local failed_packages="$1"

  echo -e "\n${RED}📋 FAILED TEST DETAILS:${NC}"
  echo -e "${WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

  for package_name in $failed_packages; do
    local log_file="test-results/${package_name}_raw.log"

    if [ -f "$log_file" ]; then
      echo -e "\n${RED}🔍 Failed tests in package: ${package_name}${NC}"
      echo -e "${WHITE}─────────────────────────────────────────────${NC}"

      # Extract failed test names and their stacktraces
      # Look for FAILED lines and the context around them
      awk '
                /^=+ FAILURES =+/ { in_failures=1; next }
                /^=+ short test summary info =+/ { in_failures=0 }
                /^=+ [0-9]+ failed/ { in_failures=0 }
                in_failures && /^_+ .*_+$/ {
                    # Test separator line
                    print "\n" $0
                    next
                }
                in_failures && /^FAILED / {
                    # Failed test line
                    print "❌ " $0
                    next
                }
                in_failures && /./ {
                    # Stacktrace content
                    print $0
                }
            ' "$log_file"

      # Also look for short test summary
      echo -e "\n${YELLOW}📝 Short test summary for ${package_name}:${NC}"
      grep -C 100 "short test summary info" "$log_file" || echo "No short summary found"
    else
      echo -e "\n${RED}❌ Log file not found for package: ${package_name}${NC}"
    fi
  done
}

# Export function for parallel execution
export -f run_package_tests

# Check if user wants sequential or parallel execution
if [ "$1" == "--sequential" ]; then
  PARALLEL=false
  echo -e "\n${YELLOW}🔄 Running tests sequentially (use without --sequential for parallel)...${NC}"
else
  PARALLEL=true
  echo -e "\n${YELLOW}⏳ Running tests in parallel (use --sequential for sequential)...${NC}"
fi

# Record start time
START_TIME=$(date +%s)

if [ "$PARALLEL" == true ]; then
  # Run all packages in parallel (matching CI)
  run_package_tests "packages/mettagrid" "$BLUE" & # Bold Blue
  sleep 2                                          # mettagrid is slowest, so give time for it to grab resources

  run_package_tests "agent" "$RED" &              # Bold Red
  run_package_tests "common" "$GREEN" &           # Bold Green
  run_package_tests "app_backend" "$YELLOW" &     # Bold Yellow
  run_package_tests "codebot" "$MAGENTA" &        # Bold Magenta
  run_package_tests "core" "$CYAN" &              # Bold Cyan
  run_package_tests "packages/cogames" "$WHITE" & # Bold White

  # Wait for all background jobs to complete
  wait
else
  # Run sequentially for easier debugging
  run_package_tests "packages/mettagrid" "$BLUE"
  run_package_tests "agent" "$RED"
  run_package_tests "common" "$GREEN"
  run_package_tests "app_backend" "$YELLOW"
  run_package_tests "codebot" "$MAGENTA"
  run_package_tests "core" "$CYAN"
  run_package_tests "packages/cogames" "$WHITE"
fi

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

# Check results
OVERALL_FAILED=0
FAILED_PACKAGES=""

for package in agent common app_backend packages/mettagrid packages/cogames codebot core; do
  package_name=$(basename "$package")
  if [ -f "test-results/${package_name}.exit" ]; then
    EXIT_CODE=$(cat "test-results/${package_name}.exit")
    if [ "$EXIT_CODE" -ne 0 ]; then
      OVERALL_FAILED=1
      FAILED_PACKAGES="$FAILED_PACKAGES $package_name"
    fi
  else
    OVERALL_FAILED=1
    FAILED_PACKAGES="$FAILED_PACKAGES $package_name"
  fi
done

# Show summary
echo -e "\n${WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "\n${WHITE}🐌 TOP 10 SLOWEST TESTS${NC}"

# Combine all duration files and sort
for package in agent common app_backend packages/mettagrid packages/cogames codebot core; do
  package_name=$(basename "$package")
  if [ -f "test-results/${package_name}_durations.txt" ]; then
    # Add package name to each line
    sed "s/^/[${package_name}] /" "test-results/${package_name}_durations.txt"
  fi
done | sort -t' ' -k2 -rn | head -10 | nl -w2 -s'. '

echo -e "\n${WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ $OVERALL_FAILED -ne 0 ]; then
  echo -e "\n${RED}💥 Tests failed in:$FAILED_PACKAGES${NC}"
  echo -e "Total time: ${TOTAL_TIME}s"

  # Print detailed stacktraces for failed tests
  print_failed_test_stacktraces "$FAILED_PACKAGES"

  echo -e "\n${YELLOW}💡 Tips for debugging:${NC}"
  echo "  - Check individual test logs in test-results/*_raw.log"
  echo "  - Run with --sequential for easier debugging"
  echo "  - Run individual package tests: cd <package> && pytest -v"
  echo "  - Re-run specific failed tests: pytest -v <test_file>::<test_name>"
  exit 1
else
  echo -e "\n${GREEN}🎉 All tests passed in ${TOTAL_TIME}s!${NC}"
fi
