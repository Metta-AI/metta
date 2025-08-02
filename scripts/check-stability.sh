#!/bin/bash
# Script to run comprehensive stability tests locally
# This runs the same tests as the stable tag advancement workflow

set -euo pipefail

echo "🔍 Metta Stability Check"
echo "========================"
echo ""

# Parse arguments
RUN_BENCHMARKS=${1:-true}
if [[ "$1" == "--skip-benchmarks" ]]; then
    RUN_BENCHMARKS=false
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall status
ALL_PASSED=true
RESULTS=""

# Function to run a test section
run_test_section() {
    local name=$1
    local command=$2
    
    echo -e "${YELLOW}Running: $name${NC}"
    if eval "$command"; then
        echo -e "${GREEN}✅ $name passed${NC}"
        RESULTS="${RESULTS}✅ $name passed\n"
    else
        echo -e "${RED}❌ $name failed${NC}"
        RESULTS="${RESULTS}❌ $name failed\n"
        ALL_PASSED=false
    fi
    echo ""
}

# 1. Run core tests
run_test_section "Core Tests" \
    "pytest tests/ -m 'not expensive' --maxfail=10 -v"

# 2. Run expensive/integration tests
run_test_section "Integration Tests" \
    "pytest tests/ -m 'expensive or integration' --maxfail=5 -v"

# 3. Run performance benchmarks (if not skipped)
if [[ "$RUN_BENCHMARKS" == "true" ]]; then
    run_test_section "Python Benchmarks" \
        "pytest --benchmark-only --benchmark-json=benchmark_results.json"
    
    # C++ benchmarks
    if [ -d "mettagrid" ]; then
        run_test_section "C++ Benchmarks" \
            "cd mettagrid && cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release && cmake --build build-release --target all && cd .."
    fi
fi

# 4. Validate configs
echo -e "${YELLOW}Validating configs...${NC}"
python3 << 'EOF'
import os
import yaml
from pathlib import Path

errors = []
for config_file in Path('configs').rglob('*.yaml'):
    try:
        with open(config_file) as f:
            yaml.safe_load(f)
    except Exception as e:
        errors.append(f'{config_file}: {e}')

if errors:
    print('Config validation errors:')
    for error in errors:
        print(f'  ❌ {error}')
    exit(1)
else:
    print('✅ All configs validated successfully')
EOF

if [ $? -eq 0 ]; then
    RESULTS="${RESULTS}✅ Config validation passed\n"
else
    RESULTS="${RESULTS}❌ Config validation failed\n"
    ALL_PASSED=false
fi

# 5. Test example training script
run_test_section "Example Training Script" \
    "python tools/train.py trainer.total_env_steps=1000 trainer.eval_interval=500 hardware=github wandb.mode=disabled"

# Summary
echo ""
echo "========================================"
echo "📊 Stability Check Summary"
echo "========================================"
echo -e "$RESULTS"

if [[ "$ALL_PASSED" == "true" ]]; then
    echo -e "${GREEN}✅ All stability checks passed!${NC}"
    echo "This commit is ready for stable tag."
    exit 0
else
    echo -e "${RED}❌ Some stability checks failed.${NC}"
    echo "This commit should not be tagged as stable."
    exit 1
fi