#!/bin/bash
# Test runner for MettaScope2

echo "================================"
echo "MettaScope2 Test Suite"
echo "================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Track results
PASSED=0
FAILED=0

# Function to run a test
run_test() {
    local test_file=$1
    local test_name=$2
    
    echo "Running: $test_name"
    echo "--------------------------------"
    
    if nim r --hints:off $test_file 2>/dev/null; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAILED${NC}"
        ((FAILED++))
    fi
    echo ""
}

# Run the test suites
run_test "tests/test_core_systems.nim" "Core Systems"
run_test "tests/test_ai_behavior.nim" "AI Behavior"
run_test "tests/test_diagonal_movement_fix.nim" "Diagonal Movement Fix"

# Summary
echo "================================"
echo "Test Summary"
echo "================================"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed${NC}"
    exit 1
fi