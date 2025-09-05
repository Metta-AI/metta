#!/bin/bash

# Run all consolidated tests for Metta Tribal System

echo "================================"
echo "Running All Consolidated Tests"
echo "================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Track test results
PASSED=0
FAILED=0

# Function to run a test
run_test() {
    local test_file=$1
    local test_name=$2
    
    echo "Running: $test_name"
    echo "------------------------"
    
    # Check if source file exists
    if [ -f "$test_file" ]; then
        # Compile and run from source
        if nim c -r "$test_file" 2>/dev/null; then
            echo -e "${GREEN}✓ $test_name passed${NC}"
            ((PASSED++))
        else
            echo -e "${RED}✗ $test_name failed${NC}"
            ((FAILED++))
        fi
    else
        # Try to run pre-compiled executable
        local exe_name="${test_file%.nim}"
        if [ -f "$exe_name" ]; then
            if ./"$exe_name" > /dev/null 2>&1; then
                echo -e "${GREEN}✓ $test_name passed${NC}"
                ((PASSED++))
            else
                echo -e "${RED}✗ $test_name failed${NC}"
                ((FAILED++))
            fi
        else
            echo -e "${RED}✗ $test_name - test not found${NC}"
            ((FAILED++))
        fi
    fi
    echo ""
}

# Run each consolidated test
run_test "test_core_systems.nim" "Core Systems Tests"
run_test "test_ai_behavior.nim" "AI Behavior Tests"
run_test "test_clippy_improved.nim" "Improved Clippy Tests"
run_test "test_clippy_debug.nim" "Clippy Debug Tests"
run_test "test_clippy_wander.nim" "Clippy Wander Tests"

# Summary
echo "================================"
echo "Test Summary"
echo "================================"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed${NC}"
    exit 1
fi