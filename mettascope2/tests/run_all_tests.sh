#!/bin/bash
# Run all Nim tests in the tests directory

echo "Running MettaScope 2 Tests"
echo "=========================="
echo

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Counter for tests
total=0
passed=0
failed=0

# Run each test file
for test_file in test_*.nim; do
    if [ -f "$test_file" ]; then
        echo "Running $test_file..."
        ((total++))
        
        # Compile and run the test
        if nim c -r --hints:off "$test_file" > /tmp/test_output_$$.txt 2>&1; then
            echo -e "${GREEN}✓${NC} $test_file passed"
            ((passed++))
        else
            echo -e "${RED}✗${NC} $test_file failed"
            echo "  Error output:"
            cat /tmp/test_output_$$.txt | head -10
            ((failed++))
        fi
        
        # Clean up compiled file
        base_name="${test_file%.nim}"
        rm -f "$base_name"
        rm -f /tmp/test_output_$$.txt
        echo
    fi
done

# Summary
echo "=========================="
echo "Test Summary:"
echo "  Total: $total"
echo -e "  ${GREEN}Passed: $passed${NC}"
if [ $failed -gt 0 ]; then
    echo -e "  ${RED}Failed: $failed${NC}"
else
    echo -e "  Failed: 0"
fi

# Exit with error if any tests failed
if [ $failed -gt 0 ]; then
    exit 1
fi