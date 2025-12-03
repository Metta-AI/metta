#!/bin/bash
# Master test runner for policy identification feature

set -e  # Exit on first failure

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     Policy Identification Feature Test Suite              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Track results
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_TESTS=()

run_test() {
    TEST_NAME=$1
    TEST_FILE=$2

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Running: $TEST_NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if uv run python "$TEST_FILE"; then
        ((TESTS_PASSED++))
        echo "✅ $TEST_NAME PASSED"
    else
        ((TESTS_FAILED++))
        FAILED_TESTS+=("$TEST_NAME")
        echo "❌ $TEST_NAME FAILED"
        return 1
    fi
}

# Run all tests
run_test "Basic Single Policy" "test_policy_basic.py" || true
run_test "Multi-Policy" "test_policy_multi.py" || true
run_test "Training Environment" "test_policy_training.py" || true
run_test "Replay Structure" "test_policy_replay_structure.py" || true

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    Test Summary                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Total tests: $((TESTS_PASSED + TESTS_FAILED))"
echo "Passed: $TESTS_PASSED ✅"
echo "Failed: $TESTS_FAILED ❌"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║            🎉 ALL TESTS PASSED! 🎉                        ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    exit 0
else
    echo "Failed tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║            ⚠️  SOME TESTS FAILED  ⚠️                      ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    exit 1
fi

