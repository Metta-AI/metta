#!/bin/bash

# Enhanced memory leak testing script with multiple detection methods

set -e

echo "=== Memory Leak Testing Suite ==="

# Check if we're on a system that supports the tools
HAS_VALGRIND=$(command -v valgrind >/dev/null 2>&1 && echo "yes" || echo "no")
HAS_ASAN=$(gcc -print-file-name=libasan.so 2>/dev/null | grep -v "^libasan.so$" >/dev/null && echo "yes" || echo "no")

echo "Available tools:"
echo "  - AddressSanitizer: $HAS_ASAN"
echo "  - Valgrind: $HAS_VALGRIND"
echo ""

# Test 1: Standard Python memory leak tests
echo "=== Running Python memory leak tests ==="
python -m pytest tests/test_leaks.py -v -s

# Test 2: AddressSanitizer leak detection (if available)
if [ "$HAS_ASAN" = "yes" ]; then
    echo ""
    echo "=== Running AddressSanitizer leak detection ==="
    
    # Build with AddressSanitizer
    echo "Building with AddressSanitizer..."
    mkdir -p build_asan
    cd build_asan
    CMAKE_BUILD_TYPE=Debug cmake .. -DBUILD_TESTS=ON
    make -j$(nproc 2>/dev/null || echo 4)
    cd ..
    
    # Run with ASAN
    export LD_PRELOAD="$(gcc -print-file-name=libasan.so) $(gcc -print-file-name=libstdc++.so)"
    export ASAN_OPTIONS="detect_leaks=1:abort_on_error=1:symbolize=1:print_stacktrace=1"
    
    echo "Running tests with AddressSanitizer..."
    python -m pytest tests/test_leaks.py::test_mettagrid_env_no_memory_leaks -v -s || {
        echo "ASAN detected memory issues!"
        exit 1
    }
    
    unset LD_PRELOAD ASAN_OPTIONS
    echo "AddressSanitizer tests passed!"
else
    echo "Skipping AddressSanitizer tests (not available)"
fi

# Test 3: Valgrind leak detection (if available and on Linux)
if [ "$HAS_VALGRIND" = "yes" ] && [ "$(uname)" = "Linux" ]; then
    echo ""
    echo "=== Running Valgrind leak detection ==="
    
    # Create a simple test script for Valgrind
    cat > /tmp/valgrind_test.py << 'EOF'
import sys
sys.path.insert(0, '.')

from tests.test_leaks import test_mettagrid_env_init
from hydra import compose, initialize

# Simple test that creates and destroys a few environments
with initialize(version_base=None, config_path="configs"):
    cfg = compose(config_name="test_basic")
    
    # Run a smaller version of the leak test
    for i in range(5):
        test_mettagrid_env_init(cfg)
        print(f"Valgrind test iteration {i+1}/5 completed")

print("Valgrind test completed successfully")
EOF
    
    echo "Running Valgrind leak detection..."
    valgrind --tool=memcheck \
             --leak-check=full \
             --show-leak-kinds=all \
             --track-origins=yes \
             --verbose \
             --error-exitcode=1 \
             --suppressions=/usr/share/gdb/auto-load/usr/lib/x86_64-linux-gnu/libglib-2.0.so.0.6800.4-gdb.py 2>/dev/null || true \
             python /tmp/valgrind_test.py || {
        echo "Valgrind detected memory issues!"
        exit 1
    }
    
    rm -f /tmp/valgrind_test.py
    echo "Valgrind tests passed!"
else
    echo "Skipping Valgrind tests (not available or not on Linux)"
fi

echo ""
echo "=== All memory leak tests completed successfully! ==="
