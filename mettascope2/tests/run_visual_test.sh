#!/bin/bash
# Quick visual test script for heatmap
echo "Compiling and running visual test..."
cd tests
nim c --hints:off test_clippy_plague.nim 2>/dev/null && ./test_clippy_plague | grep -E "(Starting with|Final|✓|✗)"