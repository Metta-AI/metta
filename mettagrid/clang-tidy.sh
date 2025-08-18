#!/bin/bash
# run-clang-tidy.sh - Script to run clang-tidy with proper includes (macOS)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Running clang-tidy...${NC}"

# Check if clang-tidy is available
if ! command -v clang-tidy &> /dev/null; then
    echo -e "${RED}❌ clang-tidy not found! Please install it:${NC}"
    echo "   macOS: brew install llvm"
    echo "   Ubuntu: apt-get install clang-tidy"
    exit 1
fi

# Determine SDKROOT (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
    echo -e "${GREEN}Using macOS SDK at:${NC} $SDKROOT"
    EXTRA_ARGS="--extra-arg=-isysroot$SDKROOT"
else
    echo -e "${YELLOW}⚠️  Non-macOS system detected. No SDKROOT set.${NC}"
    EXTRA_ARGS=""
fi

# Make sure we have a compile_commands.json
if [ ! -f "build-debug/compile_commands.json" ]; then
    echo -e "${YELLOW}⚠️  No compile_commands.json found. Building first...${NC}"
    make build
fi

# Find all C++ source files
CPP_FILES=$(find src -name '*.cpp' -o -name '*.hpp')

# Run clang-tidy with compile commands
echo -e "${GREEN}Running clang-tidy with compile_commands.json...${NC}"
echo ""

# Initialize counters
TOTAL_FILES=0
FILES_WITH_WARNINGS=0
FILES_WITH_ERRORS=0
TOTAL_WARNINGS=0
TOTAL_ERRORS=0

# Run clang-tidy on each file
for file in $CPP_FILES; do
    TOTAL_FILES=$((TOTAL_FILES + 1))
    echo -e "${GREEN}[$TOTAL_FILES] Processing $file...${NC}"

    # Run clang-tidy and capture output
    OUTPUT=$(clang-tidy -p build-debug $EXTRA_ARGS "$file" 2>&1 || true)

    # Count warnings and errors
    FILE_WARNINGS=$(echo "$OUTPUT" | grep -c "warning:" || true)
    FILE_ERRORS=$(echo "$OUTPUT" | grep -c "error:" || true)

    if [ $FILE_WARNINGS -gt 0 ] || [ $FILE_ERRORS -gt 0 ]; then
        echo "$OUTPUT"
        echo ""

        if [ $FILE_WARNINGS -gt 0 ]; then
            FILES_WITH_WARNINGS=$((FILES_WITH_WARNINGS + 1))
            TOTAL_WARNINGS=$((TOTAL_WARNINGS + FILE_WARNINGS))
            echo -e "${YELLOW}  ⚠️  $FILE_WARNINGS warnings${NC}"
        fi

        if [ $FILE_ERRORS -gt 0 ]; then
            FILES_WITH_ERRORS=$((FILES_WITH_ERRORS + 1))
            TOTAL_ERRORS=$((TOTAL_ERRORS + FILE_ERRORS))
            echo -e "${RED}  ❌ $FILE_ERRORS errors${NC}"
        fi
        echo ""
    else
        echo -e "${GREEN}  ✅ Clean${NC}"
    fi
done

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Summary:${NC}"
echo -e "  Total files analyzed: $TOTAL_FILES"
echo -e "  Files with warnings: $FILES_WITH_WARNINGS"
echo -e "  Files with errors: $FILES_WITH_ERRORS"
echo -e "  Total warnings: ${YELLOW}$TOTAL_WARNINGS${NC}"
echo -e "  Total errors: ${RED}$TOTAL_ERRORS${NC}"

if [ $TOTAL_ERRORS -gt 0 ]; then
    echo -e "${RED}❌ Fix errors before committing!${NC}"
    exit 1
elif [ $TOTAL_WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Consider fixing warnings${NC}"
    exit 0
else
    echo -e "${GREEN}✅ All files are clean!${NC}"
    exit 0
fi
