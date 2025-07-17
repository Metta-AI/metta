#!/bin/bash
# generate_coverage.sh - Generate C++ coverage report

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting C++ coverage generation...${NC}"

# Configuration
BUILD_DIR="build-coverage"
COVERAGE_FILE="coverage.info"

# Clean and create build directory
echo -e "${YELLOW}Setting up coverage build...${NC}"
rm -rf ${BUILD_DIR}
cmake --preset coverage

# Build the project
echo -e "${YELLOW}Building project with coverage...${NC}"
cmake --build ${BUILD_DIR} -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Run tests
echo -e "${YELLOW}Running tests...${NC}"
cd ${BUILD_DIR}
ctest --output-on-failure || true  # Continue even if some tests fail
cd ..

# Detect which coverage format we have
echo -e "${YELLOW}Detecting coverage format...${NC}"
GCDA_FILES=$(find ${BUILD_DIR} -name "*.gcda" 2>/dev/null | head -1)
PROFRAW_FILES=$(find ${BUILD_DIR} -name "*.profraw" 2>/dev/null | head -1)

if [ -n "$GCDA_FILES" ]; then
    # GCC/gcov format detected
    echo -e "${GREEN}Found GCC coverage data (.gcda files)${NC}"

    if ! command -v lcov &> /dev/null; then
        echo -e "${RED}Error: lcov is required for GCC coverage data${NC}"
        echo "Install lcov:"
        echo "  Ubuntu/Debian: sudo apt-get install lcov"
        echo "  macOS: brew install lcov"
        echo "  RHEL/CentOS: sudo yum install lcov"
        exit 1
    fi

    echo -e "${GREEN}Using lcov for coverage...${NC}"

    # Capture coverage data with inline config to suppress warnings
    lcov --capture \
         --directory ${BUILD_DIR} \
         --output-file ${BUILD_DIR}/${COVERAGE_FILE} \
         --rc branch_coverage=1 \
         --ignore-errors inconsistent,inconsistent,format,format,unused,unused,unsupported \
         --quiet
         # somehow, these doubled flags are correct syntax!

    # Remove unwanted files from coverage
    # Exclude external dependencies, test files, and system headers
    lcov --remove ${BUILD_DIR}/${COVERAGE_FILE} \
         '*/usr/*' \
         '/Applications/*' \
         '*/site-packages/*' \
         '*/include/*' \
         '*/tests/*.cpp' \
         '*/benchmarks/*.cpp' \
         --output-file ${BUILD_DIR}/${COVERAGE_FILE} \
         --rc branch_coverage=1 \
         --ignore-errors empty,unused,unused,inconsistent,format,format \
         --quiet

    # Display summary
    echo -e "${GREEN}Coverage summary:${NC}"
    lcov --list ${BUILD_DIR}/${COVERAGE_FILE} 2>/dev/null

elif [ -n "$PROFRAW_FILES" ]; then
    # LLVM/Clang format detected
    echo -e "${GREEN}Found LLVM coverage data (.profraw files)${NC}"

    if ! command -v llvm-profdata &> /dev/null || ! command -v llvm-cov &> /dev/null; then
        echo -e "${RED}Error: llvm-profdata and llvm-cov are required for LLVM coverage data${NC}"
        echo "Install LLVM tools:"
        echo "  Ubuntu/Debian: sudo apt-get install llvm"
        echo "  macOS: Should be included with Xcode"
        exit 1
    fi

    echo -e "${GREEN}Using llvm-cov for coverage...${NC}"

    # Find all test executables
    TEST_EXECUTABLES=$(find ${BUILD_DIR} -name "test_*" -type f -perm +111 2>/dev/null)

    # If no test executables found with permission check, try without
    if [ -z "$TEST_EXECUTABLES" ]; then
        TEST_EXECUTABLES=$(find ${BUILD_DIR} -name "test_*" -type f 2>/dev/null | grep -v ".cpp" | grep -v ".o")
    fi

    # Check if we found any executables
    if [ -z "$TEST_EXECUTABLES" ]; then
        echo -e "${RED}Error: No test executables found in ${BUILD_DIR}${NC}"
        echo -e "${YELLOW}Looking for test_* files:${NC}"
        find ${BUILD_DIR} -name "test_*" -type f | head -20
        exit 1
    fi

    echo -e "${GREEN}Found test executables:${NC}"
    echo "$TEST_EXECUTABLES"

    # Check for profraw files first
    PROFRAW_FILES=$(find ${BUILD_DIR} -name "*.profraw" 2>/dev/null)

    if [ -z "$PROFRAW_FILES" ]; then
        echo -e "${RED}Error: No .profraw files found. Make sure tests were run with coverage enabled.${NC}"
        exit 1
    fi

    # Merge all profraw files
    llvm-profdata merge -sparse ${PROFRAW_FILES} -o ${BUILD_DIR}/coverage.profdata

    # Generate lcov format with better filtering
    llvm-cov export ${TEST_EXECUTABLES} \
        -instr-profile=${BUILD_DIR}/coverage.profdata \
        -format=lcov \
        -ignore-filename-regex='(tests|benchmarks|googletest|googlebenchmark|/usr/|/Applications/)' \
        > ${BUILD_DIR}/${COVERAGE_FILE}

    # Display summary
    echo -e "${GREEN}Coverage summary:${NC}"
    llvm-cov report ${TEST_EXECUTABLES} \
        -instr-profile=${BUILD_DIR}/coverage.profdata \
        -ignore-filename-regex='(tests|benchmarks|googletest|googlebenchmark|/usr/|/Applications/)'

else
    echo -e "${RED}Error: No coverage data found!${NC}"
    echo -e "${YELLOW}Looking for any coverage files...${NC}"
    echo "GCC coverage files (.gcda):"
    find ${BUILD_DIR} -name "*.gcda" 2>/dev/null | head -5 || echo "  None found"
    echo "LLVM coverage files (.profraw):"
    find ${BUILD_DIR} -name "*.profraw" 2>/dev/null | head -5 || echo "  None found"
    echo ""
    echo "Make sure your build was configured with coverage enabled and tests were run."
    exit 1
fi

# Verify coverage file was created
if [ ! -f "${BUILD_DIR}/${COVERAGE_FILE}" ]; then
    echo -e "${RED}Error: Coverage file was not created${NC}"
    exit 1
fi

# Check if coverage file has content
if [ ! -s "${BUILD_DIR}/${COVERAGE_FILE}" ]; then
    echo -e "${RED}Error: Coverage file is empty${NC}"
    exit 1
fi

echo -e "${GREEN}Coverage generation complete!${NC}"
echo -e "Coverage report: ${BUILD_DIR}/${COVERAGE_FILE}"
echo -e "File size: $(ls -lh ${BUILD_DIR}/${COVERAGE_FILE} | awk '{print $5}')"
echo -e "\n${YELLOW}Upload to Codecov with:${NC}"
echo -e "  codecov -f ${BUILD_DIR}/${COVERAGE_FILE}"
echo -e "  # or"
echo -e "  bash <(curl -s https://codecov.io/bash) -f ${BUILD_DIR}/${COVERAGE_FILE}"
