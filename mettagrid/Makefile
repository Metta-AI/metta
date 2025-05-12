# Makefile for code formatting, linting, and testing 
.PHONY: help format check-tools install-tools test benchmark clean check-test-tools install-test-tools check-bench-tools install-bench-tools build build-clean all build-for-ci

# Default target when just running 'make'
help:
	@echo "Available targets:"
	@echo "  build             - Build mettagrid using the rebuild script"
	@echo "  build-clean       - Build mettagrid with clean option"
	@echo "  build-for-ci      - Build all source, test, and benchmark files without running tests (for CI)"
	@echo "  format            - Format C++/C files"
	@echo "  check-tools       - Check if required formatting tools are installed"
	@echo "  install-tools     - Install required formatting tools (macOS only)"
	@echo "  test              - Run all unit tests"
	@echo "  check-test-tools  - Check if required testing tools are installed"
	@echo "  install-test-tools - Install required testing tools"
	@echo "  benchmark         - Run all benchmarks"
	@echo "  check-bench-tools - Check if required benchmark tools are installed"
	@echo "  install-bench-tools - Install required benchmark tools"
	@echo "  clean             - Clean build and test files"
	@echo "  all               - Run format and test"

# Directories
SRC_DIR = mettagrid
THIRD_PARTY_DIR = third_party
TEST_DIR = tests
BENCH_DIR = benchmarks
BUILD_DIR = build
BUILD_SRC_DIR = $(BUILD_DIR)/mettagrid
BUILD_TEST_DIR = $(BUILD_DIR)/tests
BUILD_BENCH_DIR = $(BUILD_DIR)/benchmarks

DEVOPS_SCRIPTS_DIR = ../devops

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++23 -Wall -g -I$(SRC_DIR) -I$(THIRD_PARTY_DIR) -I$(TEST_DIR)

# Google Test settings - with detection for different install locations
GTEST_INCLUDE = $(shell pkg-config --cflags gtest 2>/dev/null || echo "-I/opt/homebrew/Cellar/googletest/1.17.0/include")
GTEST_LIBS = $(shell pkg-config --libs gtest_main 2>/dev/null || echo "-L/opt/homebrew/Cellar/googletest/1.17.0/lib -lgtest -lgtest_main -pthread")

# Add gtest includes to CXXFLAGS
CXXFLAGS += $(GTEST_INCLUDE)

# Google Benchmark settings
BENCHMARK_INCLUDE = $(shell pkg-config --cflags benchmark 2>/dev/null || echo "-I/usr/local/include -I/usr/include")
BENCHMARK_LIBS = $(shell pkg-config --libs benchmark 2>/dev/null || echo "-lbenchmark -lpthread")

# Add benchmark includes to CXXFLAGS when needed
BENCH_CXXFLAGS = $(CXXFLAGS) $(BENCHMARK_INCLUDE)

# Source files for mettagrid core library
SRC_SOURCES := $(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/**/*.cpp)
SRC_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_SRC_DIR)/%.o,$(SRC_SOURCES))

#-----------------------
# CI Build Target
#-----------------------

# Build all test and benchmark executables without running them (for CI environments)
build-for-ci: $(SRC_OBJECTS) $(TEST_EXECUTABLES) $(BENCH_EXECUTABLES)
	@echo "Built all source files, test executables, and benchmark executables"
	@echo "Source objects: $(words $(SRC_OBJECTS))"
	@echo "Test executables: $(words $(TEST_EXECUTABLES))"
	@echo "Benchmark executables: $(words $(BENCH_EXECUTABLES))"

#-----------------------
# Build
#-----------------------

# Build target that calls the build_mettagrid.sh script
build:
	@echo "Building mettagrid..."
	@if [ -f $(DEVOPS_SCRIPTS_DIR)/build_mettagrid.sh ]; then \
		$(DEVOPS_SCRIPTS_DIR)/build_mettagrid.sh; \
	else \
		echo "Error: build_mettagrid.sh script not found"; \
		echo "Expected path: $(DEVOPS_SCRIPTS_DIR)/build_mettagrid.sh"; \
		exit 1; \
	fi

# Build with clean option
build-clean:
	@echo "Building mettagrid with clean option..."
	@if [ -f $(DEVOPS_SCRIPTS_DIR)/build_mettagrid.sh ]; then \
		$(DEVOPS_SCRIPTS_DIR)/build_mettagrid.sh --clean; \
	else \
		echo "Error: build_mettagrid.sh script not found"; \
		echo "Expected path: $(DEVOPS_SCRIPTS_DIR)/build_mettagrid.sh"; \
		exit 1; \
	fi

#-----------------------
# Formatting
#-----------------------

# Check if the required formatting tools are installed
check-tools:
	@echo "Checking for required formatting tools..."
	@which clang-format >/dev/null 2>&1 || \
		{ echo "clang-format is not installed. On macOS use 'brew install clang-format'"; \
		  echo "On Linux use 'apt-get install clang-format'"; \
		  echo "Or run 'make install-tools' on macOS"; exit 1; }
	@echo "All required formatting tools are installed."

# Install formatting tools on macOS
install-tools:
	@echo "Installing required formatting tools..."
	@if [ "$(shell uname)" = "Darwin" ]; then \
		echo "Detected macOS. Installing tools via Homebrew..."; \
		brew install clang-format || echo "Failed to install clang-format. Please install manually."; \
	else \
		echo "This command only works on macOS. Please install tools manually:"; \
		echo "  - clang-format: apt-get install clang-format (Linux)"; \
	fi

# Format only C/C++ code and skip Cython files entirely
format: check-tools
	@echo "Formatting C/C++ code only (skipping all Cython files)..."
	@find . -type f \( -name "*.c" -o -name "*.h" -o -name "*.cpp" -o -name "*.hpp" \) \
		-not -path "*/\.*" -not -path "*/build/*" -not -path "*/venv/*" -not -path "*/dist/*" \
		-exec echo "Formatting {}" \; \
		-exec clang-format -style=file -i {} \;
	@echo "C/C++ formatting complete."
	@echo "Note: Cython files (.pyx, .pxd) were intentionally skipped to preserve their syntax."

#-----------------------
# Core Library Build
#-----------------------

# Create build directory for source files
$(BUILD_SRC_DIR):
	@mkdir -p $(BUILD_SRC_DIR)

# Compile individual source files
$(BUILD_SRC_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_SRC_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Build a static library from all source files
$(BUILD_DIR)/libmettagrid.a: $(SRC_OBJECTS)
	@mkdir -p $(dir $@)
	ar rcs $@ $^

#-----------------------
# Testing
#-----------------------

# Check if testing tools are installed
check-test-tools:
	@echo "Checking for required testing tools..."
	@which g++ >/dev/null 2>&1 || \
		{ echo "g++ compiler not found. On macOS use 'brew install gcc'"; \
		  echo "On Linux use 'apt-get install g++'"; exit 1; }
	@echo "Checking for Google Test library..."
	@(ldconfig -p 2>/dev/null | grep -q libgtest.so) || \
		(test -f /usr/local/lib/libgtest.a) || \
		(test -f /usr/local/lib/libgtest.dylib) || \
		(test -f /opt/homebrew/Cellar/googletest/1.17.0/lib/libgtest.a) || \
		(test -f /opt/homebrew/Cellar/googletest/1.17.0/lib/libgtest.dylib) || \
		(pkg-config --exists gtest 2>/dev/null) || \
		{ echo "Google Test library not found. Run 'make install-test-tools' to install."; exit 1; }
	@echo "All required testing tools are installed."

# Install testing tools
install-test-tools:
	@echo "Installing required testing tools..."
	@if [ "$(shell uname)" = "Darwin" ]; then \
		echo "Detected macOS. Installing tools via Homebrew..."; \
		brew install googletest || echo "Failed to install googletest. Please install manually."; \
	elif [ -f /etc/debian_version ]; then \
		echo "Detected Debian/Ubuntu. Installing tools via apt..."; \
		sudo apt-get update && sudo apt-get install -y libgtest-dev cmake; \
		cd /usr/src/gtest && sudo cmake CMakeLists.txt && sudo make && \
		sudo cp lib/*.a /usr/lib || \
		echo "Failed to build googletest. Please install manually."; \
	else \
		echo "Unsupported OS. Please install Google Test manually:"; \
		echo "  - See https://github.com/google/googletest for instructions."; \
	fi

# Create build directory for tests
$(BUILD_TEST_DIR):
	@mkdir -p $(BUILD_TEST_DIR)

# Find all test source files
TEST_SOURCES := $(wildcard $(TEST_DIR)/*.cpp $(TEST_DIR)/**/*.cpp)
TEST_OBJECTS := $(patsubst $(TEST_DIR)/%.cpp,$(BUILD_TEST_DIR)/%.o,$(TEST_SOURCES))
TEST_EXECUTABLES := $(patsubst $(BUILD_TEST_DIR)/%.o,$(BUILD_TEST_DIR)/%,$(TEST_OBJECTS))

# Compile individual test files
$(BUILD_TEST_DIR)/%.o: $(TEST_DIR)/%.cpp | $(BUILD_TEST_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link test executables with the mettagrid library
$(BUILD_TEST_DIR)/%: $(BUILD_TEST_DIR)/%.o $(SRC_OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $^ -o $@ $(GTEST_LIBS)

# Run all tests
test: check-test-tools $(SRC_OBJECTS) $(TEST_EXECUTABLES)
	@echo "Running all tests..."
	@for test in $(TEST_EXECUTABLES); do \
		echo "Running $$test"; \
		$$test --gtest_color=yes; \
	done

# Test a specific test file
test-%: check-test-tools $(BUILD_TEST_DIR)/%
	@echo "Running test $*..."
	$(BUILD_TEST_DIR)/$* --gtest_color=yes

#-----------------------
# Benchmarking
#-----------------------

# Check if benchmark tools are installed
check-bench-tools:
	@echo "Checking for required benchmark tools..."
	@which g++ >/dev/null 2>&1 || \
		{ echo "g++ compiler not found. On macOS use 'brew install gcc'"; \
		  echo "On Linux use 'apt-get install g++'"; exit 1; }
	@echo "Checking for Google Benchmark library..."
	@(ldconfig -p 2>/dev/null | grep -q libbenchmark.so) || \
		(test -f /usr/local/lib/libbenchmark.a) || \
		(test -f /usr/local/lib/libbenchmark.dylib) || \
		(pkg-config --exists benchmark 2>/dev/null) || \
		{ echo "Google Benchmark library not found. Run 'make install-bench-tools' to install."; exit 1; }
	@echo "All required benchmark tools are installed."

# Install benchmark tools
install-bench-tools:
	@echo "Installing required benchmark tools..."
	@if [ "$(shell uname)" = "Darwin" ]; then \
		echo "Detected macOS. Installing tools via Homebrew..."; \
		brew install google-benchmark || echo "Failed to install google-benchmark. Please install manually."; \
	elif [ -f /etc/debian_version ]; then \
		echo "Detected Debian/Ubuntu. Installing tools via apt..."; \
		sudo apt-get update && sudo apt-get install -y libbenchmark-dev; \
	else \
		echo "Unsupported OS. Please install Google Benchmark manually:"; \
		echo "  - See https://github.com/google/benchmark for instructions."; \
	fi

# Create build directory for benchmarks
$(BUILD_BENCH_DIR):
	@mkdir -p $(BUILD_BENCH_DIR)

# Find all benchmark source files
BENCH_SOURCES := $(wildcard $(BENCH_DIR)/*.cpp $(BENCH_DIR)/**/*.cpp)
BENCH_OBJECTS := $(patsubst $(BENCH_DIR)/%.cpp,$(BUILD_BENCH_DIR)/%.o,$(BENCH_SOURCES))
BENCH_EXECUTABLES := $(patsubst $(BUILD_BENCH_DIR)/%.o,$(BUILD_BENCH_DIR)/%,$(BENCH_OBJECTS))

# Compile individual benchmark files
$(BUILD_BENCH_DIR)/%.o: $(BENCH_DIR)/%.cpp | $(BUILD_BENCH_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(BENCH_CXXFLAGS) -c $< -o $@

# Link benchmark executables with the mettagrid library
$(BUILD_BENCH_DIR)/%: $(BUILD_BENCH_DIR)/%.o $(SRC_OBJECTS)
	$(CXX) $^ -o $@ $(BENCHMARK_LIBS)

# Run all benchmarks
benchmark: check-bench-tools $(SRC_OBJECTS) $(BENCH_EXECUTABLES)
	@echo "Running all benchmarks..."
	@for bench in $(BENCH_EXECUTABLES); do \
		echo "Running $$bench"; \
		$$bench; \
	done

# Add this new target for JSON benchmark output
bench-json: check-bench-tools $(SRC_OBJECTS) $(BENCH_EXECUTABLES)
	@echo "Running all benchmarks with JSON output..."
	@mkdir -p benchmark_output
	@for bench in $(BENCH_EXECUTABLES); do \
		echo "Running $$bench with JSON output..."; \
		$$bench --benchmark_format=json > benchmark_output/$$(basename $$bench).json || \
		echo "Error running $$bench with JSON format"; \
	done
	@echo "JSON outputs created in benchmark_output directory"

#-----------------------
# Other targets
#-----------------------

# Clean build files
clean:
	@echo "Cleaning build files..."
	@rm -rf $(BUILD_DIR)
	@find mettagrid -name "*.so" -type f -delete
	@echo "Removed .so files from mettagrid directory"

# Run format and test
all: format test
	@echo "All tasks completed."
