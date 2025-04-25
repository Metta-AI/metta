# Makefile for code formatting, linting, and testing
.PHONY: help format check-tools install-tools test benchmark clean check-test-tools install-test-tools check-bench-tools install-bench-tools all

# Default target when just running 'make'
help:
	@echo "Available targets:"
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
TEST_DIR = tests
BENCH_DIR = benchmarks
BUILD_DIR = build
BUILD_TEST_DIR = $(BUILD_DIR)/tests
BUILD_BENCH_DIR = $(BUILD_DIR)/benchmarks

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++14 -Wall -g -I$(SRC_DIR)

# Google Test settings - with detection for different install locations
GTEST_INCLUDE = $(shell pkg-config --cflags gtest 2>/dev/null || echo "-I/opt/homebrew/Cellar/googletest/1.16.0/include")
GTEST_LIBS = $(shell pkg-config --libs gtest_main 2>/dev/null || echo "-L/opt/homebrew/Cellar/googletest/1.16.0/lib -lgtest -lgtest_main -pthread")

# Add gtest includes to CXXFLAGS
CXXFLAGS += $(GTEST_INCLUDE)

# Google Benchmark settings
BENCHMARK_INCLUDE = $(shell pkg-config --cflags benchmark 2>/dev/null || echo "-I/usr/local/include -I/usr/include")
BENCHMARK_LIBS = $(shell pkg-config --libs benchmark 2>/dev/null || echo "-lbenchmark -lpthread")

# Add benchmark includes to CXXFLAGS when needed
BENCH_CXXFLAGS = $(CXXFLAGS) $(BENCHMARK_INCLUDE)

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
		(test -f /opt/homebrew/Cellar/googletest/1.16.0/lib/libgtest.dylib) || \
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

# Create build objects with preserved directory structure
define make-test-object
$(BUILD_TEST_DIR)/$(patsubst $(TEST_DIR)/%.cpp,%.o,$1): $1
	@mkdir -p $$(dir $$@)
	$$(CXX) $$(CXXFLAGS) -c $$< -o $$@
endef

$(foreach src,$(TEST_SOURCES),$(eval $(call make-test-object,$(src))))

TEST_OBJECTS := $(patsubst $(TEST_DIR)/%.cpp,$(BUILD_TEST_DIR)/%.o,$(TEST_SOURCES))
TEST_EXECUTABLES := $(patsubst $(BUILD_TEST_DIR)/%.o,$(BUILD_TEST_DIR)/%,$(TEST_OBJECTS))

# Compile individual test files
$(BUILD_TEST_DIR)/%.o: $(TEST_DIR)/%.cpp | $(BUILD_TEST_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link test executables
$(BUILD_TEST_DIR)/%: $(BUILD_TEST_DIR)/%.o
	$(CXX) $< -o $@ $(GTEST_LIBS)

# Run all tests
test: check-test-tools $(TEST_EXECUTABLES)
	@echo "Running all tests..."
	@for test in $(TEST_EXECUTABLES); do \
		echo "Running $$test"; \
		$$test --gtest_color=yes; \
	done

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

# Link benchmark executables
$(BUILD_BENCH_DIR)/%: $(BUILD_BENCH_DIR)/%.o
	$(CXX) $< -o $@ $(BENCHMARK_LIBS)

# Run all benchmarks
benchmark: check-bench-tools $(BENCH_EXECUTABLES)
	@echo "Running all benchmarks..."
	@for bench in $(BENCH_EXECUTABLES); do \
		echo "Running $$bench"; \
		$$bench; \
	done

#-----------------------
# Other targets
#-----------------------

# Clean build files
clean:
	@echo "Cleaning build files..."
	@rm -rf $(BUILD_DIR)

# Run format and test
all: format test
	@echo "All tasks completed."