# Makefile for code formatting, linting, and testing 
.PHONY: help format check-tools install-tools test benchmark clean check-test-tools install-test-tools check-bench-tools install-bench-tools build build-clean all build-for-ci check-pybind11 install-pybind11

# Enable/disable Pybind11 (set to 1 to enable, 0 to disable)
USE_PYBIND ?= 0

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
	@echo "  test p=name       - Run tests matching the specified pattern p"
	@echo "  check-test-tools  - Check if required testing tools are installed"
	@echo "  install-test-tools - Install required testing tools"
	@echo "  benchmark         - Run all benchmarks"
	@echo "  check-bench-tools - Check if required benchmark tools are installed"
	@echo "  install-bench-tools - Install required benchmark tools"
	@echo "  check-pybind11     - Check if pybind11 is installed (only if USE_PYBIND=1)"
	@echo "  install-pybind11   - Install pybind11 (only if USE_PYBIND=1)"
	@echo "  clean             - Clean build and test files"
	@echo "  all               - Run format and test"
	@echo ""
	@echo "Configuration:"
	@echo "  USE_PYBIND=$(USE_PYBIND) - Set to 1 to enable Pybind11 (make USE_PYBIND=1 <target>)"

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

# Get the project root directory (where the Makefile is located)
PROJECT_ROOT := $(shell pwd)

# Get Python paths directly using UV instead of virtual environment
PYTHON_VERSION := $(shell python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.11")
PYTHON_INCLUDE := $(shell uv run --active python -c "import sysconfig; print(sysconfig.get_path('include'))" 2>/dev/null)
PYTHON_STDLIB := $(shell uv run --active python -c "import sysconfig; print(sysconfig.get_path('stdlib'))" 2>/dev/null)
PYTHON_DYNLOAD := $(PYTHON_STDLIB)/lib-dynload
PYTHON_SITE_PACKAGES := $(shell uv run --active python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
PYTHON_LIB_DIR := $(shell uv run --active python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))" 2>/dev/null)
PYBIND11_INCLUDES := $(shell uv run --active python -m pybind11 --includes 2>/dev/null)
PYBIND11_INCLUDE := $(shell echo "$(PYBIND11_INCLUDES)" | grep -o '\-I[^ ]*pybind11[^ ]*' | head -1 | sed 's/-I//')
UV_ROOT := $(shell dirname $(PYTHON_STDLIB))

# Print the paths for debugging
$(info UV_ROOT: $(UV_ROOT))
$(info PYTHON_STDLIB: $(PYTHON_STDLIB))
$(info PYTHON_INCLUDE: $(PYTHON_INCLUDE))
$(info PYTHON_SITE_PACKAGES: $(PYTHON_SITE_PACKAGES))
$(info PYTHON_LIB_DIR: $(PYTHON_LIB_DIR))
$(info PYBIND11_INCLUDE: $(PYBIND11_INCLUDE))

# Update the Python library definitions
PYTHON_LIBS = -L$(PYTHON_LIB_DIR) -lpython$(PYTHON_VERSION)

# Define CXXFLAGS with all the necessary includes
CXXFLAGS = -std=c++23 -Wall -g -I$(SRC_DIR) -I$(THIRD_PARTY_DIR) -I$(TEST_DIR) -I$(PYTHON_INCLUDE) -I$(PYBIND11_INCLUDE) $(NUMPY_INCLUDE)
# Add Python path definitions to CXXFLAGS
CXXFLAGS += -DPYTHON_HOME=\"$(UV_ROOT)\" -DPYTHON_STDLIB=\"$(PYTHON_STDLIB)\"

# Conditionally add Pybind11 configurations
ifeq ($(USE_PYBIND),1)
    PYBIND11_INCLUDE = $(shell python -m pybind11 --includes 2>/dev/null || echo "")
    PYBIND11_LIBS = $(shell python -m pybind11 --includes 2>/dev/null || echo "")
    NUMPY_INCLUDE = $(shell python -c "import numpy; print(numpy.get_include())" 2>/dev/null || echo "")
    PYTHON_INCLUDE = $(shell python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())" 2>/dev/null || echo "")
    PYTHON_VERSION = $(shell python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
    PYTHON_LIB_DIR = $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))" 2>/dev/null || echo "")
    PYTHON_LIBS = -L$(PYTHON_LIB_DIR) -lpython$(PYTHON_VERSION)
    CXXFLAGS += -I$(EIGEN_INCLUDE) -I$(PYBIND11_INCLUDE) -I$(NUMPY_INCLUDE) -I$(PYTHON_INCLUDE) -DUSE_PYBIND
    
    # Add RPATH settings for macOS
    ifeq ($(shell uname), Darwin)
        RPATH_FLAGS = -Wl,-rpath,$(PYTHON_LIB_DIR)
    else
        RPATH_FLAGS =
    endif
else
    RPATH_FLAGS =
endif

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
ifeq ($(USE_PYBIND),1)
    # Include all source files when using Pybind11
    SRC_SOURCES := $(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/**/*.cpp)
else
    # Exclude Pybind11-specific files when not using Pybind11
    SRC_SOURCES := $(filter-out $(SRC_DIR)/mettagrid/mettagrid_c.cpp $(SRC_DIR)/mettagrid/mettagrid_c.hpp, $(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/**/*.cpp))
endif

# Create the list of object files from source files
SRC_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_SRC_DIR)/%.o,$(SRC_SOURCES))

# Report build mode at the beginning of each target execution
define report_pybind_mode
	@echo "----------------------------------------"
	@echo "Build Mode: Pybind11 support is $(if $(filter 1,$(USE_PYBIND)),ENABLED,DISABLED)"
	@echo "----------------------------------------"
endef

SRC_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_SRC_DIR)/%.o,$(SRC_SOURCES))

#-----------------------
# CI Build Target
#-----------------------

# Build all test and benchmark executables without running them (for CI environments)
ifeq ($(USE_PYBIND),1)
build-for-ci: check-pybind11 clean-build-dir $(SRC_OBJECTS) $(TEST_EXECUTABLES) $(BENCH_EXECUTABLES)
else
build-for-ci: clean-build-dir $(SRC_OBJECTS) $(TEST_EXECUTABLES) $(BENCH_EXECUTABLES)
endif
	@echo "Built all source files, test executables, and benchmark executables"
	@echo "Source objects: $(words $(SRC_OBJECTS))"
	@echo "Test executables: $(words $(TEST_EXECUTABLES))"
	@echo "Benchmark executables: $(words $(BENCH_EXECUTABLES))"
	@echo "Pybind11 support: $(if $(filter 1,$(USE_PYBIND)),enabled,disabled)"

#-----------------------
# Build
#-----------------------

# Build target that calls the build_mettagrid.sh script
ifeq ($(USE_PYBIND),1)
build: check-pybind11 clean-build-dir
else
build: clean-build-dir
endif
	$(call report_pybind_mode)
	@echo "Building mettagrid..."
	@if [ -f $(DEVOPS_SCRIPTS_DIR)/build_mettagrid.sh ]; then \
		$(DEVOPS_SCRIPTS_DIR)/build_mettagrid.sh $(if $(filter 0,$(USE_PYBIND)),--no-pybind,); \
	else \
		echo "Error: build_mettagrid.sh script not found"; \
		echo "Expected path: $(DEVOPS_SCRIPTS_DIR)/build_mettagrid.sh"; \
		exit 1; \
	fi

# Build with clean option
build-clean: clean-build-dir
	@echo "Building mettagrid with clean option..."
	@if [ -f $(DEVOPS_SCRIPTS_DIR)/build_mettagrid.sh ]; then \
		$(DEVOPS_SCRIPTS_DIR)/build_mettagrid.sh --clean $(if $(filter 0,$(USE_PYBIND)),--no-pybind,); \
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

# Clean the build directory first to ensure no stale objects remain
clean-build-dir:
	@echo "Cleaning build directory..."
	@rm -rf $(BUILD_DIR)
	@mkdir -p $(BUILD_SRC_DIR)
	@mkdir -p $(BUILD_TEST_DIR)
	@mkdir -p $(BUILD_BENCH_DIR)

# Create build directory for source files
$(BUILD_SRC_DIR):
	@mkdir -p $(BUILD_SRC_DIR)

# Compile individual source files
$(BUILD_SRC_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_SRC_DIR)
	@mkdir -p $(dir $@)
ifeq ($(USE_PYBIND),1)
	$(CXX) $(CXXFLAGS) -c $< -o $@
else
	@if echo "$<" | grep -q "mettagrid_c.cpp\|mettagrid_c.hpp"; then \
		echo "Skipping Pybind11 file $< (USE_PYBIND=0)"; \
	else \
		$(CXX) $(CXXFLAGS) -c $< -o $@; \
	fi
endif

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
ifeq ($(USE_PYBIND),1)
    # Include all test files when Pybind11 is enabled
    TEST_SOURCES := $(wildcard $(TEST_DIR)/*.cpp $(TEST_DIR)/**/*.cpp)
else
    # Exclude Pybind11-specific test files when Pybind11 is disabled
    TEST_SOURCES := $(filter-out $(TEST_DIR)/pybind_mettagrid_test.cpp, $(wildcard $(TEST_DIR)/*.cpp $(TEST_DIR)/**/*.cpp))
endif
TEST_OBJECTS := $(patsubst $(TEST_DIR)/%.cpp,$(BUILD_TEST_DIR)/%.o,$(TEST_SOURCES))
TEST_EXECUTABLES := $(patsubst $(BUILD_TEST_DIR)/%.o,$(BUILD_TEST_DIR)/%,$(TEST_OBJECTS))

# Compile individual test files
$(BUILD_TEST_DIR)/%.o: $(TEST_DIR)/%.cpp | $(BUILD_TEST_DIR)
	@mkdir -p $(dir $@)
ifeq ($(USE_PYBIND),0)
	@if echo "$<" | grep -q "pybind_mettagrid_test.cpp"; then \
		echo "Skipping Pybind11 test file $< (USE_PYBIND=0)"; \
	else \
		$(CXX) $(CXXFLAGS) -c $< -o $@; \
	fi
else
	$(CXX) $(CXXFLAGS) -c $< -o $@
endif

# Link test executables with the mettagrid library
$(BUILD_TEST_DIR)/%: $(BUILD_TEST_DIR)/%.o $(SRC_OBJECTS)
	@mkdir -p $(dir $@)
	@echo "Linking test executable $@..."
ifeq ($(USE_PYBIND),1)
	$(CXX) $< $(SRC_OBJECTS) -o $@ $(GTEST_LIBS) $(PYTHON_LIBS) $(PYBIND11_LIBS) $(RPATH_FLAGS)
else
	# When pybind is disabled, don't try to link non-existent mettagrid_c.o
	$(CXX) $< $(filter-out %mettagrid_c.o,$(SRC_OBJECTS)) -o $@ $(GTEST_LIBS)
endif

# Run all tests or tests matching a pattern
ifeq ($(USE_PYBIND),1)
test: check-test-tools check-pybind11 clean-build-dir $(SRC_OBJECTS) $(TEST_EXECUTABLES)
else
test: check-test-tools clean-build-dir $(SRC_OBJECTS) $(TEST_EXECUTABLES)
endif
	$(call report_pybind_mode)
	@echo "Running all tests with Python environment..."
	@for f in $(TEST_EXECUTABLES); do \
		echo "Running $$f"; \
		PYTHONHOME="$(UV_ROOT)" \
		PYTHONPATH="$(PYTHON_STDLIB):$(PYTHON_DYNLOAD):$(PYTHON_SITE_PACKAGES)" \
		LD_LIBRARY_PATH="$(PYTHON_LIB_DIR):$$LD_LIBRARY_PATH" \
		DYLD_LIBRARY_PATH="$(PYTHON_LIB_DIR):$$DYLD_LIBRARY_PATH" \
		"$$f" --gtest_color=yes || exit 1; \
	done

# Test a specific test file - modified to use quotes
test-%: check-test-tools $(BUILD_TEST_DIR)/%
	@echo "Running test $*..."
	PYTHONHOME="$(UV_ROOT)" \
	PYTHONPATH="$(PYTHON_STDLIB):$(PYTHON_DYNLOAD):$(PYTHON_SITE_PACKAGES)" \
	LD_LIBRARY_PATH="$(PYTHON_LIB_DIR):$$LD_LIBRARY_PATH" \
	DYLD_LIBRARY_PATH="$(PYTHON_LIB_DIR):$$DYLD_LIBRARY_PATH" \
	"$(BUILD_TEST_DIR)/$*" --gtest_color=yes

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
	@mkdir -p $(dir $@)
	@echo "Linking benchmark executable $@..."
ifeq ($(USE_PYBIND),1)
	$(CXX) $< $(SRC_OBJECTS) -o $@ $(BENCHMARK_LIBS) $(PYTHON_LIBS) $(PYBIND11_LIBS) $(RPATH_FLAGS)
else
	# When pybind is disabled, don't try to link non-existent mettagrid_c.o
	$(CXX) $< $(filter-out %mettagrid_c.o,$(SRC_OBJECTS)) -o $@ $(BENCHMARK_LIBS)
endif

# Run all benchmarks
ifeq ($(USE_PYBIND),1)
benchmark: check-bench-tools check-pybind11 clean-build-dir $(SRC_OBJECTS) $(BENCH_EXECUTABLES)
else
benchmark: check-bench-tools clean-build-dir $(SRC_OBJECTS) $(BENCH_EXECUTABLES)
endif
	$(call report_pybind_mode)
	@echo "Running all benchmarks..."
	@for f in $(BENCH_EXECUTABLES); do \
		echo "Running $$f"; \
		PYTHONHOME="$(UV_ROOT)" \
		PYTHONPATH="$(PYTHON_STDLIB):$(PYTHON_DYNLOAD):$(PYTHON_SITE_PACKAGES)" \
		LD_LIBRARY_PATH="$(PYTHON_LIB_DIR):$$LD_LIBRARY_PATH" \
		DYLD_LIBRARY_PATH="$(PYTHON_LIB_DIR):$$DYLD_LIBRARY_PATH" \
		"$$f" || exit 1; \
	done

# Add this new target for JSON benchmark output
bench-json: check-bench-tools clean-build-dir $(SRC_OBJECTS) $(BENCH_EXECUTABLES)
	$(call report_pybind_mode)
	@echo "Running all benchmarks with JSON output..."
	@mkdir -p benchmark_output
	@for f in $(BENCH_EXECUTABLES); do \
		echo "Running $$f with JSON output..."; \
		PYTHONHOME="$(UV_ROOT)" \
		PYTHONPATH="$(PYTHON_STDLIB):$(PYTHON_DYNLOAD):$(PYTHON_SITE_PACKAGES)" \
		LD_LIBRARY_PATH="$(PYTHON_LIB_DIR):$$LD_LIBRARY_PATH" \
		DYLD_LIBRARY_PATH="$(PYTHON_LIB_DIR):$$DYLD_LIBRARY_PATH" \
		"$$f" --benchmark_format=json > benchmark_output/$$(basename "$$f").json || \
		echo "Error running $$f with JSON format"; \
	done
	@echo "JSON outputs created in benchmark_output directory"

#-----------------------
# Pybind11
#-----------------------

ifeq ($(USE_PYBIND),1)
check-pybind11:
	@echo "Checking for pybind11..."
	@python3 -c "import pybind11" >/dev/null 2>&1 || \
		{ echo "pybind11 Python package not found. Run 'make install-pybind11' to install."; exit 1; }
	@if [ "$(shell uname)" = "Darwin" ]; then \
		if brew list pybind11 &>/dev/null; then \
			echo "pybind11 is installed via Homebrew."; \
		else \
			echo "pybind11 is not installed via Homebrew. Run 'make install-pybind11' to install."; \
			exit 1; \
		fi; \
	elif [ -f /etc/debian_version ]; then \
		dpkg -l python3-pybind11 >/dev/null 2>&1 || \
			{ echo "pybind11 is not installed. Run 'make install-pybind11' to install."; exit 1; }; \
	fi
	@echo "pybind11 is properly installed."

# Install pybind11
install-pybind11:
	@echo "Installing pybind11..."
	@if [ "$(shell uname)" = "Darwin" ]; then \
		echo "Detected macOS. Installing pybind11 via Homebrew..."; \
		brew install pybind11 || echo "Failed to install pybind11 via Homebrew. Trying pip..."; \
		pip3 install pybind11 || echo "Failed to install pybind11. Please install manually."; \
	elif [ -f /etc/debian_version ]; then \
		echo "Detected Debian/Ubuntu. Installing pybind11 via apt..."; \
		sudo apt-get update && sudo apt-get install -y python3-pybind11 || \
			{ echo "Failed to install pybind11 via apt. Trying pip..."; \
			  pip3 install pybind11 || echo "Failed to install pybind11. Please install manually."; }; \
	else \
		echo "Installing pybind11 via pip..."; \
		pip3 install pybind11 || echo "Failed to install pybind11. Please install manually."; \
	fi
else
# Dummy targets when pybind is disabled
check-pybind11:
	@echo "Pybind11 support is disabled (USE_PYBIND=0)."

install-pybind11:
	@echo "Pybind11 support is disabled (USE_PYBIND=0). Skipping installation."
endif

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
	@echo "All tasks completed (Pybind11 support: $(if $(filter 1,$(USE_PYBIND)),enabled,disabled))."
