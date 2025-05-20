# Default target when just running 'make'
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  install-dependencies - Install all Python and C++ build dependencies"
	@echo "  check-dependencies   - Verify system is ready to build"
	@echo "  build                - Build mettagrid using setup-tools"
	@echo "  build-tests          - Build all test files"
	@echo "  build-benchmarks     - Build all benchmark files"
	@echo "  format               - Format C++/C files"
	@echo "  test                 - Run all unit tests"
	@echo "  benchmark            - Run all benchmarks"
	@echo "  clean                - Clean build and test files"
	@echo "  all                  - Run format and test"
	@echo ""
	@echo "If build fails, try: make check-dependencies"

# Use UV venv at repo root
REPO_ROOT := $(realpath ..)
PYTHON_BIN := $(REPO_ROOT)/.venv/bin/python

PROJECT_ROOT := $(REPO_ROOT)/mettagrid

# Directories
SRC_DIR = $(PROJECT_ROOT)/mettagrid
THIRD_PARTY_DIR = $(PROJECT_ROOT)/third_party
TEST_DIR = $(PROJECT_ROOT)/tests
BENCH_DIR = $(PROJECT_ROOT)/benchmarks
BUILD_DIR = $(PROJECT_ROOT)/build
BUILD_SRC_DIR = $(BUILD_DIR)/mettagrid
BUILD_TEST_DIR = $(BUILD_DIR)/tests
BUILD_BENCH_DIR = $(BUILD_DIR)/benchmarks

DEPS_INSTALL_DIR := $(BUILD_DIR)/deps
GTEST_DIR := $(DEPS_INSTALL_DIR)/gtest
GBENCHMARK_DIR := $(DEPS_INSTALL_DIR)/gbenchmark

# Detect platform
UNAME_S := $(shell uname)

ifeq ($(UNAME_S), Darwin)
	HOMEBREW_PREFIX := $(shell brew --prefix)

	# Use Homebrew googletest if available
	ifneq ($(wildcard $(HOMEBREW_PREFIX)/opt/googletest/include/gtest/gtest.h),)
		GTEST_DIR := $(HOMEBREW_PREFIX)/opt/googletest
	endif

	# Use Homebrew google-benchmark if available
	ifneq ($(wildcard $(HOMEBREW_PREFIX)/opt/google-benchmark/include/benchmark/benchmark.h),)
		GBENCHMARK_DIR := $(HOMEBREW_PREFIX)/opt/google-benchmark
	endif

endif

FIND_EXECUTABLE := $(if $(filter Darwin,$(UNAME_S)),-perm -111,-executable)

GTEST_LIB_DIR := $(GTEST_DIR)/lib
ifeq ($(UNAME_S), Darwin)
  FORCE_GTEST_MAIN := $(shell GTEST_LIB_DIR=$(GTEST_LIB_DIR) ; echo -Wl,-force_load,$${GTEST_LIB_DIR}/libgtest_main.a)
else
  FORCE_GTEST_MAIN := $(shell GTEST_LIB_DIR=$(GTEST_LIB_DIR) ; echo -Wl,--whole-archive $${GTEST_LIB_DIR}/libgtest_main.a -Wl,--no-whole-archive)
endif

# Compiler settings
CXX = g++

# Get Python paths from the UV venv
PYTHON_VERSION := $(shell $(PYTHON_BIN) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_INCLUDE := $(shell $(PYTHON_BIN) -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_STDLIB := $(shell $(PYTHON_BIN) -c "import sysconfig; print(sysconfig.get_path('stdlib'))")
PYTHON_DYNLOAD := $(PYTHON_STDLIB)/lib-dynload
PYTHON_SITE_PACKAGES := $(shell $(PYTHON_BIN) -c "import site; print(site.getsitepackages()[0])")
PYTHON_LIB_DIR := $(shell $(PYTHON_BIN) -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PYTHON_HOME := $(shell $(PYTHON_BIN) -c "import os, sysconfig; print(os.path.dirname(sysconfig.get_path('stdlib')))")
PYTHON_CFLAGS := $(shell $(PYTHON_BIN) -m pybind11 --includes)

# Update the Python library definitions
PYTHON_LIBS = -L$(PYTHON_LIB_DIR) -lpython$(PYTHON_VERSION)

CXXFLAGS = -std=c++20 -Wall -g \
	-I$(SRC_DIR) -I$(THIRD_PARTY_DIR) -I$(TEST_DIR) \
	-I$(PYTHON_INCLUDE) \
	-DPYTHON_HOME=\"$(PYTHON_HOME)\" \
	-DPYTHON_STDLIB=\"$(PYTHON_STDLIB)\"

PYBIND11_INCLUDE = $(shell $(PYTHON_BIN) -m pybind11 --includes 2>/dev/null | grep -o '\-I[^ ]*pybind11[^ ]*' | head -1 | sed 's/-I//')
NUMPY_INCLUDE = $(shell $(PYTHON_BIN) -c "import numpy; print(numpy.get_include())" 2>/dev/null)

CXXFLAGS += -I$(PYBIND11_INCLUDE)
CXXFLAGS += -I$(NUMPY_INCLUDE)

export LD_LIBRARY_PATH := $(PYTHON_LIB_DIR):$(LD_LIBRARY_PATH)
export DYLD_LIBRARY_PATH := $(PYTHON_LIB_DIR):$(DYLD_LIBRARY_PATH)
export PYTHONHOME := $(PYTHON_HOME)
export PYTHONPATH := $(PYTHON_STDLIB):$(PYTHON_DYNLOAD):$(PYTHON_SITE_PACKAGES)

# Add RPATH settings for macOS
ifeq ($(shell uname), Darwin)
    RPATH_FLAGS = -Wl,-rpath,$(PYTHON_LIB_DIR)
else
    RPATH_FLAGS =
endif

CXXFLAGS += -I$(GTEST_DIR)/include
CXXFLAGS += -I$(GBENCHMARK_DIR)/include

LDFLAGS += -L$(GBENCHMARK_DIR)/lib -lbenchmark_main -lbenchmark \
           -L$(GTEST_DIR)/lib -lgtest -lgtest_main

ifeq ($(shell uname), Darwin)
    CXXFLAGS += -I/opt/homebrew/include
    LDFLAGS  += -L/opt/homebrew/lib
endif

CXXFLAGS_METTAGRID = $(CXXFLAGS) -fvisibility=hidden -O3

# Source files for mettagrid core library
SRC_SOURCES := $(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/**/*.cpp)
SRC_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_SRC_DIR)/%.o,$(SRC_SOURCES))

$(BUILD_TEST_DIR):
	@mkdir -p $(BUILD_TEST_DIR)

TEST_SOURCES := $(wildcard $(TEST_DIR)/*.cpp $(TEST_DIR)/**/*.cpp)
TEST_OBJECTS := $(patsubst $(TEST_DIR)/%.cpp,$(BUILD_TEST_DIR)/%.o,$(TEST_SOURCES))
TEST_EXECUTABLES := $(patsubst $(BUILD_TEST_DIR)/%.o,$(BUILD_TEST_DIR)/%,$(TEST_OBJECTS))

$(BUILD_BENCH_DIR):
	@mkdir -p $(BUILD_BENCH_DIR)

BENCH_SOURCES := $(wildcard $(BENCH_DIR)/*.cpp $(BENCH_DIR)/**/*.cpp)
BENCH_OBJECTS := $(patsubst $(BENCH_DIR)/%.cpp,$(BUILD_BENCH_DIR)/%.o,$(BENCH_SOURCES))
BENCH_EXECUTABLES := $(patsubst $(BUILD_BENCH_DIR)/%.o,$(BUILD_BENCH_DIR)/%,$(BENCH_OBJECTS))

# Check if we're in a virtual environment
.PHONY: check-venv
check-venv:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "ERROR: Not in a virtual environment!"; \
		echo "Please activate the virtual environment first with:"; \
		echo "  source .venv/bin/activate"; \
		exit 1; \
	else \
		echo "... in active virtual environment: $$VIRTUAL_ENV"; \
	fi

#-----------------------
# Install Dependencies
#-----------------------

# Install formatting tools on macOS
.PHONY: install-format-tools
install-format-tools:
	@echo "Installing required formatting tools..."
	@if [ "$(shell uname)" = "Darwin" ]; then \
		echo "Detected macOS. Installing tools via Homebrew..."; \
		brew install clang-format || echo "Failed to install clang-format. Please install manually."; \
	else \
		echo "This command only works on macOS. Please install tools manually:"; \
		echo "  - clang-format: apt-get install clang-format (Linux)"; \
	fi

# Install testing tools
.PHONY: install-test-tools
install-test-tools:
	@echo "Installing GoogleTest..."
	@if [ "$(shell uname)" = "Darwin" ]; then \
		brew install googletest || echo "Failed to install googletest via homebrew."; \
	else \
		@echo "Building GoogleTest from source..."; \
		mkdir -p $(GTEST_DIR); \
		cd $(BUILD_DIR) && \
		git clone --depth 1 --branch release-1.12.1 https://github.com/google/googletest.git && \
		mkdir -p googletest/build && cd googletest/build && \
		cmake -DCMAKE_INSTALL_PREFIX=$(abspath $(GTEST_DIR)) .. && \
		make -j$$(nproc) && make install; \
	fi


# Install benchmark tools
.PHONY: install-bench-tools
install-bench-tools:
	@echo "Installing Google Benchmark..."
	@if [ "$(shell uname)" = "Darwin" ]; then \
		brew install google-benchmark || echo "Failed to install google-benchmark via homebrew."; \
	else \
		@echo "Building Google Benchmark from source..."; \
		mkdir -p $(GBENCHMARK_DIR); \
		cd $(BUILD_DIR) && \
		rm -rf benchmark && \
		git clone --depth 1 https://github.com/google/benchmark.git && \
		git clone --depth 1 https://github.com/google/googletest.git benchmark/googletest && \
		mkdir -p benchmark/build && cd benchmark/build && \
		cmake -DCMAKE_INSTALL_PREFIX=$(abspath $(GBENCHMARK_DIR)) \
		      -DCMAKE_BUILD_TYPE=Release \
		      -DBENCHMARK_ENABLE_TESTING=OFF \
		      -DBENCHMARK_ENABLE_LTO=OFF \
		      -DBENCHMARK_BUILD_32_BITS=OFF \
		      .. && \
		make -j$$(nproc) && make install; \
	fi


.PHONY: install-dependencies
install-dependencies:
	@echo "📦 Installing pybind11 and numpy using uv..."
	@uv pip install --python $(PYTHON_BIN) pybind11 numpy
	@echo "📦 Installing clang-format"
	@$(MAKE) install-format-tools
	@echo "📦 Installing googletest"
	@$(MAKE) install-test-tools
	@echo "📦 Installing google-benchmark"
	@$(MAKE) install-bench-tools
	@echo "✅ All dependencies installed."

#-----------------------
# Check Dependencies
#-----------------------

.PHONY: check-dependencies
check-dependencies:
	@echo "Environment Info"
	@echo "  VIRTUAL_ENV          = $(VIRTUAL_ENV)"
	@echo "  PYTHON               = $(PYTHON_BIN)"
	@echo "  PYTHON_VERSION       = $(PYTHON_VERSION)"
	@echo "  REPO_ROOT            = $(REPO_ROOT)"
	@echo "  PROJECT_ROOT         = $(PROJECT_ROOT)"
	@echo "  PYTHON_BIN           = $(PYTHON_BIN)"
	@echo "  SRC_DIR              = $(SRC_DIR)"
	@echo "  BENCH_DIR            = $(BENCH_DIR)"
	@echo "  GTEST_DIR            = $(GTEST_DIR)"
	@echo "  GBENCHMARK_DIR       = $(GBENCHMARK_DIR)"
	@echo ""
	@echo "📦 Python Includes"
	@echo "  PYTHON_INCLUDE       = $(PYTHON_INCLUDE)"
	@echo "  PYTHON_STDLIB        = $(PYTHON_STDLIB)"
	@echo "  PYTHON_DYNLOAD       = $(PYTHON_DYNLOAD)"
	@echo "  PYTHON_SITE_PACKAGES = $(PYTHON_SITE_PACKAGES)"
	@echo "  PYTHON_HOME          = $(PYTHON_HOME)"
	@echo "  PYTHON_CFLAGS        = $(PYTHON_CFLAGS)"
	@echo ""
	@echo "🧱 Build dependencies"
	@if [ -n "$(PYBIND11_INCLUDE)" ]; then \
		echo "  PYBIND11_INCLUDE     = $(PYBIND11_INCLUDE)"; \
	else \
		echo "  PYBIND11_INCLUDE     = ❌ not found -- run \"make install-dependencies\""; \
		exit 1; \
	fi

	@echo "🔍 Python import test for numpy..."
	@$(PYTHON_BIN) -c "import numpy; print('✅ numpy imported successfully')" || \
		{ echo "❌ Failed to import numpy — possibly missing .so or metadata files"; exit 1; }

	@if [ -n "$(NUMPY_INCLUDE)" ]; then \
		echo "  NUMPY_INCLUDE        = $(NUMPY_INCLUDE)"; \
	else \
		echo "  NUMPY_INCLUDE        = ❌ not found -- run \"make install-dependencies\""; \
		exit 1; \
	fi

	@echo "Checking for Google Test headers..."
	@if [ ! -f "$(GTEST_DIR)/include/gtest/gtest.h" ]; then \
		if [ -f "$$(brew --prefix)/opt/googletest/include/gtest/gtest.h" ]; then \
			echo "✅ Found gtest headers in Homebrew path"; \
		else \
			echo "❌ gtest/gtest.h not found at $(GTEST_DIR)/include/gtest/gtest.h"; \
			echo "   Run 'make install-test-tools' or check Homebrew setup."; \
			exit 1; \
		fi; \
	else \
		echo "✅ Found gtest headers in $(GTEST_DIR)/include"; \
	fi

	@echo "Checking for Google Benchmark headers..."
	@if [ ! -f "$(GBENCHMARK_DIR)/include/benchmark/benchmark.h" ]; then \
		if [ -f "$$(brew --prefix)/opt/benchmark/include/benchmark/benchmark.h" ]; then \
			echo "✅ Found benchmark headers in Homebrew path"; \
		else \
			echo "❌ benchmark/benchmark.h not found at $(GBENCHMARK_DIR)/include/benchmark/benchmark.h"; \
			echo "   Run 'make install-bench-tools' or check Homebrew setup."; \
			exit 1; \
		fi; \
	else \
		echo "✅ Found benchmark headers in $(GBENCHMARK_DIR)/include"; \
	fi


	@echo "🔧 Tooling Checks"
	@which g++ >/dev/null 2>&1 || { echo "❌ g++ not found"; exit 1; }
	@which cmake >/dev/null 2>&1 || { echo "❌ cmake not found"; exit 1; }
	@which uv >/dev/null 2>&1 || { echo "❌ uv not found"; exit 1; }
	@echo "  ✅ Found g++"
	@echo "  ✅ Found cmake"
	@echo "  ✅ Found uv"
	@echo ""
	@echo "⚙️ Compiler"
	@echo "  CXX                  = $(CXX)"
	@echo "  CXXFLAGS             = $(CXXFLAGS)" 
	@echo "  CXXFLAGS_METTAGRID   = $(CXXFLAGS_METTAGRID)"
	@echo ""
	@echo "📚 Linking"
	@echo "  PYTHON_LIBS          = $(PYTHON_LIBS)"
	@echo "  LDFLAGS	          = $(LDFLAGS)"
	@echo "  RPATH_FLAGS          = $(RPATH_FLAGS)"

#-----------------------
# Build Targets
#-----------------------

# Build target that calls the setup.py script with UV venv activated
.PHONY: build
build: check-dependencies
	@echo "🔨 Building mettagrid using setup-tools..."
	@uv pip install -e .
	@echo "✅ Build completed successfully."

# Build just the test files
.PHONY: build-tests
build-tests: $(TEST_EXECUTABLES)
	@echo "🧪 Built $(words $(TEST_EXECUTABLES)) test executable(s):"
	@find $(BUILD_TEST_DIR) -type f $(FIND_EXECUTABLE) -exec ls -lh {} \;
	@echo "✅ Finished building test executables"

# Build just the benchmark files
.PHONY: build-benchmarks
build-benchmarks: $(BENCH_EXECUTABLES)
	@echo "📈 Built $(words $(BENCH_EXECUTABLES)) benchmark executable(s):"
	@find $(BUILD_BENCH_DIR) -type f $(FIND_EXECUTABLE) -exec ls -lh {} \;
	@echo "✅ Finished building benchmark executables"

#-----------------------
# Formatting
#-----------------------

# Check if the required formatting tools are installed
.PHONY: check-format-tools
check-format-tools:
	@echo "Checking for required formatting tools..."
	@which clang-format >/dev/null 2>&1 || \
		{ echo "clang-format is not installed. On macOS use 'brew install clang-format'"; \
		  echo "On Linux use 'apt-get install clang-format'"; \
		  echo "Or run 'make install-format-tools' on macOS"; exit 1; }
	@echo "All required formatting tools are installed."

# Format only C/C++ code and skip Cython files entirely
.PHONY: format
format: check-format-tools
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
	$(CXX) $(CXXFLAGS_METTAGRID) -c $< -o $@

# Build a static library from all source files
$(BUILD_DIR)/libmettagrid.a: $(SRC_OBJECTS)
	@mkdir -p $(dir $@)
	ar rcs $@ $^

#-----------------------
# Testing
#-----------------------

# Compile individual test files
$(BUILD_TEST_DIR)/%.o: $(TEST_DIR)/%.cpp | $(BUILD_TEST_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_TEST_DIR)/%: $(BUILD_TEST_DIR)/%.o $(SRC_OBJECTS)
	@mkdir -p $(dir $@)
	@echo "[LINK] $(CXX) $^ $(FORCE_GTEST_MAIN) -o $@ $(LDFLAGS) $(RPATH_FLAGS) $(PYTHON_LIBS)"
	$(CXX) $^ $(FORCE_GTEST_MAIN) -o $@ $(LDFLAGS) $(RPATH_FLAGS) $(PYTHON_LIBS)

# Run C++ tests

.PHONY: debug-test-vars
debug-test-vars:
	@echo ""
	@echo "TEST_DIR = $(TEST_DIR)"
	@echo "BUILD_TEST_DIR = $(BUILD_TEST_DIR)"
	@echo "TEST_SOURCES = $(TEST_SOURCES)"
	@echo "TEST_OBJECTS = $(TEST_OBJECTS)"
	@echo "TEST_EXECUTABLES = $(TEST_EXECUTABLES)"
	@echo ""

.PHONY: test
test: build build-tests debug-test-vars
	@echo "Running all C++ tests..."
	@[ -z "$(TEST_EXECUTABLES)" ] && echo "WARNING: No test executables found!" || true
	@for f in $(TEST_EXECUTABLES); do \
		if [ -f "$$f" ]; then \
			echo "Running $$f"; \
			"$$f" --gtest_color=yes || exit 1; \
		else \
			echo "ERROR: Test executable $$f not found!"; \
			exit 1; \
		fi; \
	done

# Test a specific test file
.PHONY: test-%
test-%: build $(BUILD_TEST_DIR)/%
	@echo "Running test $*..."
	@if [ -f "$(BUILD_TEST_DIR)/$*" ]; then \
		"$(BUILD_TEST_DIR)/$*" --gtest_color=yes; \
	else \
		echo "ERROR: Test executable $(BUILD_TEST_DIR)/$* not found!"; \
		exit 1; \
	fi

.PHONY: test-python
test-python: build check-venv
	@echo "Running python tests with coverage"
	pytest --cov=metta --cov-report=term-missing

#-----------------------
# Benchmarking
#-----------------------

# Compile individual benchmark files
$(BUILD_BENCH_DIR)/%.o: $(BENCH_DIR)/%.cpp | $(BUILD_BENCH_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link benchmark executables with the mettagrid library
$(BUILD_BENCH_DIR)/%: $(BUILD_BENCH_DIR)/%.o $(SRC_OBJECTS)
	$(CXX) $^ -o $@ $(LDFLAGS) $(RPATH_FLAGS) $(PYTHON_LIBS) $(PYBIND11_LIBS)

# Run all benchmarks
.PHONY: debug-bench-vars
debug-bench-vars:
	@echo ""
	@echo "BENCH_DIR = $(BENCH_DIR)"
	@echo "BUILD_BENCH_DIR = $(BUILD_BENCH_DIR)"
	@echo "BENCH_SOURCES = $(BENCH_SOURCES)"
	@echo "BENCH_OBJECTS = $(BENCH_OBJECTS)"
	@echo "BENCH_EXECUTABLES = $(BENCH_EXECUTABLES)"
	@echo ""

.PHONY: benchmark
benchmark: build build-benchmarks debug-bench-vars
	@echo "Running all benchmarks..."
	@[ -z "$(BENCH_EXECUTABLES)" ] && echo "WARNING: No benchmark executables found!" || true
	@for f in $(BENCH_EXECUTABLES); do \
		if [ -f "$$f" ]; then \
			echo "Running $$f"; \
			"$$f" || exit 1; \
		else \
			echo "ERROR: Benchmark executable $$f not found!"; \
			exit 1; \
		fi; \
	done


.PHONY: bench-json
bench-json: build build-benchmarks debug-bench-vars
	@echo "Running all benchmarks with JSON output..."
	@mkdir -p benchmark_output
	@[ -z "$(BENCH_EXECUTABLES)" ] && echo "WARNING: No benchmark executables found!" || true
	@for f in $(BENCH_EXECUTABLES); do \
		if [ -f "$$f" ]; then \
			echo "Running $$f with JSON output..."; \
			"$$f" --benchmark_format=json > benchmark_output/$$(basename "$$f").json || \
			echo "Error running $$f with JSON format"; \
		else \
			echo "ERROR: Benchmark executable $$f not found!"; \
			exit 1; \
		fi; \
	done
	@echo "JSON outputs created in benchmark_output directory"


#-----------------------
# Other targets
#-----------------------

.PHONY: clean
clean:
	@echo "(MettaGrid) Removing any accidental venvs (preserving UV venvs)..."
	@bash -c '\
		VENV_PATHS=(".venv" "venv" ".env" "env" "virtualenv" ".virtualenv"); \
		for venv_name in "$${VENV_PATHS[@]}"; do \
			venv_path="$$(pwd)/$$venv_name"; \
			if [ -d "$$venv_path" ]; then \
				if [ -f "$$venv_path/pyvenv.cfg" ] && grep -q "UV_VENV" "$$venv_path/pyvenv.cfg" 2>/dev/null; then \
					echo "(MettaGrid) ✅ Preserving $$venv_name (UV virtual environment)"; \
				else \
					echo "(MettaGrid) Removing $$venv_name virtual environment..."; \
					rm -rf "$$venv_path"; \
					echo "(MettaGrid) ✅ Removed $$venv_name virtual environment"; \
				fi; \
			fi; \
		done'

	@echo "(MettaGrid) Cleaning build files..."
	@if [ -d "$(BUILD_DIR)" ]; then \
		rm -rf $(BUILD_DIR); \
		echo "(MettaGrid) ✅ Removed $(BUILD_DIR)"; \
	else \
		echo "(MettaGrid) ✅ No build directory to clean"; \
	fi

	@echo "(MettaGrid) Cleaning .so files from mettagrid directory..."
	@if [ -d "mettagrid" ]; then \
		found_files=$$(find mettagrid -name "*.so" -type f | wc -l); \
		if [ $$found_files -gt 0 ]; then \
			find mettagrid -name "*.so" -type f -delete; \
			echo "(MettaGrid) ✅ Removed $$found_files .so files from mettagrid"; \
		else \
			echo "(MettaGrid) ✅ No .so files found in mettagrid"; \
		fi; \
	else \
		echo "(MettaGrid) ✅ No mettagrid directory found, skipping .so cleanup"; \
	fi
	
	@echo "(MettaGrid) ✅ Clean completed successfully"

# Run format and test
.PHONY: all
all: format test
	@echo "All tasks completed."
