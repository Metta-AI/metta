# Default target when just running 'make'
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  install-dependencies - Install all Python and C++ build dependencies"
	@echo "  build                - Build mettagrid using setup-tools"
	@echo "  build-tests          - Build all test files"
	@echo "  build-benchmarks     - Build all benchmark files"
	@echo "  format               - Format C++/C files"
	@echo "  test                 - Run all unit tests"
	@echo "  benchmark            - Run all benchmarks"
	@echo "  clean                - Clean build and test files"
	@echo "  all                  - Run format and test"


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
DEPS_TRACKING_DIR := $(BUILD_DIR)/.deps

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

# Check for ccache and auto-install on macOS if missing
CCACHE := $(shell which ccache 2>/dev/null)
ifneq ($(CCACHE),)
    CXX := $(CCACHE) $(CXX)
    $(info Using ccache: $(CCACHE))
else
    $(info ccache not found - consider installing for faster rebuilds)
    ifeq ($(shell uname), Darwin)
        $(info Run 'make install-ccache' to install via Homebrew)
    else
        $(info Install ccache: apt-get install ccache (Linux) or yum install ccache (RHEL/CentOS))
    endif
endif

# Get Python paths from the UV venv
PYTHON_PREFIX := $(shell $(PYTHON_BIN) -c "import sys; print(sys.prefix)")
PYTHON_VERSION := $(shell $(PYTHON_BIN) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_STDLIB := $(shell $(PYTHON_BIN) -c "import sysconfig; print(sysconfig.get_path('stdlib'))")
PYTHON_DYNLOAD := $(PYTHON_STDLIB)/lib-dynload
PYTHON_SITE_PACKAGES := $(shell $(PYTHON_BIN) -c "import site; print(site.getsitepackages()[0])")
PYTHON_LIB_DIR := $(shell $(PYTHON_BIN) -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PYTHON_HOME := $(PYTHON_PREFIX)
PYTHON_CFLAGS := $(shell $(PYTHON_BIN) -m pybind11 --includes)

# Update the Python library definitions
PYTHON_LIBS = -L$(PYTHON_LIB_DIR) -lpython$(PYTHON_VERSION)

PYBIND11_INCLUDE = $(shell $(PYTHON_BIN) -m pybind11 --includes 2>/dev/null | grep -o '\-I[^ ]*pybind11[^ ]*' | head -1 | sed 's/-I//')
NUMPY_INCLUDE = $(shell $(PYTHON_BIN) -c "import numpy; print(numpy.get_include())" 2>/dev/null)
PYTHON_INCLUDE := $(shell $(PYTHON_BIN) -c "import sysconfig; print(sysconfig.get_config_var('INCLUDEPY'))")

CXXFLAGS = -std=c++20 -Wall -g \
	-I$(SRC_DIR) -I$(THIRD_PARTY_DIR) -I$(TEST_DIR) \
	-I$(PYTHON_INCLUDE) -I$(PYBIND11_INCLUDE) -I$(NUMPY_INCLUDE)\
	-I$(GTEST_DIR)/include -I$(GBENCHMARK_DIR)/include \
	-DPYTHON_HOME=\"$(PYTHON_HOME)\" \
	-DPYTHON_STDLIB=\"$(PYTHON_STDLIB)\"

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

LDFLAGS += -L$(GBENCHMARK_DIR)/lib -lbenchmark_main -lbenchmark \
           -L$(GTEST_DIR)/lib -lgtest -lgtest_main

ifeq ($(shell uname), Darwin)
    CXXFLAGS += -I/opt/homebrew/include
    LDFLAGS  += -L/opt/homebrew/lib
endif

CXXFLAGS_METTAGRID = $(CXXFLAGS) -fvisibility=hidden -O3

# Source files for mettagrid core library
SRC_SOURCES := $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*/*.cpp) $(wildcard $(SRC_DIR)/*/*/*.cpp)
SRC_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_SRC_DIR)/%.o,$(SRC_SOURCES))

TEST_SOURCES := $(wildcard $(TEST_DIR)/*.cpp $(TEST_DIR)/**/*.cpp)
TEST_OBJECTS := $(patsubst $(TEST_DIR)/%.cpp,$(BUILD_TEST_DIR)/%.o,$(TEST_SOURCES))
TEST_EXECUTABLES := $(patsubst $(BUILD_TEST_DIR)/%.o,$(BUILD_TEST_DIR)/%,$(TEST_OBJECTS))

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

# Install ccache on macOS
.PHONY: install-ccache
install-ccache:
	@echo "Installing ccache via Homebrew..."
	@if [ "$(shell uname)" = "Darwin" ]; then \
		brew install ccache && \
		echo "✅ ccache installed. Restart your build to use it." && \
		echo "💡 Consider adding ccache to your PATH: export PATH=\"/opt/homebrew/opt/ccache/libexec:\$$PATH\""; \
	else \
		echo "This target is for macOS only. Install ccache manually:"; \
		echo "  Ubuntu/Debian: apt-get install ccache"; \
		echo "  RHEL/CentOS: yum install ccache"; \
		echo "  Arch: pacman -S ccache"; \
	fi

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

.PHONY: info check-common-dependencies check-test-dependencies check-build-dependencies check-benchmark-dependencies

# Main dependency check target that calls all others
info:
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

# Build-specific dependencies
check-build-dependencies: info
	@echo ""
	@echo "🧱 Build dependencies"

	@echo "🔍 Checking for uv..."
	@if which uv >/dev/null 2>&1; then \
		echo "  ✅ Found uv"; \
	else \
		echo "  ❌ uv not found -- install with 'pip install uv' or check https://github.com/astral-sh/uv for installation instructions"; \
		exit 1; \
	fi

	@echo "🔍 Checking for pybind11..."
	@if [ -n "$(PYBIND11_INCLUDE)" ]; then \
		echo "  PYBIND11_INCLUDE     = $(PYBIND11_INCLUDE)"; \
	else \
		echo "  PYBIND11_INCLUDE     = ❌ not found -- run \"make install-dependencies\""; \
		exit 1; \
	fi

	@echo "🔍 Checking for numpy..."
	@if [ -n "$(NUMPY_INCLUDE)" ]; then \
		echo "  NUMPY_INCLUDE        = $(NUMPY_INCLUDE)"; \
	else \
		echo "  NUMPY_INCLUDE        = ❌ not found -- run \"make install-dependencies\""; \
		exit 1; \
	fi

	@echo "🔍 Checking for Python development headers..."
	@if ! echo '#include <Python.h>' | $(CXX) $(PYTHON_CFLAGS) -x c++ -E - >/dev/null 2>&1; then \
		echo "❌ Python.h not found. Attempting to install..."; \
		if [ "$$CI" = "true" ] && command -v apt-get >/dev/null 2>&1; then \
			echo "📦 Installing python3-dev in CI environment..."; \
			sudo apt-get update -qq && sudo apt-get install -y python3-dev python3-distutils; \
		else \
			echo "❌ Python development headers missing. Please install:"; \
			echo "   Ubuntu/Debian: sudo apt-get install python3-dev python3-distutils"; \
			echo "   RHEL/CentOS:   sudo yum install python3-devel"; \
			echo "   Fedora:        sudo dnf install python3-devel"; \
			echo "   macOS:         Headers should be included with Python installation"; \
			exit 1; \
		fi; \
	fi
	@echo "✅ Python development headers found."

# Common dependencies needed for test and benchmark
check-common-dependencies: info
	@echo "🔍 Checking for g++..."
	@if which g++ >/dev/null 2>&1; then \
		echo "  ✅ Found g++"; \
	else \
		echo "  ❌ g++ not found -- install with your package manager (e.g., 'apt install g++' or 'brew install gcc')"; \
		exit 1; \
	fi
	@echo "🔍 Checking for cmake..."
	@if which cmake >/dev/null 2>&1; then \
		echo "  ✅ Found cmake"; \
	else \
		echo "  ❌ cmake not found -- install with your package manager (e.g., 'apt install cmake' or 'brew install cmake')"; \
		exit 1; \
	fi

# Test-specific dependencies
check-test-dependencies: check-common-dependencies
	@echo ""
	@echo "🧪 Test dependencies"
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

# Benchmark-specific dependencies
check-benchmark-dependencies: check-common-dependencies
	@echo ""
	@echo "⏱️ Benchmark dependencies"
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

#-----------------------
# Build Targets
#-----------------------

# Build target calls the setup.py script with UV venv activated
# NOTE: setup-tools does not build the SRC_OBJECTS we want for tests
.PHONY: build
build: check-build-dependencies create-dirs
	@echo "🔨 Building mettagrid using setup-tools..."
	@uv pip install -e .
	@echo "✅ Build completed successfully."

# Build the source object files needed for tests and benchmarks
.PHONY: build-src-objects
build-src-objects: check-build-dependencies create-dirs $(SRC_OBJECTS)
	@echo "🧱 Building source object files..."
	@echo "Built $(words $(SRC_OBJECTS)) object files:"
	@for obj in $(SRC_OBJECTS); do \
		if [ -f "$$obj" ]; then \
			echo "  ✅ $$obj"; \
		else \
			echo "  ❌ $$obj (missing)"; \
		fi; \
	done
	@echo "✅ Source object files ready for linking"

# Modified build target to also build source objects
.PHONY: build-all
build-all: build build-src-objects
	@echo "✅ Complete build (Python extension + source objects) completed"

# Build just the test files
.PHONY: build-tests
build-tests: check-test-dependencies create-dirs $(SRC_OBJECTS)
	@echo "🧪 Building test executables..."
	@for test_exec in $(TEST_EXECUTABLES); do \
		echo "Building $$test_exec..."; \
		$(MAKE) "$$test_exec" || exit 1; \
	done
	@echo "✅ Finished building test executables"

# Build just the benchmark files
.PHONY: build-benchmarks
build-benchmarks: check-benchmark-dependencies create-dirs $(SRC_OBJECTS)
	@echo "📈 Building benchmark executables..."
	@for bench_exec in $(BENCH_EXECUTABLES); do \
		echo "Building $$bench_exec..."; \
		$(MAKE) "$$bench_exec" || exit 1; \
	done
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
# Directory Creation
#-----------------------

# Define all directories that need to be created
BUILD_DIRS := $(BUILD_DIR) \
              $(BUILD_SRC_DIR) \
              $(BUILD_TEST_DIR) \
              $(BUILD_BENCH_DIR) \
              $(DEPS_INSTALL_DIR) \
              $(DEPS_TRACKING_DIR)

# Create all directories with a single rule
$(BUILD_DIRS):
	@mkdir -p $@

# Convenience target to create all directories at once
.PHONY: create-dirs
create-dirs: $(BUILD_DIRS)
	@echo "✅ All build directories created"

#-----------------------
# Core Library Build
#-----------------------

# Generate dependency files automatically
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPS_TRACKING_DIR)/$*.d

# Create build configuration tracking file
BUILD_CONFIG := $(BUILD_DIR)/.build_config
$(BUILD_CONFIG): | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	@echo "PYTHON_BIN=$(PYTHON_BIN)" > $@
	@echo "PYTHON_VERSION=$(PYTHON_VERSION)" >> $@
	@echo "CXXFLAGS=$(CXXFLAGS)" >> $@
	@echo "CXX=$(CXX)" >> $@
	@echo "PYTHON_INCLUDE=$(PYTHON_INCLUDE)" >> $@

# Updated compilation rules with dependency tracking
$(BUILD_SRC_DIR)/%.o: $(SRC_DIR)/%.cpp $(BUILD_CONFIG) | $(BUILD_SRC_DIR) $(DEPS_TRACKING_DIR)
	@mkdir -p $(dir $@) $(dir $(DEPS_TRACKING_DIR)/src_$(subst /,_,$*).d)
	$(CXX) $(CXXFLAGS_METTAGRID) $(DEPFLAGS) -c $< -o $@
	@if [ -f "$(DEPS_TRACKING_DIR)/$*.d" ]; then \
		mv "$(DEPS_TRACKING_DIR)/$*.d" "$(DEPS_TRACKING_DIR)/src_$(subst /,_,$*).d"; \
	fi

$(BUILD_TEST_DIR)/%.o: $(TEST_DIR)/%.cpp $(BUILD_CONFIG) | $(BUILD_TEST_DIR) $(DEPS_TRACKING_DIR)
	@mkdir -p $(dir $@) $(dir $(DEPS_TRACKING_DIR)/test_$(subst /,_,$*).d)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@
	@if [ -f "$(DEPS_TRACKING_DIR)/$*.d" ]; then \
		mv "$(DEPS_TRACKING_DIR)/$*.d" "$(DEPS_TRACKING_DIR)/test_$(subst /,_,$*).d"; \
	fi

$(BUILD_BENCH_DIR)/%.o: $(BENCH_DIR)/%.cpp $(BUILD_CONFIG) | $(BUILD_BENCH_DIR) $(DEPS_TRACKING_DIR)
	@mkdir -p $(dir $@) $(dir $(DEPS_TRACKING_DIR)/bench_$(subst /,_,$*).d)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@
	@if [ -f "$(DEPS_TRACKING_DIR)/$*.d" ]; then \
		mv "$(DEPS_TRACKING_DIR)/$*.d" "$(DEPS_TRACKING_DIR)/bench_$(subst /,_,$*).d"; \
	fi

# Include dependency files from the tracking directory:
-include $(wildcard $(DEPS_TRACKING_DIR)/*.d)

# Build a static library from all source files
$(BUILD_DIR)/libmettagrid.a: $(SRC_OBJECTS)
	@mkdir -p $(dir $@)
	ar rcs $@ $^

#-----------------------
# Testing Build
#-----------------------

$(BUILD_TEST_DIR)/%: $(BUILD_TEST_DIR)/%.o $(SRC_OBJECTS)
	@mkdir -p $(dir $@)
	@echo "[LINK] $(CXX) $^ $(FORCE_GTEST_MAIN) -o $@ $(LDFLAGS) $(RPATH_FLAGS) $(PYTHON_LIBS)"
	$(CXX) $^ $(FORCE_GTEST_MAIN) -o $@ $(LDFLAGS) $(RPATH_FLAGS) $(PYTHON_LIBS)


#-----------------------
# Benchmarking Build
#-----------------------

# Link benchmark executables with the mettagrid library
$(BUILD_BENCH_DIR)/%: $(BUILD_BENCH_DIR)/%.o $(SRC_OBJECTS)
	$(CXX) $^ -o $@ $(LDFLAGS) $(RPATH_FLAGS) $(PYTHON_LIBS)


#-----------------------
# Testing
#-----------------------

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
