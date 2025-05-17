# Default target when just running 'make'
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  install-dependencies - Install Python and C++ dependencies"
	@echo "  check-dependencies   - Verify system readiness to build"
	@echo "  build                - Build mettagrid using the rebuild script"
	@echo "  build-clean          - Build mettagrid with clean option"
	@echo "  build-for-ci         - Build all source, test, and benchmark files without running tests (for CI)"
	@echo "  format               - Format C++/C files"
	@echo "  check-tools          - Check if required formatting tools are installed"
	@echo "  install-tools        - Install required formatting tools (macOS only)"
	@echo "  test                 - Run all unit tests"
	@echo "  check-test-tools     - Check if required testing tools are installed"
	@echo "  install-test-tools   - Install required testing tools"
	@echo "  benchmark            - Run all benchmarks"
	@echo "  check-bench-tools    - Check if required benchmark tools are installed"
	@echo "  install-bench-tools  - Install required benchmark tools"
	@echo "  clean                - Clean build and test files"
	@echo "  all                  - Run format and test"
	@echo ""
	@echo "If build fails, try: make check-dependencies"

# Directories
SRC_DIR = mettagrid
THIRD_PARTY_DIR = third_party
TEST_DIR = tests
BENCH_DIR = benchmarks
BUILD_DIR = build
BUILD_SRC_DIR = $(BUILD_DIR)/mettagrid
BUILD_TEST_DIR = $(BUILD_DIR)/tests
BUILD_BENCH_DIR = $(BUILD_DIR)/benchmarks

# Use UV venv at repo root
REPO_ROOT := $(realpath ..)
PYTHON3 := "$(REPO_ROOT)/.venv/bin/python"

# Compiler settings
CXX = g++

# Get Python paths from the UV venv
PYTHON_VERSION := $(shell $(PYTHON3) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_INCLUDE := $(shell $(PYTHON3) -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_STDLIB := $(shell $(PYTHON3) -c "import sysconfig; print(sysconfig.get_path('stdlib'))")
PYTHON_DYNLOAD := $(PYTHON_STDLIB)/lib-dynload
PYTHON_SITE_PACKAGES := $(shell $(PYTHON3) -c "import site; print(site.getsitepackages()[0])")
PYTHON_LIB_DIR := $(shell $(PYTHON3) -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PYTHON_HOME := $(shell $(PYTHON3) -c "import os, sysconfig; print(os.path.dirname(sysconfig.get_path('stdlib')))")
PYTHON_CFLAGS := $(shell $(PYTHON3) -m pybind11 --includes)
PYTHON_LDFLAGS := $(PYTHON_LIBS)

# Update the Python library definitions
PYTHON_LIBS = -L$(PYTHON_LIB_DIR) -lpython$(PYTHON_VERSION)

CXXFLAGS = -std=c++23 -Wall -g \
	-I$(SRC_DIR) -I$(THIRD_PARTY_DIR) -I$(TEST_DIR) \
	-I$(PYTHON_INCLUDE) \
	-DPYTHON_HOME=\"$(PYTHON_HOME)\" \
	-DPYTHON_STDLIB=\"$(PYTHON_STDLIB)\"


PYBIND11_INCLUDE = $(shell $(PYTHON3) -m pybind11 --includes 2>/dev/null | grep -o '\-I[^ ]*pybind11[^ ]*' | head -1 | sed 's/-I//')
NUMPY_INCLUDE = $(shell $(PYTHON3) -c "import numpy; print(numpy.get_include())" 2>/dev/null)

CXXFLAGS += -I$(PYBIND11_INCLUDE)
CXXFLAGS += -I$(NUMPY_INCLUDE)

# Add RPATH settings for macOS
ifeq ($(shell uname), Darwin)
    RPATH_FLAGS = -Wl,-rpath,$(PYTHON_LIB_DIR)
else
    RPATH_FLAGS =
endif

# Google Test settings
GTEST_INCLUDE = $(shell pkg-config --cflags gtest 2>/dev/null || echo "-I/opt/homebrew/Cellar/googletest/1.17.0/include")

GTEST_LIBS := $(shell \
	if pkg-config --exists gtest_main; then \
		pkg-config --libs gtest_main; \
	elif [ -f /usr/lib/libgtest.a ]; then \
		echo "-L/usr/lib -lgtest -lgtest_main -pthread"; \
	else \
		echo "-lgtest -lgtest_main -pthread"; \
	fi)

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

$(BUILD_TEST_DIR):
	@mkdir -p $(BUILD_TEST_DIR)

TEST_SOURCES := $(wildcard $(TEST_DIR)/*.cpp $(TEST_DIR)/**/*.cpp)
TEST_OBJECTS := $(patsubst $(TEST_DIR)/%.cpp,$(BUILD_TEST_DIR)/%.o,$(TEST_SOURCES))
TEST_EXECUTABLES := $(patsubst $(BUILD_TEST_DIR)/%.o,$(BUILD_TEST_DIR)/%,$(TEST_OBJECTS))

.PHONY: debug-test-vars
debug-test-vars:
	@echo "TEST_DIR = $(TEST_DIR)"
	@echo "BUILD_TEST_DIR = $(BUILD_TEST_DIR)"
	@echo "TEST_SOURCES = $(TEST_SOURCES)"
	@echo "TEST_OBJECTS = $(TEST_OBJECTS)"
	@echo "TEST_EXECUTABLES = $(TEST_EXECUTABLES)"

$(BUILD_BENCH_DIR):
	@mkdir -p $(BUILD_BENCH_DIR)

BENCH_SOURCES := $(wildcard $(BENCH_DIR)/*.cpp $(BENCH_DIR)/**/*.cpp)
BENCH_OBJECTS := $(patsubst $(BENCH_DIR)/%.cpp,$(BUILD_BENCH_DIR)/%.o,$(BENCH_SOURCES))
BENCH_EXECUTABLES := $(patsubst $(BUILD_BENCH_DIR)/%.o,$(BUILD_BENCH_DIR)/%,$(BENCH_OBJECTS))

.PHONY: debug-bench-vars
debug-bench-vars:
	@echo "BENCH_DIR = $(BENCH_DIR)"
	@echo "BUILD_BENCH_DIR = $(BUILD_BENCH_DIR)"
	@echo "BENCH_SOURCES = $(BENCH_SOURCES)"
	@echo "BENCH_OBJECTS = $(BENCH_OBJECTS)"
	@echo "BENCH_EXECUTABLES = $(BENCH_EXECUTABLES)"

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

# Install benchmark tools
.PHONY: install-bench-tools
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


.PHONY: install-dependencies
install-dependencies:
	@echo "📦 Installing pybind11 and numpy using uv..."
	@uv pip install --python $(PYTHON3) pybind11 numpy
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
	@echo "  PYTHON               = $(PYTHON3)"
	@echo "  Python version       = $(PYTHON_VERSION)"
	@echo ""
	@echo "📦 Python Includes"
	@echo "  PYTHON_INCLUDE       = $(PYTHON_INCLUDE)"
	@echo "  PYTHON_STDLIB        = $(PYTHON_STDLIB)"
	@echo "  PYTHON_DYNLOAD       = $(PYTHON_DYNLOAD)"
	@echo "  PYTHON_SITE_PACKAGES = $(PYTHON_SITE_PACKAGES)"
	@echo "  PYTHON_HOME          = $(PYTHON_HOME)"
	@echo "  PYTHON_CFLAGS        = $(PYTHON_CFLAGS)"
	@echo "  PYTHON_LDFLAGS       = $(PYTHON_LDFLAGS)"
	@echo ""
	@echo "🧱 Build dependencies"
	@if [ -n "$(PYBIND11_INCLUDE)" ]; then \
		echo "  PYBIND11_INCLUDE     = $(PYBIND11_INCLUDE)"; \
	else \
		echo "  PYBIND11_INCLUDE     = ❌ not found -- run \"make install-dependencies\""; \
		exit 1; \
	fi
	@if [ -n "$(NUMPY_INCLUDE)" ]; then \
		echo "  NUMPY_INCLUDE        = $(NUMPY_INCLUDE)"; \
	else \
		echo "  NUMPY_INCLUDE        = ❌ not found -- run \"make install-dependencies\""; \
		exit 1; \
	fi

	@echo "🔧 Tooling Checks"
	@which g++ >/dev/null 2>&1 || { echo "❌ g++ not found"; exit 1; }
	@which cmake >/dev/null 2>&1 || { echo "❌ cmake not found"; exit 1; }
	@which uv >/dev/null 2>&1 || { echo "❌ uv not found"; exit 1; }
	@echo "  ✓ g++ found"
	@echo "  ✓ cmake found"
	@echo "  ✓ uv found"
	@echo ""
	@echo "⚙️ Compiler"
	@echo "  CXX                  = $(CXX)"
	@echo "  CXXFLAGS             = $(CXXFLAGS)"
	@echo ""
	@echo "📚 Linking"
	@echo "  PYTHON_LIBS          = $(PYTHON_LIBS)"
	@echo "  RPATH_FLAGS          = $(RPATH_FLAGS)"

#-----------------------
# CI Build Target
#-----------------------

.PHONY: build-for-ci
build-for-ci: check-dependencies
	$(SRC_OBJECTS) $(TEST_EXECUTABLES) $(BENCH_EXECUTABLES)
	@echo "Source objects: $(words $(SRC_OBJECTS))"
	@echo "Test executables: $(words $(TEST_EXECUTABLES))"
	@echo "Benchmark executables: $(words $(BENCH_EXECUTABLES))"
	@echo "✅ Build completed."

#-----------------------
# Build with setup-tools
#-----------------------

# Build target that calls the setup.py script with UV venv activated
.PHONY: build
build: 
	@echo "Building mettagrid..."
	uv pip install -e .

#-----------------------
# Formatting
#-----------------------

# Check if the required formatting tools are installed
.PHONY: check-tools
check-tools:
	@echo "Checking for required formatting tools..."
	@which clang-format >/dev/null 2>&1 || \
		{ echo "clang-format is not installed. On macOS use 'brew install clang-format'"; \
		  echo "On Linux use 'apt-get install clang-format'"; \
		  echo "Or run 'make install-tools' on macOS"; exit 1; }
	@echo "All required formatting tools are installed."


# Format only C/C++ code and skip Cython files entirely
.PHONY: format
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
.PHONY: check-test-tools
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

# Compile individual test files
$(BUILD_TEST_DIR)/%.o: $(TEST_DIR)/%.cpp | $(BUILD_TEST_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link test executables with the mettagrid library
$(BUILD_TEST_DIR)/%: $(BUILD_TEST_DIR)/%.o $(SRC_OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $^ -o $@ $(GTEST_LIBS) $(PYTHON_LIBS) $(PYBIND11_LIBS) $(RPATH_FLAGS)

# Run C++ tests
.PHONY: test
test: check-test-tools check-dependencies $(SRC_OBJECTS) $(TEST_EXECUTABLES) debug-test-vars
	@echo "Running all C++ tests..."
	@[ -z "$(TEST_EXECUTABLES)" ] && echo "WARNING: No test executables found!" || true
	@for f in $(TEST_EXECUTABLES); do \
		if [ -f "$$f" ]; then \
			echo "Running $$f"; \
			PYTHONHOME="$(PYTHON_HOME)" \
			PYTHONPATH="$(PYTHON_STDLIB):$(PYTHON_DYNLOAD):$(PYTHON_SITE_PACKAGES)" \
			LD_LIBRARY_PATH="$(PYTHON_LIB_DIR):$$LD_LIBRARY_PATH" \
			DYLD_LIBRARY_PATH="$(PYTHON_LIB_DIR):$$DYLD_LIBRARY_PATH" \
			"$$f" --gtest_color=yes || exit 1; \
		else \
			echo "ERROR: Test executable $$f not found!"; \
			exit 1; \
		fi; \
	done

# Test a specific test file
.PHONY: test-%
test-%: check-test-tools check-dependencies $(BUILD_TEST_DIR)/%
	@echo "Running test $*..."
	@if [ -f "$(BUILD_TEST_DIR)/$*" ]; then \
		PYTHONHOME="$(PYTHON_HOME)" \
		PYTHONPATH="$(PYTHON_STDLIB):$(PYTHON_DYNLOAD):$(PYTHON_SITE_PACKAGES)" \
		LD_LIBRARY_PATH="$(PYTHON_LIB_DIR):$$LD_LIBRARY_PATH" \
		DYLD_LIBRARY_PATH="$(PYTHON_LIB_DIR):$$DYLD_LIBRARY_PATH" \
		"$(BUILD_TEST_DIR)/$*" --gtest_color=yes; \
	else \
		echo "ERROR: Test executable $(BUILD_TEST_DIR)/$* not found!"; \
		exit 1; \
	fi

.PHONY: test-python
test-python: check-venv
	@echo "Running python tests with coverage"
	pytest --cov=metta --cov-report=term-missing

#-----------------------
# Benchmarking
#-----------------------

# Check if benchmark tools are installed
.PHONY: check-bench-tools
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

# Compile individual benchmark files
$(BUILD_BENCH_DIR)/%.o: $(BENCH_DIR)/%.cpp | $(BUILD_BENCH_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(BENCH_CXXFLAGS) -c $< -o $@

# Link benchmark executables with the mettagrid library
$(BUILD_BENCH_DIR)/%: $(BUILD_BENCH_DIR)/%.o $(SRC_OBJECTS)
	$(CXX) $^ -o $@ $(BENCHMARK_LIBS) $(PYTHON_LIBS) $(PYBIND11_LIBS) $(RPATH_FLAGS)

# Run all benchmarks
.PHONY: benchmark
benchmark: check-bench-tools check-dependencies $(SRC_OBJECTS) $(BENCH_EXECUTABLES) debug-bench-vars
	@echo "Running all benchmarks..."
	@[ -z "$(BENCH_EXECUTABLES)" ] && echo "WARNING: No benchmark executables found!" || true
	@for f in $(BENCH_EXECUTABLES); do \
		if [ -f "$$f" ]; then \
			echo "Running $$f"; \
			PYTHONHOME="$(PYTHON_HOME)" \
			PYTHONPATH="$(PYTHON_STDLIB):$(PYTHON_DYNLOAD):$(PYTHON_SITE_PACKAGES)" \
			LD_LIBRARY_PATH="$(PYTHON_LIB_DIR):$$LD_LIBRARY_PATH" \
			DYLD_LIBRARY_PATH="$(PYTHON_LIB_DIR):$$DYLD_LIBRARY_PATH" \
			"$$f" || exit 1; \
		else \
			echo "ERROR: Benchmark executable $$f not found!"; \
			exit 1; \
		fi; \
	done


.PHONY: bench-json
bench-json: check-bench-tools check-dependencies $(SRC_OBJECTS) $(BENCH_EXECUTABLES) debug-bench-vars
	@echo "Running all benchmarks with JSON output..."
	@mkdir -p benchmark_output
	@[ -z "$(BENCH_EXECUTABLES)" ] && echo "WARNING: No benchmark executables found!" || true
	@for f in $(BENCH_EXECUTABLES); do \
		if [ -f "$$f" ]; then \
			echo "Running $$f with JSON output..."; \
			PYTHONHOME="$(PYTHON_HOME)" \
			PYTHONPATH="$(PYTHON_STDLIB):$(PYTHON_DYNLOAD):$(PYTHON_SITE_PACKAGES)" \
			LD_LIBRARY_PATH="$(PYTHON_LIB_DIR):$$LD_LIBRARY_PATH" \
			DYLD_LIBRARY_PATH="$(PYTHON_LIB_DIR):$$DYLD_LIBRARY_PATH" \
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
					echo "(MettaGrid) ✓ Preserving $$venv_name (UV virtual environment)"; \
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
		echo "(MettaGrid) ✓ No build directory to clean"; \
	fi

	@echo "(MettaGrid) Cleaning .so files from mettagrid directory..."
	@if [ -d "mettagrid" ]; then \
		found_files=$$(find mettagrid -name "*.so" -type f | wc -l); \
		if [ $$found_files -gt 0 ]; then \
			find mettagrid -name "*.so" -type f -delete; \
			echo "(MettaGrid) ✅ Removed $$found_files .so files from mettagrid"; \
		else \
			echo "(MettaGrid) ✓ No .so files found in mettagrid"; \
		fi; \
	else \
		echo "(MettaGrid) ✓ No mettagrid directory found, skipping .so cleanup"; \
	fi
	
	@echo "(MettaGrid) ✅ Clean completed successfully"

# Run format and test
.PHONY: all
all: format test
	@echo "All tasks completed."