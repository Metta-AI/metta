# Makefile for code formatting and linting

.PHONY: help format check-tools install-tools lint

# Default target when just running 'make'
help:
	@echo "Available targets:"
	@echo "  format      - Format C++/C files"
	@echo "  check-tools - Check if required formatting tools are installed"
	@echo "  install-tools - Install required formatting tools (macOS only)"
	@echo "  all         - Run format and lint"

# Check if the required tools are installed
check-tools:
	@echo "Checking for required tools..."
	@which clang-format >/dev/null 2>&1 || \
		{ echo "clang-format is not installed. On macOS use 'brew install clang-format'"; \
		  echo "On Linux use 'apt-get install clang-format'"; \
		  echo "Or run 'make install-tools' on macOS"; exit 1; }
	@echo "All required tools are installed."

# Install tools on macOS
install-tools:
	@echo "Installing required tools..."
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

all: format
	@echo "All tasks completed."