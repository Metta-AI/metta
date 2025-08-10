.PHONY: help all dev test clean install pytest test-setup

override METTA_TEST_PROFILE ?= softmax

# Default target when just running 'make'
help:
	@echo "Available targets:"
	@echo " dev - Prepare the dev environment"
	@echo " test - Run all unit tests with coverage"
	@echo " pytest - Run unit tests without benchmarks (parallel)"
	@echo " bench - Run all unit tests (serial)"
	@echo " test-setup - Run setup integration tests"
	@echo " all - Run dev and test"
	@echo " clean - Remove cmake build artifacts and temporary files"


# Clean cmake build artifacts
clean:
	@echo "(Metta) Running clean command..."
	uv run --active --frozen metta clean

# Dev all project dependencies and external components
dev:
	@echo "Running full devops/setup_dev.sh installation script..."
	@bash devops/setup_dev.sh

test:
	@echo "Running python tests with coverage"
	uv run --active --frozen metta test -n auto --cov=metta --cov-report=term-missing --durations=10

test-setup:
	@echo "Running setup integration tests..."
	METTA_TEST_ENV=1 \
	METTA_TEST_SETUP=1 \
	METTA_TEST_PROFILE=$(METTA_TEST_PROFILE) \
	AWS_SSO_NONINTERACTIVE=1 \
		uv run --active --frozen metta test tests/setup -v -n auto

install:
	@echo "Installing package in editable mode..."
	uv sync --active --frozen

pytest: install
	@echo "Running Python tests..."
	uv run --active --frozen -m pytest --benchmark-disable

bench:
	uv run --active --frozen -m pytest -n 0 -k benchmark

all: dev test
