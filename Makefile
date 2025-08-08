.PHONY: help all dev test clean install pytest test-setup

override METTA_TEST_PROFILE ?= softmax

# Default target when just running 'make'
help:
	@echo "Available targets:"
	@echo " dev - Prepare the dev environment"
	@echo " test - Run all unit tests with coverage"
	@echo " pytest - Run all unit tests"
	@echo " test-setup - Run setup integration tests"
	@echo " all - Run dev and test"
	@echo " clean - Remove cmake build artifacts and temporary files"


# Clean cmake build artifacts
clean:
	@echo "(Metta) Running clean command..."
	uv run  --active metta clean

# Dev all project dependencies and external components
dev:
	@echo "Running full devops/setup_dev.sh installation script..."
	@bash devops/setup_dev.sh

test:
	@echo "Running python tests with coverage"
	uv run --active metta test -n auto --cov=metta --cov-report=term-missing --durations=10

test-setup:
	@echo "Running setup integration tests..."
	METTA_TEST_ENV=1 \
	METTA_TEST_SETUP=1 \
	METTA_TEST_PROFILE=$(METTA_TEST_PROFILE) \
	AWS_SSO_NONINTERACTIVE=1 \
		uv run --active metta test tests/setup -v -n auto

install:
	@echo "Installing package in editable mode..."
	uv sync --inexact --active

pytest: install
	@echo "Running Python tests..."
	uv run --active metta test -n auto

all: dev test
