.PHONY: help list run-all run-navigation run-object-use run-nav-memory-sequence validate clean

# Default target
help:
	@echo "Available commands:"
	@echo "  list                    - List all available recipes"
	@echo "  run-all                 - Run all stable recipes"
	@echo "  run-navigation          - Run navigation recipe"
	@echo "  run-object-use          - Run object use recipe"
	@echo "  run-nav-memory-sequence - Run nav memory sequence recipe (experimental)"
	@echo "  validate                - Validate all recipe configurations"
	@echo "  clean                   - Clean up temporary files"

# List all recipes
list:
	@python run_recipe.py --list

# Run all stable recipes
run-all: run-navigation run-object-use

# Run individual recipes
run-navigation:
	@echo "Running navigation recipe..."
	@python run_recipe.py navigation

run-object-use:
	@echo "Running object use recipe..."
	@python run_recipe.py object_use

run-nav-memory-sequence:
	@echo "Running nav memory sequence recipe (experimental)..."
	@python run_recipe.py nav_memory_sequence

# Validate all recipes
validate:
	@echo "Validating all recipe configurations..."
	@python run_recipe.py navigation --dry-run
	@python run_recipe.py object_use --dry-run
	@python run_recipe.py nav_memory_sequence --dry-run

# Clean up
clean:
	@echo "Cleaning up temporary files..."
	@find . -name "*.tmp" -delete
	@find . -name "*.log" -delete
