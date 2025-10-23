.PHONY: help build pull push list list-metrics export diff clean delete

# Default target
help:
	@echo "Datadog Dashboard Management (Jsonnet-based)"
	@echo "=============================================="
	@echo ""
	@echo "Setup:"
	@echo "  make env           Load environment variables from .env"
	@echo ""
	@echo "Jsonnet Workflow (Recommended):"
	@echo "  1. vim components/ci.libsonnet     # Edit widget components"
	@echo "  2. vim dashboards/health.jsonnet   # Compose dashboard"
	@echo "  3. make build                      # Build JSON from Jsonnet"
	@echo "  4. make diff                       # Review changes"
	@echo "  5. make push                       # Upload to Datadog"
	@echo "  6. git add . && git commit         # Version control"
	@echo ""
	@echo "Build Commands:"
	@echo "  make build         Build all dashboards from Jsonnet"
	@echo "  make build-one     Build single dashboard (FILE=dashboards/name.jsonnet)"
	@echo ""
	@echo "Discovery:"
	@echo "  make list          List all dashboards in Datadog account"
	@echo "  make list-metrics  List available metrics (for building widgets)"
	@echo ""
	@echo "Dashboard Commands:"
	@echo "  make pull          Export all dashboards from Datadog"
	@echo "  make push          Upload all dashboards to Datadog"
	@echo "  make export ID=xxx Export specific dashboard by ID"
	@echo "  make delete ID=xxx Delete dashboard by ID (or IDS='id1 id2')"
	@echo "  make diff          Show git diff of dashboard changes"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         Remove generated JSON files"
	@echo ""
	@echo "Note: We use Jsonnet (like Grafonnet) for composable dashboards."
	@echo "      Edit .jsonnet files, not .json files!"

# Check for environment variables
check-env:
	@if [ -z "$$DD_API_KEY" ] || [ "$$DD_API_KEY" = "your_api_key_here" ]; then \
		echo "Error: DD_API_KEY not set or using placeholder value"; \
		echo "Run: source ./load_env.sh"; \
		exit 1; \
	fi
	@if [ -z "$$DD_APP_KEY" ] || [ "$$DD_APP_KEY" = "your_app_key_here" ]; then \
		echo "Error: DD_APP_KEY not set or using placeholder value"; \
		echo "Run: source ./load_env.sh"; \
		exit 1; \
	fi

# Load environment variables
env:
	@echo "To load environment variables, run:"
	@echo "  source ./load_env.sh"

# Build all dashboards from Jsonnet
build:
	@if [ ! -d dashboards ] || [ -z "$$(ls -A dashboards/*.jsonnet 2>/dev/null)" ]; then \
		echo "No .jsonnet files found in dashboards/"; \
		echo "See JSONNET_PROTOTYPE.md to get started"; \
		exit 1; \
	fi
	@echo "Building dashboards from Jsonnet..."
	@mkdir -p templates
	@for file in dashboards/*.jsonnet; do \
		base=$$(basename $$file .jsonnet); \
		echo "  Building $$base..."; \
		jsonnet $$file > templates/$$base.json || exit 1; \
	done
	@echo ""
	@echo "✓ Built $$(ls dashboards/*.jsonnet | wc -l | tr -d ' ') dashboard(s)"
	@echo ""
	@echo "Next steps:"
	@echo "  - Review: ls -l templates/"
	@echo "  - Diff: make diff"
	@echo "  - Push: make push"

# Build single dashboard
build-one:
	@if [ -z "$(FILE)" ]; then \
		echo "Error: Please specify FILE"; \
		echo "Usage: make build-one FILE=dashboards/my_dashboard.jsonnet"; \
		exit 1; \
	fi
	@if [ ! -f "$(FILE)" ]; then \
		echo "Error: File not found: $(FILE)"; \
		exit 1; \
	fi
	@base=$$(basename $(FILE) .jsonnet); \
	echo "Building $$base..."; \
	jsonnet $(FILE) > templates/$$base.json && \
	echo "✓ Built templates/$$base.json"

# List all dashboards
list: check-env
	@echo "Fetching dashboard list from Datadog..."
	./scripts/fetch_dashboards.py --format=summary

# List available metrics
list-metrics: check-env
	@echo "Fetching available metrics from Datadog..."
	./scripts/list_metrics.py

# Pull all dashboards (export to JSON)
pull: check-env
	@echo "Exporting all dashboards from Datadog..."
	./scripts/batch_export.py
	@echo ""
	@echo "✓ Dashboards exported to templates/"
	@echo ""
	@echo "Next steps:"
	@echo "  - Review: ls -l templates/"
	@echo "  - Edit: vim templates/my_dashboard.json"
	@echo "  - Push: make push"

# Export specific dashboard by ID
export: check-env
	@if [ -z "$(ID)" ]; then \
		echo "Error: Please specify dashboard ID"; \
		echo "Usage: make export ID=abc-123-def"; \
		exit 1; \
	fi
	@echo "Exporting dashboard $(ID)..."
	./scripts/export_dashboard.py $(ID) > templates/dashboard_$(ID).json
	@echo "✓ Exported to templates/dashboard_$(ID).json"

# Push all dashboards (upload to Datadog)
push: check-env
	@if [ ! -d templates ] || [ -z "$$(ls -A templates/*.json 2>/dev/null)" ]; then \
		echo "Error: No JSON files found in templates/"; \
		echo "Run: make pull"; \
		exit 1; \
	fi
	@echo "Pushing dashboards to Datadog..."
	./scripts/push_dashboard.py templates/*.json
	@echo ""
	@echo "✓ Push complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  - Verify in Datadog UI"
	@echo "  - Commit: git add templates/ && git commit"

# Show diff of dashboard changes
diff:
	@if [ ! -d templates ] || [ -z "$$(ls -A templates/*.json 2>/dev/null)" ]; then \
		echo "No dashboard files to diff"; \
		exit 0; \
	fi
	@echo "Dashboard changes:"
	@echo "=================="
	@git diff templates/ || echo "No changes detected"

# Dry run - show what would be pushed
dry-run: check-env
	@if [ ! -d templates ] || [ -z "$$(ls -A templates/*.json 2>/dev/null)" ]; then \
		echo "Error: No JSON files found in templates/"; \
		echo "Run: make pull"; \
		exit 1; \
	fi
	@echo "Dry run - showing what would be pushed..."
	./scripts/push_dashboard.py templates/*.json --dry-run

# Delete dashboard(s) by ID
delete: check-env
	@if [ -z "$(ID)$(IDS)" ]; then \
		echo "Error: Please specify ID or IDS"; \
		echo "Usage: make delete ID=abc-123-def"; \
		echo "   or: make delete IDS='id1 id2 id3'"; \
		exit 1; \
	fi
	@if [ -n "$(IDS)" ]; then \
		echo "Deleting dashboard(s): $(IDS)"; \
		./scripts/delete_dashboard.py $(IDS); \
	else \
		echo "Deleting dashboard: $(ID)"; \
		./scripts/delete_dashboard.py $(ID); \
	fi

# Clean up exported files
clean:
	@echo "Cleaning up..."
	rm -f templates/*.json
	rm -f dashboards.json dashboards_detailed.json
	@echo "✓ Cleaned up JSON files"
	@echo ""
	@echo "Note: Terraform files not removed (see dashboards/)"
