# Metta Sweep Docker Testing Environment
# Provides idempotent operations for managing distributed sweep testing

.PHONY: help build up down clean test test-basic test-intermediate test-advanced logs status shell

# Configuration
COMPOSE_FILE = docker-compose.yml
PROJECT_NAME = metta-sweep-test
DOCKER_BUILDKIT = 1

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[0;33m
BLUE = \033[0;34m
NC = \033[0m # No Color

# Help target
help: ## Show this help message
	@echo "$(BLUE)Metta Sweep Docker Testing Environment$(NC)"
	@echo "====================================="
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Test Complexity Levels:$(NC)"
	@echo "  $(YELLOW)basic$(NC)        - Quick validation (2 rollouts, 5 min)"
	@echo "  $(YELLOW)intermediate$(NC) - Medium complexity (5 rollouts, 10 min)"
	@echo "  $(YELLOW)advanced$(NC)     - Stress testing (10 rollouts, 20 min)"

# Build targets
build: ## Build all Docker images with aggressive caching
	@echo "$(BLUE)Building Metta sweep test environment...$(NC)"
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) build --parallel
	@echo "$(GREEN)✓ Build completed$(NC)"

build-no-cache: ## Build without cache (clean build)
	@echo "$(BLUE)Building without cache...$(NC)"
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) build --no-cache --parallel
	@echo "$(GREEN)✓ Clean build completed$(NC)"

# Environment management
up: ## Start the distributed testing environment
	@echo "$(BLUE)Starting distributed test environment...$(NC)"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) up -d
	@echo "$(GREEN)✓ Environment started$(NC)"
	@echo ""
	@echo "$(YELLOW)Container Status:$(NC)"
	@make --no-print-directory status
	@echo ""
	@echo "$(YELLOW)Monitoring:$(NC) http://localhost:3000 (admin/metta123)"

down: ## Stop and remove all containers
	@echo "$(BLUE)Stopping test environment...$(NC)"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) down
	@echo "$(GREEN)✓ Environment stopped$(NC)"

restart: down up ## Restart the entire environment

# Testing targets
test: up ## Run all tests (basic -> intermediate -> advanced)
	@echo "$(BLUE)Running complete test suite...$(NC)"
	@docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) logs -f test-runner

test-basic: up ## Run only basic tests
	@echo "$(BLUE)Running basic distributed sweep tests...$(NC)"
	@docker exec metta-test-runner /home/metta/metta/test-scripts/run-single-test.sh basic

test-intermediate: up ## Run only intermediate tests
	@echo "$(BLUE)Running intermediate distributed sweep tests...$(NC)"
	@docker exec metta-test-runner /home/metta/metta/test-scripts/run-single-test.sh intermediate

test-advanced: up ## Run only advanced tests
	@echo "$(BLUE)Running advanced distributed sweep tests...$(NC)"
	@docker exec metta-test-runner /home/metta/metta/test-scripts/run-single-test.sh advanced

# Debugging and inspection
status: ## Show container status and health
	@echo "$(YELLOW)Container Status:$(NC)"
	@docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) ps
	@echo ""
	@echo "$(YELLOW)Health Checks:$(NC)"
	@docker exec metta-sweep-master python3 -c "import torch; print(f'Master - CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "❌ Master not ready"
	@docker exec metta-sweep-worker python3 -c "import torch; print(f'Worker - CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "❌ Worker not ready"
	@docker exec metta-sweep-worker ping -c 1 sweep-master >/dev/null 2>&1 && echo "✓ Network connectivity OK" || echo "❌ Network issue"

logs: ## Show logs from all containers
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) logs -f

logs-master: ## Show logs from master node only
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) logs -f sweep-master

logs-worker: ## Show logs from worker node only
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) logs -f sweep-worker

logs-test: ## Show logs from test runner only
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) logs -f test-runner

shell: ## Open shell in master container for debugging
	@echo "$(BLUE)Opening shell in master container...$(NC)"
	docker exec -it metta-sweep-master bash

shell-worker: ## Open shell in worker container for debugging
	@echo "$(BLUE)Opening shell in worker container...$(NC)"
	docker exec -it metta-sweep-worker bash

shell-test: ## Open shell in test runner container
	@echo "$(BLUE)Opening shell in test runner container...$(NC)"
	docker exec -it metta-test-runner bash

# Results and reporting
results: ## Show test results summary
	@echo "$(BLUE)Test Results Summary:$(NC)"
	@if docker exec metta-test-runner test -f /home/metta/test-results/test-summary.json; then \
		docker exec metta-test-runner cat /home/metta/test-results/test-summary.json | python3 -m json.tool; \
	else \
		echo "$(RED)No test results found$(NC)"; \
	fi

results-copy: ## Copy test results to local machine
	@echo "$(BLUE)Copying test results...$(NC)"
	@mkdir -p ./test-results
	docker cp metta-test-runner:/home/metta/test-results/. ./test-results/
	docker cp metta-test-runner:/home/metta/test-logs/. ./test-results/logs/
	@echo "$(GREEN)✓ Results copied to ./test-results/$(NC)"

# Cleanup targets
clean: down ## Stop containers and remove volumes
	@echo "$(BLUE)Cleaning up test environment...$(NC)"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) down -v
	docker system prune -f --filter label=com.docker.compose.project=$(PROJECT_NAME)
	@echo "$(GREEN)✓ Cleanup completed$(NC)"

clean-all: clean ## Complete cleanup including images
	@echo "$(BLUE)Performing complete cleanup...$(NC)"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) down -v --rmi all
	docker system prune -a -f --filter label=com.docker.compose.project=$(PROJECT_NAME)
	@echo "$(GREEN)✓ Complete cleanup done$(NC)"

# Utility targets
monitoring: ## Open monitoring dashboard
	@echo "$(BLUE)Opening monitoring dashboard...$(NC)"
	@if command -v open >/dev/null 2>&1; then \
		open http://localhost:3000; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open http://localhost:3000; \
	else \
		echo "$(YELLOW)Open http://localhost:3000 in your browser$(NC)"; \
	fi

ps: status ## Alias for status

# Validation targets
validate: ## Validate Docker setup and requirements
	@echo "$(BLUE)Validating environment...$(NC)"
	@command -v docker >/dev/null 2>&1 || (echo "$(RED)❌ Docker not found$(NC)" && exit 1)
	@command -v docker-compose >/dev/null 2>&1 || (echo "$(RED)❌ Docker Compose not found$(NC)" && exit 1)
	@docker info >/dev/null 2>&1 || (echo "$(RED)❌ Docker daemon not running$(NC)" && exit 1)
	@echo "$(GREEN)✓ Environment validation passed$(NC)"

# Default target
.DEFAULT_GOAL := help
