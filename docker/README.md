# Policy Evaluator Docker Service

A Docker-based service that wraps `sim.py` to evaluate policies from WandB artifacts.

## Quick Start

1. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your WandB API key and policy URI
   ```

2. **Build and run with Docker Compose**:
   ```bash
   # Development mode (with source code mounted)
   docker-compose up policy-evaluator
   
   # Production mode (self-contained image)
   docker-compose up policy-evaluator-prod
   ```

3. **Or run directly with Docker**:
   ```bash
   # Build the image
   docker build -t metta-policy-evaluator .
   
   # Run evaluation
   docker run \
     -e WANDB_API_KEY=your_key \
     -e POLICY_URI=wandb://metta-ai/metta-research/model/test_run:latest \
     -v $(pwd)/data:/app/data \
     metta-policy-evaluator
   ```

## Environment Variables

### Required
- `WANDB_API_KEY`: Your WandB API key for authentication
- `POLICY_URI`: URI of the policy to evaluate

### Optional
- `WANDB_ENTITY`: WandB organization (default: metta-ai)
- `WANDB_PROJECT`: WandB project name (default: metta-research)
- `SIMULATION_SUITE`: Evaluation suite to run (default: navigation)
- `DEVICE`: Compute device (default: cpu)
- `DATA_DIR`: Data directory for artifacts (default: /app/data)
- `RUN_NAME`: Custom run name (auto-generated if not provided)
- `OUTPUT_URI`: Where to upload results (optional)

## Policy URI Formats

The service supports multiple policy URI formats:

1. **WandB artifacts**: `wandb://entity/project/model/run_name:version`
2. **Local files**: `file://./path/to/policy.pt`
3. **S3 buckets**: `s3://bucket/path/to/policy.pt` (future)

## Output Format

The service outputs structured JSON with evaluation results:

```json
{
  "simulation_suite": "navigation",
  "policies": [
    {
      "policy_uri": "wandb://metta-ai/metta-research/model/test_run:latest",
      "checkpoints": [
        {
          "name": "test_run",
          "uri": "wandb://metta-ai/metta-research/model/test_run:latest",
          "metrics": {...},
          "replay_url": "..."
        }
      ]
    }
  ]
}
```

## Error Handling

If evaluation fails, the service outputs error information:

```json
{
  "error": "Error message",
  "traceback": "Full traceback...",
  "policy_uri": "wandb://..."
}
```

## Development

### Building Different Targets

```bash
# Development image (includes dev dependencies)
docker build --target development -t metta-policy-evaluator:dev .

# Production image (minimal, optimized)
docker build --target production -t metta-policy-evaluator:prod .
```

### Local Testing

```bash
# Set up environment
export WANDB_API_KEY=your_key
export POLICY_URI=wandb://metta-ai/metta-research/model/test_run:latest

# Run evaluation
docker-compose up policy-evaluator
```

## Architecture

- **docker-entrypoint.py**: Main entry point, handles environment setup
- **policy_evaluator.py**: Core evaluation logic, wraps sim.py
- **Dockerfile**: Multi-stage build for development and production
- **docker-compose.yml**: Local development and testing setup

## Integration with EKS

This Docker image is designed to be deployed on EKS as:
1. **Kubernetes Jobs**: One-time evaluation jobs
2. **Deployments**: Long-running services for webhook processing
3. **CronJobs**: Periodic evaluation of new policies

See the main README for EKS deployment instructions.