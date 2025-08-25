# Background Workers with BullMQ + Redis

This document explains how to set up and run background workers for processing institutions, authors, and LLM abstracts using BullMQ and Redis.

## 🚀 Quick Start

### 1. Install and Start Redis

**On macOS:**

```bash
brew install redis
brew services start redis
```

**On Ubuntu/Debian:**

```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
```

**Using Docker:**

```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

### 2. Start Background Workers

```bash
# Start all workers (blocking process)
pnpm workers

# Or start workers in development mode (auto-restart on changes)
pnpm worker:dev
```

### 3. Test Job Processing

Create a post with an arXiv URL in the web interface. You should see jobs being queued and processed in the worker logs.

## 📊 Architecture

```
Web App (Amplify) → BullMQ Queues → Workers → Database
                           ↓
                       Redis Store
```

### Queues

- **`institution-extraction`** - Processes arXiv papers to extract institution data
- **`llm-processing`** - Generates LLM abstracts for papers
- **`auto-tagging`** - Auto-tags papers based on content
- **`author-extraction`** - Extracts and processes author information

### Workers

- **Institution Worker** - 2 concurrent jobs, rate limited to 10 jobs/minute
- **LLM Worker** - 1 concurrent job (resource intensive), 5 jobs/minute
- **Tagging Worker** - 3 concurrent jobs, 20 jobs/minute

## 🔧 Configuration

### Environment Variables

Add to your `.env.local`:

```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your-password  # Required for production

# Database
DATABASE_URL=your-database-url

# LLM Service
ANTHROPIC_API_KEY=your-anthropic-key

# Adobe PDF Services
ADOBE_CLIENT_ID=your-adobe-client-id
ADOBE_CLIENT_SECRET=your-adobe-client-secret

# AWS Services
AWS_REGION=us-east-1
AWS_S3_BUCKET=your-s3-bucket-name

# Asana Integration (optional)
ASANA_API_KEY=your-asana-key
ASANA_PAPERS_PROJECT_ID=your-project-id
ASANA_WORKSPACE_ID=your-workspace-id
ASANA_PAPER_LINK_FIELD_ID=your-field-id
ASANA_ARXIV_ID_FIELD_ID=your-field-id
ASANA_ABSTRACT_FIELD_ID=your-field-id
```

### Job Queue Features

- **Automatic Retries** - Failed jobs retry with exponential backoff
- **Rate Limiting** - Respects external API limits (arXiv, Anthropic, Adobe)
- **Job Persistence** - Jobs survive worker restarts
- **Monitoring** - Real-time queue statistics

## 🏗️ Production Deployment (EKS)

For Slava - here's what you'll need for the EKS setup:

### Docker Image

```dockerfile
# Dockerfile.workers
FROM node:20-alpine
WORKDIR /app
RUN npm install -g pnpm
COPY package*.json pnpm-lock.yaml ./
RUN pnpm ci --frozen-lockfile --prod
COPY src/ ./src/
CMD ["pnpm", "workers"]
```

### Helm Values

```yaml
# values.yaml
redis:
  enabled: true
  auth: { enabled: true, password: "${REDIS_PASSWORD}" }
  master: { persistence: { size: "20Gi" } }

workers:
  image: "library-workers:latest"
  replicas: 2
  env:
    # Redis Configuration
    REDIS_HOST: "redis-master"
    REDIS_PASSWORD: "${REDIS_PASSWORD}"
    # Database
    DATABASE_URL: "${DATABASE_URL}"
    # LLM Service
    ANTHROPIC_API_KEY: "${ANTHROPIC_API_KEY}"
    # Adobe PDF Services
    ADOBE_CLIENT_ID: "${ADOBE_CLIENT_ID}"
    ADOBE_CLIENT_SECRET: "${ADOBE_CLIENT_SECRET}"
    # AWS Services
    AWS_REGION: "us-east-1"
    AWS_S3_BUCKET: "${AWS_S3_BUCKET}"
    # Asana Integration
    ASANA_API_KEY: "${ASANA_API_KEY}"
    ASANA_PAPERS_PROJECT_ID: "${ASANA_PAPERS_PROJECT_ID}"
    ASANA_WORKSPACE_ID: "${ASANA_WORKSPACE_ID}"
    ASANA_PAPER_LINK_FIELD_ID: "${ASANA_PAPER_LINK_FIELD_ID}"
    ASANA_ARXIV_ID_FIELD_ID: "${ASANA_ARXIV_ID_FIELD_ID}"
    ASANA_ABSTRACT_FIELD_ID: "${ASANA_ABSTRACT_FIELD_ID}"
  resources:
    requests: { cpu: "250m", memory: "512Mi" }
    limits: { cpu: "1000m", memory: "2Gi" }
```

### Monitoring

Consider adding:

- **BullMQ Dashboard** - Web UI for queue monitoring
- **Prometheus metrics** - For alerting on job failures
- **Log aggregation** - Centralized worker logs

## 🐛 Troubleshooting

### Redis Connection Issues

```bash
# Test Redis connection
redis-cli ping
# Should return "PONG"
```

### Worker Not Processing Jobs

1. Check Redis is running: `redis-cli ping`
2. Check worker logs for errors
3. Verify environment variables are set
4. Test queue connection manually

### Performance Tuning

- **Increase concurrency** for non-API-limited jobs
- **Adjust rate limits** based on external API quotas
- **Monitor memory usage** especially for LLM workers

## 📈 Monitoring Commands

```bash
# Check queue stats
redis-cli
> KEYS bull:*:waiting
> LLEN bull:institution-extraction:waiting

# View job details
> HGETALL bull:institution-extraction:1
```

## 🔄 Migration from setImmediate()

The previous approach used `setImmediate()` which doesn't work in serverless environments:

```typescript
// ❌ Old approach (doesn't work on Amplify)
setImmediate(async () => {
  await processArxivInstitutionsAsync(paperId, arxivUrl);
});

// ✅ New approach (works everywhere)
await JobQueueService.queueInstitutionExtraction(paperId, arxivUrl);
```

This provides:

- **Reliability** - Jobs persist across restarts
- **Observability** - Monitor job progress and failures
- **Scalability** - Multiple workers, rate limiting
- **Error Handling** - Automatic retries with backoff
