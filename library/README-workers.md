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
# REDIS_PASSWORD=your-password  # Optional
```

### Job Queue Features

- **Automatic Retries** - Failed jobs retry with exponential backoff
- **Rate Limiting** - Respects external API limits (arXiv, OpenAI)
- **Job Persistence** - Jobs survive worker restarts
- **Monitoring** - Real-time queue statistics

## 🏗️ Production Deployment (EKS)

For Slava - here's what you'll need for the EKS setup:

### Docker Image

```dockerfile
# Dockerfile.workers
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY dist/ ./dist/
CMD ["node", "dist/lib/workers/worker-manager.js"]
```

### Helm Values

```yaml
# values.yaml
redis:
  enabled: true
  auth: { enabled: false }
  master: { persistence: { size: "20Gi" } }

workers:
  image: "library-workers:latest"
  replicas: 2
  env:
    REDIS_HOST: "redis-master"
    DATABASE_URL: "${DATABASE_URL}"
    OPENAI_API_KEY: "${OPENAI_API_KEY}"
    ANTHROPIC_API_KEY: "${ANTHROPIC_API_KEY}"
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

