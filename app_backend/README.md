# Observatory Backend

Backend API for https://observatory.softmax-research.net/

## Local Development

```bash
docker compose -f docker-compose.dev.yml up
```

This starts:

- Postgres on port 5432
- Backend API on port 8000 (with migrations auto-applied)

Edit code locally, then restart to pick up changes:

```bash
docker compose -f docker-compose.dev.yml restart observatory-backend
```

## Production

Deployed to EKS via Helm chart at `devops/charts/observatory-backend/`.

- Host: `api.observatory.softmax-research.net`
- Image built by `.github/workflows/build-app-backend-image.yml`
- Database: RDS Postgres (credentials in k8s secret `observatory-backend-env`)
