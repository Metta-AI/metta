# Observatory Backend

Backend API for https://observatory.softmax-research.net/

## Local Development

```bash
metta observatory backend [up|build|down|restart|logs]
```

This is a wrapper around `docker compose` operating on the `docker-compose.dev.yml` in this folder.

`metta observatory backend up` starts:

- Postgres on port 5432
- Backend API on port 8000 (with migrations auto-applied)

## Production

Deployed to EKS via Helm chart at `devops/charts/observatory-backend/`.

- Host: `api.observatory.softmax-research.net`
- Image built by `.github/workflows/build-app-backend-image.yml`
- Database: RDS Postgres (credentials in k8s secret `observatory-backend-env`)
