# Observatory Backend

Backend API for https://observatory.softmax-research.net/

## Local Development

```bash
metta observatory --help
```

## Production

Deployed to EKS via Helm chart at `devops/charts/observatory-backend/`.

- Host: `api.observatory.softmax-research.net`
- Image built by `.github/workflows/build-app-backend-image.yml`
- Database: RDS Postgres (credentials in k8s secret `observatory-backend-env`)
