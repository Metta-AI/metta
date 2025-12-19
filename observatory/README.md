# Observatory

Frontend for https://observatory.softmax-research.net/

## Setup

```bash
metta install nodejs
```

## Development

**Frontend only (against prod API):**

```bash
metta local observatory --backend prod
```

**Full local stack (frontend + backend + database):**

Terminal 1 - Backend:

```bash
docker compose -f app_backend/docker-compose.dev.yml up
```

Terminal 2 - Frontend:

```bash
metta local observatory
```

Open http://localhost:5173

## Production

Deployed to EKS via Helm chart at `devops/charts/observatory/`.

- Host: `observatory.softmax-research.net`
- Image built by `.github/workflows/build-observatory-image.yml`
