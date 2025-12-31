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

### Webhook Service

The GitHub-Asana webhook service is integrated as a route at `/webhooks/github`.

**Configuration:**
- Environment variables set via Helm `extra_args` in CI/CD:
  - `USE_AWS_SECRETS=true` (enables Secrets Manager access)
  - `ASANA_WORKSPACE_GID` (from GitHub Variables)
  - `ASANA_PROJECT_GID` (from GitHub Variables)

**Required GitHub Variables:**
- `vars.ASANA_WORKSPACE_GID`: Asana workspace GID
- `vars.ASANA_PROJECT_GID`: Asana project GID for bugs

**Required AWS Secrets Manager secrets** (accessed via IRSA role `observatory-backend`):
- `github/webhook-secret`: GitHub webhook secret for signature verification
- `asana/access-token` or `asana/api-key`: Asana Personal Access Token
- `asana/atlas_app`: Asana OAuth app credentials (JSON)
- `github/token`: GitHub token for PR description updates
