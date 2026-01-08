# Observatory

Frontend for https://observatory.softmax-research.net/

## Development

**Frontend only (against prod API):**

```bash
metta observatory frontend --backend [prod|local]
```

## Production

Deployed to EKS via Helm chart at `devops/charts/observatory/`.

- Host: `observatory.softmax-research.net`
- Image built by `.github/workflows/build-observatory-image.yml`
