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

## Observability (Tournament Jobs)

OpenTelemetry traces are exported via OTLP to the Datadog Agent. Logs are emitted as JSON with `trace_id` and
`span_id` fields for correlation.

Environment variables used by the tournament/job runner path:

- `OTEL_EXPORTER_OTLP_ENDPOINT` (default `http://localhost:4318`)
- `OTEL_SERVICE_NAME`
- `OTEL_RESOURCE_ATTRIBUTES` (set `deployment.environment` and `service.version`)
- `LOG_JSON=true`

### How to Verify

1. Run locally:
   - `metta observatory local-k8s setup`
   - `metta observatory up`
   - Submit a test job:
     ```bash
     uv run python app_backend/scripts/submit_test_jobs.py \
       --policy-uri metta://policy/<your-policy-name>
     ```
2. Confirm traces in Datadog APM:
   - Filter for services `tournament-commissioner` and `episode-runner`
   - Look for spans named `tournament.job.enqueue` and `tournament.job.run`
3. Confirm log correlation:
   - Open a `tournament.job.run` span and use the Logs tab to verify related logs include matching `trace_id` and
     `span_id`
