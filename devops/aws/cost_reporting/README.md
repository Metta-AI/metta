AWS Cost Reporting 

Overview
- Collects AWS Cost Explorer data into DuckDB
- Runs simple trend/anomaly analyses
- Generates HTML and CSV reports

CLI
- Collect: `uv run aws-cost collect --start 2025-01-01 --end 2025-01-31`
- Analyze: `uv run aws-cost analyze`
- Report:  `uv run aws-cost report --out devops/aws/cost_reporting/out`

Config (env with prefix `AWS_COST_`)
- `AWS_REGION`: AWS region (default: us-east-1)
- `AWS_COST_TAG_KEYS`: Comma-separated tag keys (optional)
- `AWS_COST_S3_BUCKET`: If set, will be used later for S3 storage

Notes
- This is v1 skeleton: collectors write to a local DuckDB at `devops/aws/cost_reporting/data/cost.duckdb`.
- Next PRs will add S3-backed Parquet, richer templates, Slack/email delivery, and multi-account assumptions.

