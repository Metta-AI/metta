# Training and Eval Data Collection

## S3-Based Architecture

The training and eval collectors derive metrics directly from S3 metadata. There is no JSON file dependency.

### Training Collector

The `TrainingCollector` discovers training jobs by listing:

- `s3://softmax-train-dir/.job_metadata/*/`

It parses the following metadata files:

- `heartbeat_file`: Last heartbeat timestamp (indicates job activity)
- `restart_count`: Number of restarts (instability indicator)
- `termination_reason`: Job termination status

**Heuristic Success Logic:**

- Success if `termination_reason` contains "0" or "completed"
- Failure if "1" or "error" or missing heartbeat for > 30 minutes
- Placeholder values:
  - `hearts = 1.0` if success else `0.0` (until real source available)
  - `sps = 0` (until real SPS source available)
  - `checkpoint success = 1.0` if success else `0.0`

**Sentinel Metrics:**

- `metta.infra.cron.training.data_missing`: Set to `1.0` when no training data found in S3, `0.0` when data is available

### Eval Collector

The `EvalCollector` searches S3 recursively for eval artifacts:

- `s3://softmax-train-dir/**/eval*/`
- `s3://rain-artifacts-751442549699-us-west-2/**/eval*/`

**Current Status:**

- No eval artifacts exist yet in S3
- Collector emits placeholder zeros and `data_missing = 1.0`
- TODO: Implement parsing when eval artifacts are discovered

**Sentinel Metrics:**

- `metta.infra.cron.eval.data_missing`: Set to `1.0` when no eval data found in S3, `0.0` when data is available

### AWS Credentials

Collectors use boto3 to access S3. Credentials are provided via:

- IAM role (in Kubernetes via service account)
- Environment variables (for local testing)
- AWS credentials chain

### Future Enhancements

When real training/eval summaries become available:

1. Update collectors to parse actual metrics (hearts, SPS, heart_delta_pct, etc.)
2. Remove placeholder logic
3. Update dashboards to reflect real metric sources
