# Policy Analysis Pipeline (Observatory API)

This pipeline analyzes the highest performing policies from the observatory database using the API. It extracts top policies, runs comprehensive evaluations, and performs factor analysis to understand policy performance dimensions.

## Quick Start

1. **Authenticate with Observatory**

```bash
python devops/observatory_login.py
```

2. **Run the Pipeline**

```bash
./tools/policy_analysis_pipeline.py
```

- Extracts the top 100 policies by average reward
- Runs comprehensive evaluations
- Performs factor analysis and generates visualizations

## Usage

- Change the number of policies:
  ```bash
  ./tools/policy_analysis_pipeline.py --num-policies 50
  ```
- Change output directory:
  ```bash
  ./tools/policy_analysis_pipeline.py --output-dir ./my_results
  ```
- Skip stages (e.g., only analyze):
  ```bash
  ./tools/policy_analysis_pipeline.py --skip-stage extract --skip-stage evaluate
  ```

## Output Structure

```
policy_analysis_results/
├── extracted_data/
│   ├── top_policies.csv/json
│   ├── policy_evaluations.csv/json
│   ├── environments.csv/json
│   └── summary_stats.json
├── evaluations/
│   ├── evaluation_results.json
│   └── performance_matrix.csv
├── analysis/
│   ├── factor_analysis_results.json
│   ├── optimal_components.json
│   ├── policy_clusters.csv
│   └── visualizations/
└── analysis_report.md
```

## How It Works

- **Extract**: Finds top policies by average reward using the observatory API (no direct DB connection needed)
- **Evaluate**: Runs evaluations for these policies
- **Analyze**: Factor analysis and clustering on the results

## Requirements
- Python 3.9+
- `uv` or `pip install pandas numpy scikit-learn matplotlib seaborn requests`
- Observatory API access (token from `observatory_login.py`)

## Notes
- All authentication is handled via the CLI token (`~/.metta/observatory_token`)
- The pipeline is fully API-driven—no need for direct SQL or DB credentials
- For custom metrics or advanced analysis, edit `tools/observatory_policy_analysis.py`

## Example: Extract Only

```bash
./tools/observatory_policy_analysis.py --num-policies 10 --output-dir ./top10
```

---
For further customization or help, see the code comments or ask the team.
