# Metta Metrics Analysis

A comprehensive toolkit for analyzing and comparing machine learning runs from Weights & Biases (WandB), with a focus on statistical rigor and reproducibility.

## Features

- **WandB Integration**: Seamlessly fetch and cache run data from WandB
- **Statistical Analysis**:
  - Interquartile Mean (IQM) with bootstrap confidence intervals
  - Performance profiles for algorithm comparison
  - Stratified bootstrap for correlated data
  - Optimality gap analysis
- **Data Processing**:
  - Flexible data transformation and aggregation
  - Missing data handling and interpolation
  - Metric normalization and derived metrics
- **Export Options**: CSV, Parquet, and direct integration with Jupyter notebooks

## Installation

```bash
# Using uv (recommended)
cd metrics_analysis
uv pip install -e .

# Or with standard pip
pip install -e .
```

## Quick Start

```python
from metta_metrics_analysis import WandBDataCollector, DataProcessor, StatisticalAnalyzer

# 1. Fetch runs from WandB
collector = WandBDataCollector(entity="metta-research", project="metta")
data = collector.fetch_runs(
    run_filter={"group": "my_sweep"},
    metrics=["eval_navigation/success_rate", "trainer/loss"],
    config_params=["trainer.algorithm"],
)

# 2. Process data
processor = DataProcessor(data)
run_summary = processor.aggregate_by_run()

# 3. Analyze
analyzer = StatisticalAnalyzer(run_summary)
iqm_results = analyzer.compute_iqm_with_ci(
    metric="eval_navigation/success_rate_mean",
    group_by="config.trainer.algorithm"
)
```

## Core Components

### WandBDataCollector

Handles all interactions with the WandB API:

```python
collector = WandBDataCollector(
    entity="metta-research",
    project="metta",
    use_cache=True,  # Cache results locally
    cache_dir="~/.metta/wandb_cache"
)

# Fetch with metric groups
data = collector.fetch_runs(
    metrics=["@eval_navigation", "@training_core"],  # Use predefined groups
    last_n_steps=1000  # Only fetch recent data
)
```

**Predefined Metric Groups:**
- `@training_core`: Core training metrics (loss, entropy, etc.)
- `@eval_navigation`: Navigation task evaluation metrics
- `@eval_memory`: Memory task evaluation metrics
- `@agent_behavior`: Agent behavior statistics
- `@system_metrics`: GPU/CPU utilization and timing

### DataProcessor

Transform and prepare data for analysis:

```python
processor = DataProcessor(data)

# Filter incomplete runs
processor = processor.filter_complete_runs(min_steps=1000)

# Aggregate metrics
summary = processor.aggregate_by_run(
    metrics=["reward", "loss"],
    aggregations={
        "reward": ["mean", "max", "last"],
        "loss": ["mean", "min"]
    }
)

# Add derived metrics
processor = processor.add_derived_metrics({
    "efficiency": "reward / steps",
    "normalized_reward": "reward / optimal_reward"
})
```

### StatisticalAnalyzer

Perform rigorous statistical analysis:

```python
analyzer = StatisticalAnalyzer(data)

# IQM with confidence intervals
iqm_results = analyzer.compute_iqm_with_ci(
    metric="success_rate",
    group_by="algorithm",
    stratify_by="task",  # Account for task correlation
    confidence_level=0.95,
    n_bootstrap=10000
)

# Performance profiles
profiles = analyzer.compute_performance_profiles(
    metric="reward",
    group_by="algorithm",
    task_column="environment"
)

# Statistical comparisons
comparisons = analyzer.compare_algorithms(
    metric="success_rate",
    group_by="algorithm",
    test="wilcoxon",  # or "mannwhitney", "ttest"
    correction="bonferroni"  # Multiple comparison correction
)
```

## Advanced Usage

### Custom Metric Groups

```python
collector.add_metric_group("my_custom_metrics", [
    "custom/metric_1",
    "custom/metric_2",
    "env_agent/special_*"  # Wildcards supported
])

data = collector.fetch_runs(metrics=["@my_custom_metrics"])
```

### Optimality Gap Analysis

```python
# Define optimal scores per task
optimal_scores = {
    "maze_easy": 100.0,
    "maze_hard": 85.0,
    "navigation_simple": 95.0
}

gaps = analyzer.compute_optimality_gaps(
    metric="score",
    optimal_scores=optimal_scores,
    task_column="environment",
    group_by="algorithm",
    normalize=True  # Express as percentage
)
```

### Export to WandB

```python
import wandb

# Create analysis run
wandb.init(project="metta-analysis", name="algorithm_comparison")

# Log results as tables
wandb.log({
    "iqm_comparison": wandb.Table(dataframe=iqm_results),
    "performance_profiles": wandb.Table(dataframe=profiles)
})

# Save processed data as artifact
artifact = wandb.Artifact("processed_runs", type="dataset")
artifact.add_file("processed_data.parquet")
wandb.log_artifact(artifact)
```

## Examples

See the `examples/` directory for complete examples:
- `basic_analysis.py`: Simple workflow demonstration
- `sweep_comparison.py`: Compare hyperparameter sweep results
- `bootstrap_analysis.py`: Advanced bootstrap techniques
- `performance_profiles.py`: Detailed performance profile analysis

## Best Practices

1. **Caching**: Enable caching for large datasets to avoid repeated API calls
2. **Metric Selection**: Use metric groups (`@group_name`) for consistency
3. **Statistical Rigor**: Always use confidence intervals for aggregate metrics
4. **Stratification**: Use `stratify_by` when data has natural groupings (e.g., tasks)
5. **Multiple Comparisons**: Apply correction when comparing multiple algorithms

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Enable caching and reduce `max_runs` for testing
2. **Memory Issues**: Use `last_n_steps` to limit data per run
3. **Missing Metrics**: Check available metrics with `collector.get_available_metrics(run_id)`

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Follow the existing code style (enforced by ruff)
2. Add tests for new functionality
3. Update documentation as needed

## License

This project is part of the Metta research framework.
