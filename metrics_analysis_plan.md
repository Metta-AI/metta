# Metrics Analysis System for WandB Run Comparison

## Overview
This system will provide a comprehensive framework for analyzing and comparing machine learning runs from Weights & Biases (wandb), with a focus on statistical analysis of performance metrics across tasks and runs.

## Architecture Components

### 1. Data Collection Module (`wandb_data_collector.py`)
**Purpose**: Interface with wandb API to fetch run data based on user specifications

**Key Features**:
- Flexible run selection using patterns (e.g., `jacke.sky_comprehensive_*`)
- Configurable metric selection (losses, entropy, rewards, etc.)
- Config parameter extraction for labeling
- Batch processing for large numbers of runs
- Caching mechanism to avoid redundant API calls
- Integration with existing metta wandb patterns (e.g., `metta-research` entity)
- Support for metta's metric naming conventions (e.g., `env_agent/`, `trainer/`, `eval_*/`)

**User Interface**:
```python
collector = WandBDataCollector(
    entity="metta-research",  # Default to metta org
    project="metta",
    run_pattern="sky_comprehensive_*",
    metrics=["trainer/loss", "env_agent/reward", "eval_navigation/success_rate"],
    config_params=["trainer.algorithm", "sim.task", "seed"],
    use_cache=True,
    cache_dir="~/.metta/wandb_cache"
)
```

**Integration with Existing Infrastructure**:
- Leverage `wandb.Api()` for data fetching
- Use metta's metric hierarchy (see `docs/wandb/metrics/`)
- Support for metta's run naming convention (`{run_id}`)
- Handle metta's config structure (nested DictConfig from Hydra)

### 2. Data Processing Module (`data_processor.py`)
**Purpose**: Transform raw wandb data into analysis-ready formats

**Key Features**:
- Convert to pandas DataFrames with proper indexing
- Handle missing data and interpolation
- Time-series alignment for comparing runs
- Export to CSV for external analysis
- Support for hierarchical data (runs → tasks → metrics)

**Data Structure**:
```
DataFrame columns:
- run_id
- run_name
- task
- step/epoch
- metric_1, metric_2, ...
- config_param_1, config_param_2, ...
- labels (user-defined)
```

### 3. Labeling System (`run_labeler.py`)
**Purpose**: Allow flexible labeling of runs for grouping and analysis

**Labeling Methods**:
1. **Manual labeling**: Interactive CLI or simple GUI
2. **Config-based labeling**: Automatic labels from wandb configs
3. **Programmatic labeling**: User-defined functions
4. **Hybrid approach**: Combine multiple methods

**Example**:
```python
labeler = RunLabeler()
labeler.add_config_rule("algorithm", label_column="method")
labeler.add_pattern_rule("*_baseline", label="baseline")
labeler.add_manual_labels({"run_123": "experimental"})
```

### 4. Statistical Analysis Module (`statistical_analysis.py`)
**Purpose**: Core statistical computations for performance analysis

**Key Analyses**:

#### 4.1 Interquartile Mean (IQM)
- Remove top and bottom 25% of scores
- Compute mean of middle 50%
- More robust than standard mean for outliers

#### 4.2 Performance Profiles
- Score distribution visualization
- Empirical CDFs for each method
- Probability of achieving various performance thresholds

#### 4.3 Stratified Bootstrap Confidence Intervals
- Account for correlation within tasks
- Stratify by task when sampling
- Compute percentile intervals (e.g., 95% CI)

#### 4.4 Optimality Gaps
- User-defined optimal scores per task
- Compute relative performance gaps
- Normalize across different metric scales

### 5. Jupyter Notebook Generator (`notebook_generator.py`)
**Purpose**: Create analysis notebooks with reusable components

**Notebook Sections**:
1. **Data Loading and Overview**
   - Summary statistics
   - Run metadata table
   - Missing data visualization

2. **Aggregate Performance Analysis**
   - IQM comparisons with confidence intervals
   - Bar plots with error bars
   - Statistical significance tests

3. **Performance Profiles**
   - Score distribution plots
   - Empirical CDF comparisons
   - Violin plots by method/task

4. **Task-Level Analysis**
   - Heatmaps of performance by task
   - Optimality gap analysis
   - Worst-case performance identification

5. **Extensible Analysis Templates**
   - Pre-built functions for common analyses
   - Clear examples for customization
   - Markdown documentation cells

### 6. Configuration System (`config.yaml`)
**Purpose**: Centralized configuration for analysis parameters

```yaml
wandb:
  entity: "your_entity"
  project: "your_project"

metrics:
  primary: ["reward", "success_rate"]
  secondary: ["loss", "entropy"]

analysis:
  iqm_trim_percentage: 0.25
  bootstrap_samples: 10000
  confidence_level: 0.95

optimal_scores:
  task_1: 100.0
  task_2: 1.0

visualization:
  style: "seaborn"
  color_palette: "Set2"
```

## Implementation Plan

### Phase 1: Core Infrastructure (Days 1-2)
1. Set up project structure
2. Implement wandb data collector with basic functionality
3. Create data processor for DataFrame conversion
4. Write unit tests for data pipeline

### Phase 2: Labeling and Processing (Days 3-4)
1. Implement flexible labeling system
2. Add data validation and cleaning
3. Create CSV export functionality
4. Test with sample wandb data

### Phase 3: Statistical Analysis (Days 5-7)
1. Implement IQM calculation
2. Create performance profile generation
3. Implement stratified bootstrap
4. Add optimality gap calculations
5. Comprehensive testing of statistical methods

### Phase 4: Visualization and Notebooks (Days 8-10)
1. Create notebook generator
2. Implement visualization functions
3. Build example analyses
4. Create documentation and tutorials

### Phase 5: Integration and Polish (Days 11-12)
1. End-to-end testing
2. Performance optimization
3. User documentation
4. Example use cases

## Key Design Decisions

### 1. Modularity
Each component is independent and can be used separately or together.

### 2. Extensibility
Users can easily add custom metrics, labeling schemes, and analyses.

### 3. Reproducibility
All analyses include random seeds and versioning for reproducibility.

### 4. Performance
- Efficient data structures for large-scale analysis
- Parallel processing for bootstrap calculations
- Caching for expensive operations

### 5. User Experience
- Clear error messages
- Progress bars for long operations
- Sensible defaults with full customization

## Dependencies
```
wandb>=0.12.0
pandas>=1.3.0
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
nbformat>=5.1.0
pyyaml>=5.4.0
tqdm>=4.62.0
click>=8.0.0  # for CLI
```

## Directory Structure
```
metrics_analysis/
├── src/
│   ├── __init__.py
│   ├── wandb_data_collector.py
│   ├── data_processor.py
│   ├── run_labeler.py
│   ├── statistical_analysis.py
│   ├── visualization.py
│   └── notebook_generator.py
├── notebooks/
│   ├── example_analysis.ipynb
│   └── templates/
│       ├── aggregate_analysis.ipynb
│       └── performance_profiles.ipynb
├── configs/
│   └── default_config.yaml
├── tests/
│   ├── test_data_collector.py
│   ├── test_processor.py
│   └── test_statistics.py
├── examples/
│   └── example_usage.py
├── requirements.txt
└── README.md
```

## Wandb Integration Details

### Leveraging Wandb Features

1. **Wandb Tables**:
   - Create wandb Tables for structured comparison
   - Export analysis results back to wandb for visualization
   - Use wandb's built-in comparison features

2. **Wandb Artifacts**:
   - Store processed datasets as artifacts
   - Version control for analysis configurations
   - Share reproducible analysis pipelines

3. **Wandb Reports**:
   - Generate programmatic reports with analysis results
   - Embed interactive plots and tables
   - Create shareable analysis dashboards

### Metta-Specific Metric Patterns

Based on the existing metric structure in `docs/wandb/metrics/`:

1. **Metric Categories**:
   - `env_agent/*`: Agent behavior metrics (1164+ metrics)
   - `trainer/*`: Training metrics (loss, gradients, etc.)
   - `eval_*/*`: Evaluation metrics by task type
   - `env_task_*/*`: Task-specific environment metrics
   - `monitor/*`: System resource monitoring

2. **Common Analysis Patterns**:
   ```python
   # Aggregate over agent steps
   collector.add_metric_group("training_progress", [
       "trainer/loss",
       "trainer/entropy",
       "trainer/value_loss",
       "trainer/policy_loss"
   ])

   # Task-specific evaluation
   collector.add_metric_group("navigation_eval", [
       "eval_navigation/success_rate",
       "eval_navigation/path_efficiency",
       "eval_navigation/steps_to_goal"
   ])
   ```

3. **Run Grouping**:
   - Group by sweep ID (for hyperparameter sweeps)
   - Group by policy generation
   - Group by task configuration

## Additional Considerations

### Data Handling Edge Cases

1. **Incomplete Runs**:
   - Handle runs that crashed or were stopped early
   - Option to exclude runs below minimum step threshold
   - Interpolation strategies for missing intermediate data

2. **Multi-Task Runs**:
   - Support for runs that switch between tasks
   - Task-specific metric extraction
   - Handling of task-conditional metrics

3. **Large-Scale Data**:
   - Streaming API for runs with millions of steps
   - Downsampling strategies (e.g., every nth step, logarithmic)
   - Memory-efficient storage formats (HDF5, Parquet)

### Statistical Robustness

1. **Small Sample Handling**:
   - Warnings when sample size is too small for reliable statistics
   - Alternative methods for n < 30 (e.g., exact tests)
   - Minimum sample requirements for bootstrap

2. **Distribution Assumptions**:
   - Non-parametric defaults (no normality assumptions)
   - Options for parametric tests when appropriate
   - Outlier detection and handling strategies

3. **Multiple Comparisons**:
   - Bonferroni correction options
   - False discovery rate control
   - Family-wise error rate considerations

### Implementation Details

1. **WandB API Optimization**:
   ```python
   # Efficient batch fetching
   def fetch_runs_batch(self, run_ids, chunk_size=50):
       for i in range(0, len(run_ids), chunk_size):
           chunk = run_ids[i:i+chunk_size]
           # Parallel API calls
   ```

2. **Data Caching Strategy**:
   ```python
   # Cache structure
   cache/
   ├── runs/
   │   ├── {run_id}_metadata.json
   │   └── {run_id}_history.parquet
   └── cache_index.json
   ```

3. **Metric Normalization**:
   - Automatic detection of metric direction (minimize/maximize)
   - Standardization options (z-score, min-max)
   - Custom normalization functions

### User Workflow Enhancements

1. **Interactive Mode**:
   - Jupyter widget for run selection
   - Real-time preview of selected data
   - Interactive labeling interface

2. **Batch Analysis**:
   - Command-line interface for automated reports
   - Scheduled analysis jobs
   - Integration with CI/CD pipelines

3. **Export Options**:
   - LaTeX tables for papers
   - Interactive HTML reports
   - Raw data in multiple formats

### Error Handling and Validation

1. **Input Validation**:
   - Verify wandb credentials and permissions
   - Check metric existence before fetching
   - Validate config parameters

2. **Graceful Degradation**:
   - Continue analysis with available data
   - Clear warnings for missing metrics
   - Fallback options for failed API calls

3. **Logging and Debugging**:
   - Comprehensive logging system
   - Debug mode with detailed traces
   - Performance profiling options

## Next Steps

Before implementation, we need to clarify:

1. **WandB Structure**:
   - How are tasks defined in your wandb runs?
   - Are they separate runs or metrics within runs?
   - What's the typical data volume?

2. **Metric Types**:
   - Are all metrics numeric?
   - Do some metrics need special handling (e.g., higher is better vs. lower is better)?
   - Are there multi-dimensional metrics?

3. **Analysis Priorities**:
   - Which statistical analyses are most critical?
   - Are there specific visualizations you prefer?
   - Do you need real-time analysis or batch processing?

4. **Integration Requirements**:
   - Should this integrate with existing metta codebase?
   - Are there specific coding standards to follow?
   - Do you need CLI, API, or both?

5. **Performance Constraints**:
   - Expected number of runs to analyze?
   - Typical length of each run?
   - Memory/compute constraints?

## Example Usage

### Basic Usage
```python
# Initialize collector with metta defaults
collector = WandBDataCollector(
    entity="metta-research",
    project="metta",
    config="configs/analysis_config.yaml"
)

# Fetch runs using wandb query syntax
data = collector.fetch_runs(
    run_filter={"group": "sky_comprehensive"},
    metrics=["trainer/loss", "eval_navigation/success_rate"],
    last_n_steps=1000
)

# Process and label
processor = DataProcessor(data)
df = processor.to_dataframe()

labeler = RunLabeler()
df = labeler.label_by_config(df, "trainer.algorithm")

# Analyze
analyzer = StatisticalAnalyzer(df)
iqm_results = analyzer.compute_iqm_with_ci(
    metric="eval_navigation/success_rate",
    group_by="algorithm",
    stratify_by="sim.task"
)

# Generate notebook
generator = NotebookGenerator()
generator.create_analysis_notebook(
    data=df,
    analyses=["iqm", "profiles", "optimality"],
    output_path="analysis_results.ipynb"
)
```

### Advanced Wandb Integration
```python
# Create wandb report with results
import wandb

# Initialize a new wandb run for analysis
wandb.init(project="metta-analysis", name="sweep_comparison")

# Log analysis results as wandb Table
results_table = wandb.Table(dataframe=iqm_results)
wandb.log({"iqm_comparison": results_table})

# Create performance profile plot
fig = analyzer.create_performance_profile(
    metric="eval_navigation/success_rate",
    group_by="algorithm"
)
wandb.log({"performance_profile": wandb.Image(fig)})

# Save processed data as artifact
artifact = wandb.Artifact("processed_runs", type="dataset")
artifact.add_file("processed_data.parquet")
wandb.log_artifact(artifact)

# Generate wandb Report
report = wandb.Report(
    project="metta-analysis",
    title="Sky Comprehensive Sweep Analysis",
    description="Statistical comparison of algorithms"
)
report.blocks = [
    wandb.ReportBlock(panel=results_table, caption="IQM Results"),
    wandb.ReportBlock(panel=fig, caption="Performance Profiles")
]
report.save()
```

This plan provides a comprehensive framework for building the metrics analysis system. The modular design allows for incremental development and easy extension based on specific needs.
