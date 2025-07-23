# Analysis Module

This module provides comprehensive analysis tools for policy performance evaluation using the Observatory API.

## Modules

### `observatory_client.py`
- `ObservatoryClient`: Client for interacting with the Observatory API
- Methods for querying policies, evaluations, and environments
- Handles authentication and API requests

### `correlation_analyzer.py`
- `CorrelationAnalyzer`: Analyzes correlations between task performances
- Calculates correlation matrices and identifies highly correlated task pairs
- Provides statistical summaries of correlations

### `meta_analyzer.py`
- `MetaAnalyzer`: Performs PCA and statistical analysis
- Generates comprehensive summary statistics
- Handles data preprocessing and dimensionality reduction

### `visualization_utils.py`
- `VisualizationUtils`: Creates various visualizations
- Heatmaps, scatter plots, distribution plots
- Publication-ready figure generation

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Observatory authentication:
```bash
python devops/observatory_login.py
```

3. Use in notebooks:
```python
from analysis.observatory_client import ObservatoryClient
from analysis.correlation_analyzer import CorrelationAnalyzer
from analysis.meta_analyzer import MetaAnalyzer
from analysis.visualization_utils import VisualizationUtils

# Initialize clients
client = ObservatoryClient()
corr_analyzer = CorrelationAnalyzer()
meta_analyzer = MetaAnalyzer()
viz_utils = VisualizationUtils()
```

## Example Workflow

1. Query top policies from Observatory
2. Get evaluation data across environments
3. Create performance matrix
4. Calculate correlations between tasks
5. Perform PCA analysis
6. Generate visualizations and reports

## Dependencies

- pandas: Data manipulation
- numpy: Numerical computing
- matplotlib: Basic plotting
- seaborn: Statistical visualizations
- scikit-learn: PCA and machine learning
- requests: API communication
