# Consolidated Curriculum Analysis

This directory contains the consolidated curriculum analysis system that accomplishes three core goals:

## Core Goals

1. **Use Main Branch Curricula**: Uses the actual learning progress, random, and prioritized regression curricula from the main codebase
2. **Compare Against Oracle**: Compares performance against an enhanced oracle baseline that uses known dependency graphs
3. **Keep Learning Progress Sweep**: Maintains the learning progress grid search code for hyperparameter optimization

## Files

### Main Analysis Script
- `consolidated_curriculum_analysis.py` - Comprehensive curriculum analysis script that:
  - Supports both chain and binary tree dependency graphs
  - Compares learning progress, random, and prioritized regression curricula
  - Uses enhanced oracle baseline with realistic learning curves
  - Generates comprehensive visualizations
  - Provides detailed performance metrics and regret analysis

### Learning Progress Sweep (Kept as Requested)
- `learning_progress_grid_search_demo.py` - Grid search for learning progress hyperparameters
  - Explores ema_timescale and progress_smoothing parameters
  - Provides optimal parameter combinations
  - Generates heatmap visualizations

### Cleanup Script
- `cleanup_curriculum_files.py` - Removes unnecessary files and consolidates the codebase

## Usage

### Basic Analysis
```bash
# Run with default settings (chain graph, 10 tasks, 150 epochs)
python examples/consolidated_curriculum_analysis.py

# Run with custom settings
python examples/consolidated_curriculum_analysis.py --graph-type binary_tree --num-tasks 8 --num-epochs 200
```

### Learning Progress Sweep
```bash
# Run grid search for learning progress hyperparameters
python examples/learning_progress_grid_search_demo.py
```

### Cleanup
```bash
# Remove unnecessary files
python cleanup_curriculum_files.py
```

## Features

### Dependency Graphs
- **Chain Graph**: Sequential dependency (A → B → C → ...)
- **Binary Tree Graph**: Hierarchical dependency structure

### Curricula Compared
1. **Learning Progress Curriculum**: Adaptive sampling based on learning progress
   - Uses optimal hyperparameters from grid search
   - Tracks fast and slow exponential moving averages
   - Applies progress smoothing

2. **Random Curriculum**: Uniform random sampling
   - Baseline comparison
   - No adaptation to performance

3. **Prioritize Regressed Curriculum**: Prioritizes tasks with performance regression
   - Tracks maximum and moving average rewards
   - Focuses on tasks that have regressed

4. **Enhanced Oracle**: Optimal baseline using known dependency graph
   - Topological sorting with learning curve optimization
   - Realistic performance prediction
   - Dependency-aware scheduling

### Analysis Metrics
- **Performance**: Average task performance over time
- **Efficiency**: Cumulative performance (area under curve)
- **Time to Threshold**: Epochs to reach performance threshold
- **Time to Mastery**: Epochs to achieve excellent performance
- **Performance Variance**: Stability of performance
- **Regret Metrics**: Comparison against oracle baseline

### Visualizations
- Dependency graph visualization
- Performance comparison across curricula
- Sampling probability analysis
- Efficiency comparison
- Oracle baseline comparison
- Learning progress grid search heatmaps

## Output Files

### Analysis Results
- `consolidated_curriculum_analysis.png` - Comprehensive visualization
- `learning_progress_grid_search.png` - Grid search results (if sweep is run)

### Generated Data
- Task completion history
- Curriculum metrics over time
- Task weights evolution
- Performance statistics

## Architecture

### Main Components
1. **DependencyGraphGenerator**: Creates chain and binary tree dependency graphs
2. **CurriculumAnalyzer**: Runs curriculum comparison using main branch curricula
3. **LearningProgressSweep**: Performs grid search for hyperparameter optimization
4. **CurriculumVisualizer**: Creates comprehensive visualizations

### Integration with Main Codebase
- Uses actual curriculum implementations from `mettagrid/src/metta/mettagrid/curriculum/`
- Leverages enhanced oracle from `metta/rl/enhanced_oracle.py`
- Integrates with analysis framework from `metta/rl/curriculum_analysis.py`

## Key Improvements

1. **Consolidation**: Single comprehensive analysis script instead of multiple separate demos
2. **Main Branch Integration**: Uses actual curriculum code instead of simulations
3. **Enhanced Oracle**: Realistic baseline with dependency-aware scheduling
4. **Comprehensive Metrics**: Detailed performance and regret analysis
5. **Flexible Graphs**: Support for different dependency structures
6. **Clean Codebase**: Removed unnecessary files while keeping essential functionality

## Dependencies

- Main branch curriculum implementations
- Enhanced oracle with realistic learning curves
- NetworkX for dependency graph analysis
- Matplotlib and Seaborn for visualizations
- NumPy for numerical computations

## Future Enhancements

1. **Additional Graph Types**: Support for more complex dependency structures
2. **More Curricula**: Integration of additional curriculum algorithms
3. **Real Training Integration**: Connect with actual training pipeline
4. **Advanced Metrics**: Additional regret and efficiency metrics
5. **Interactive Visualizations**: Web-based interactive analysis tools 