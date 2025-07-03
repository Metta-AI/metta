#!/usr/bin/env python3
# This script should be run from within the metta workspace environment

import os
from collections import defaultdict
from pathlib import Path

import yaml

from metta.common.util.fs import get_repo_root


def parse_metrics_file(filepath):
    """Parse the wandb_metrics.csv file and return organized metrics."""
    sections = defaultdict(lambda: defaultdict(list))

    with open(filepath, "r") as f:
        lines = f.readlines()

    # Parse metrics - each line is a metric
    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Only process lines that look like metrics (contain '/')
        if "/" in line:
            parts = line.split("/")
            if len(parts) >= 2:
                section = parts[0]
                subsection = parts[1] if len(parts) > 2 else "general"
                sections[section][subsection].append(line)

    return dict(sections)


def analyze_metric_patterns(metrics):
    """Analyze common patterns in metrics."""
    patterns = {
        "statistics": [".avg", ".std_dev", ".min", ".max", ".first_step", ".last_step", ".rate", ".updates"],
        "activity": [".activity_rate", ".activity_rate.std_dev"],
        "agent_specific": [".agent", ".agent.agent"],
        "outcomes": [".success", ".failed", ".won", ".lost"],
        "timing": ["msec/", "frac/", "active_frac/"],
    }

    categorized = defaultdict(list)
    for metric in metrics:
        for category, pattern_list in patterns.items():
            if any(p in metric for p in pattern_list):
                categorized[category].append(metric)
                break
        else:
            categorized["other"].append(metric)

    return dict(categorized)


def generate_aggregation_section():
    """Generate the metric aggregation documentation section."""
    aggregation_section = """
## Metric Aggregation Strategy

Metta uses a multi-stage aggregation pipeline to produce the final metrics logged to WandB:

### Aggregation Pipeline

```
Per-Agent Values → Per-Episode Means → Cross-Episode Means → WandB Logs
```

### Detailed Aggregation Table

| Metric Category | Stage 1: Environment<br>(per episode) | Stage 2: Rollout<br>(collection) | Stage 3: Trainer<br>(final processing) | Final Output |
|----------------|------------------------|------------------|----------------------|--------------|
| **Agent Rewards** | Sum across agents ÷ num_agents | Collect all episode means into list | Mean of all episodes | `env_map_reward/*` = mean<br>`env_map_reward/*.std_dev` = std |
| **Agent Stats**<br>(e.g., actions, items) | Sum across agents ÷ num_agents | Collect all episode values into list | Mean of all episodes | `env_agent/*` = mean<br>`env_agent/*.std_dev` = std |
| **Game Stats**<br>(environment-level) | Single value (no aggregation) | Collect all episode values | Mean of all episodes | `env_game/*` = mean<br>`env_game/*.std_dev` = std |
| **Per-Epoch Timing** | Single value per operation | Keep latest value only | Pass through latest | `env_timing_per_epoch/*` = latest<br>`timing_per_epoch/*` = latest |
| **Cumulative Timing** | Single value per operation | Running average over all steps | Current running average | `env_timing_cumulative/*` = running avg<br>`timing_cumulative/*` = running avg |
| **Attributes**<br>(seed, map size, etc.) | Single value (no aggregation) | Last value overwrites | Pass through | `env_attributes/*` = value |
| **Task Rewards** | Mean across agents | Collect all episode means | Mean of all episodes | `env_task_reward/*` = mean |
| **Curriculum Stats** | Single value | Last value overwrites | Pass through | `env_curriculum/*` = value |

### Timing Metrics Explained

Metta tracks two types of timing metrics:

1. **Per-Epoch Timing** (`*_per_epoch`):
   - Shows the time taken for the most recent epoch/step only
   - Not averaged - each logged value represents that specific step's timing
   - Useful for: Identifying performance changes or spikes in specific steps

2. **Cumulative Timing** (`*_cumulative`):
   - Shows the running average of all steps up to the current point
   - At step N, this is the average of steps 1 through N
   - Useful for: Understanding overall performance trends and smoothing out variance

### Example: Timing Metrics Over 3 Steps

If rollout timing has values [100ms, 150ms, 120ms]:

- **Per-Epoch**:
  - Step 1: `env_timing_per_epoch/rollout` = 100ms
  - Step 2: `env_timing_per_epoch/rollout` = 150ms
  - Step 3: `env_timing_per_epoch/rollout` = 120ms

- **Cumulative**:
  - Step 1: `env_timing_cumulative/rollout` = 100ms (avg of: 100)
  - Step 2: `env_timing_cumulative/rollout` = 125ms (avg of: 100, 150)
  - Step 3: `env_timing_cumulative/rollout` = 123ms (avg of: 100, 150, 120)

### Key Points

1. **Double Averaging**: Most metrics undergo two averaging operations:
   - First: Average across all agents in an episode
   - Second: Average across all episodes in the rollout

2. **Standard Deviation**: The trainer adds `.std_dev` variants showing variance across episodes

3. **Episode Granularity**: Aggregation preserves episode boundaries - partial episodes are not mixed with complete ones

4. **Multi-GPU Training**: Each GPU computes its own statistics independently; WandB handles any cross-GPU aggregation

### Example: Tracing a Reward Metric

Consider `env_map_reward/collectibles` with 4 agents and 3 completed episodes:

1. **Episode 1**: Agents score [2, 3, 1, 4] → Mean: 2.5
2. **Episode 2**: Agents score [3, 3, 2, 2] → Mean: 2.5
3. **Episode 3**: Agents score [1, 2, 3, 2] → Mean: 2.0

**Rollout Collection**: `[2.5, 2.5, 2.0]`

**Final Processing**:
- `env_map_reward/collectibles` = 2.33 (mean)
- `env_map_reward/collectibles.std_dev` = 0.29 (standard deviation)

### Special Cases

- **Diversity Bonus**: Applied to individual agent rewards before any aggregation
- **Kickstarter Losses**: Not aggregated by episode, averaged across all training steps
- **Gradient Stats**: Computed across all parameters, not per-episode
"""  # noqa: E501
    return aggregation_section


def generate_main_readme(sections, output_dir, descriptions):
    """Generate the main README.md file."""
    content = """# WandB Metrics Documentation

This directory contains comprehensive documentation for all metrics logged to Weights & Biases (WandB) during
Metta training runs.

## Overview

Our WandB logging captures detailed metrics across multiple categories to monitor training progress, agent
behavior, environment dynamics, and system performance.

## Metric Categories

| Section | Description | Metric Count |
|---------|-------------|--------------|
"""

    # Sort sections by metric count
    section_counts = [(s, sum(len(metrics) for metrics in subs.values())) for s, subs in sections.items()]
    section_counts.sort(key=lambda x: x[1], reverse=True)

    for section, count in section_counts:
        desc = get_section_description(section, descriptions)
        # Truncate long descriptions for the table
        if desc is not None and "\n" in desc:
            desc = desc.split("\n")[0] + "..."
        content += f"| [`{section}/`](./{section}/) | {desc} | {count} |\n"

    content += f"\n**Total Metrics:** {sum(c for _, c in section_counts)}\n"

    # Add the aggregation section
    content += generate_aggregation_section()

    content += """
## Metric Naming Convention

Metrics follow a hierarchical naming structure:
```
section/subsection/metric_name[.statistic][.qualifier]
```

### Common Statistics Suffixes
- `.avg` - Average value
- `.std_dev` - Standard deviation
- `.min` - Minimum value
- `.max` - Maximum value
- `.first_step` - First step where metric was recorded
- `.last_step` - Last step where metric was recorded
- `.rate` - Rate of occurrence
- `.updates` - Number of updates
- `.activity_rate` - Fraction of time the metric was active

### Common Qualifiers
- `.agent` - Per-agent breakdown
- `.success` / `.failed` - Outcome-specific metrics
- `.gained` / `.lost` - Change tracking

## Usage

Each subdirectory contains:
- `README.md` - Detailed documentation for that metric category
- Explanations of what each metric measures
- Relationships between related metrics
- Tips for interpretation and debugging

## Quick Start

To explore specific metric categories:
1. Navigate to the relevant subdirectory
2. Read the README for detailed explanations
3. Use the metric names when querying WandB or analyzing logs

## Related Tools

- [`collect_metrics.py`](../../collect_metrics.py) - Script to fetch metrics from WandB runs
- [`generate_docs.py`](../../generate_docs.py) - Script to generate this documentation

## Updating Documentation

To update this documentation with metrics from a new run:
```bash
cd common/src/metta/common/wandb
./collect_metrics.py <run_id>  # Fetches metrics to wandb_metrics.csv
./generate_docs.py             # Regenerates documentation
```
"""

    # Create output directory
    readme_path = Path(output_dir) / "README.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)

    with open(readme_path, "w") as f:
        f.write(content)

    return readme_path


def get_section_description(section, descriptions=None):
    """Get a description for each section."""
    # Check if we have a custom description in the YAML
    if descriptions and "sections" in descriptions and section in descriptions["sections"]:
        return descriptions["sections"][section]["description"].strip()

    # Fall back to default descriptions
    default_descriptions = {
        "env_agent": "Agent actions, rewards, and item interactions",
        "env_attributes": "Environment configuration and episode attributes",
        "env_game": "Game object counts and token tracking",
        "env_map_reward": "Map-specific reward statistics",
        "env_task_reward": "Task completion rewards",
        "env_task_timing": "Task initialization timing",
        "env_timing_cumulative": "Cumulative environment timing statistics",
        "env_timing_per_epoch": "Per-epoch environment timing breakdown",
        "experience": "Training experience buffer statistics",
        "losses": "Training loss components",
        "metric": "Core training metrics (steps, epochs, time)",
        "monitor": "System resource monitoring",
        "overview": "High-level training progress",
        "parameters": "Training hyperparameters",
        "timing_cumulative": "Cumulative training timing",
        "timing_per_epoch": "Per-epoch training timing",
        "trainer_memory": "Memory usage by trainer components",
    }
    return default_descriptions.get(section, "Metrics for " + section.replace("_", " "))


def group_related_metrics(metrics):
    """Group metrics by their base name, consolidating all related statistics."""
    groups = defaultdict(list)

    for metric in metrics:
        # Remove the section prefix to get the metric path
        parts = metric.split("/")
        if len(parts) > 2:
            metric_path = "/".join(parts[2:])
        else:
            metric_path = parts[-1]

        # Find the base metric name by removing all known suffixes
        base = metric_path

        # List of base statistic suffixes
        base_suffixes = [
            ".activity_rate",
            ".avg",
            ".std_dev",
            ".min",
            ".max",
            ".first_step",
            ".last_step",
            ".rate",
            ".updates",
        ]

        # Create list with .std_dev variants first (more specific patterns first)
        stat_suffixes = [s + ".std_dev" for s in base_suffixes] + base_suffixes

        # Remove the suffix to find the base
        for suffix in stat_suffixes:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break

        groups[base].append(metric)

    return dict(groups)


def generate_section_readme(section, subsections, output_dir, descriptions):
    """Generate README for a specific section."""
    section_dir = Path(output_dir) / section
    section_dir.mkdir(parents=True, exist_ok=True)

    # Analyze all metrics in this section
    all_metrics = []
    for metrics in subsections.values():
        all_metrics.extend(metrics)

    content = f"""# {section.replace("_", " ").title()} Metrics

## Overview

{get_section_description(section, descriptions)}

**Total metrics in this section:** {len(all_metrics)}

"""

    # Check if we need to add suffix explanations
    # Look for common suffixes in the metrics
    suffixes_found = set()
    suffix_patterns = [
        ".avg",
        ".std_dev",
        ".min",
        ".max",
        ".first_step",
        ".last_step",
        ".rate",
        ".updates",
        ".activity_rate",
    ]

    for metric in all_metrics:
        for suffix in suffix_patterns:
            if suffix in metric:
                suffixes_found.add(suffix)

    # Add suffix explanations if any are found
    if suffixes_found:
        content += "## Metric Suffixes\n\n"
        content += "This section contains metrics with the following statistical suffixes:\n\n"

        suffix_explanations = {
            ".avg": "**`.avg`** - Average value of the metric across updates within an episode\n"
            "  - Formula: `sum(values) / update_count`",
            ".std_dev": "**`.std_dev`** - Standard deviation across episodes (variance measure)\n"
            "  - Formula: `sqrt(sum((x - mean)²) / n)`",
            ".min": "**`.min`** - Minimum value observed during the episode",
            ".max": "**`.max`** - Maximum value observed during the episode",
            ".first_step": "**`.first_step`** - First step where this metric was recorded",
            ".last_step": "**`.last_step`** - Last step where this metric was recorded",
            ".rate": "**`.rate`** - Frequency of updates (updates per step over entire episode)\n"
            "  - Formula: `update_count / current_step`",
            ".updates": "**`.updates`** - Total number of times this metric was updated in an episode",
            ".activity_rate": "**`.activity_rate`** - Frequency during active period only "
            "(updates per step between first and last occurrence)\n"
            "  - Formula: `(update_count - 1) / (last_step - first_step)`\n"
            "  - Note: Subtracts 1 because the first update just marks the start of activity",
        }

        for suffix in sorted(suffixes_found):
            if suffix in suffix_explanations:
                content += f"- {suffix_explanations[suffix]}\n"

        # Add note about .std_dev variants
        if any(".std_dev" in m for m in all_metrics if m.count(".std_dev") > 1):
            content += "\n**Note:** Metrics ending in `.std_dev` (e.g., `.avg.std_dev`) represent the standard "
            content += "deviation of that statistic across episodes.\n"

        content += "\n"

    content += "## Subsections\n\n"

    # Document subsections
    for subsection, metrics in sorted(subsections.items()):
        if subsection == "general":
            content += "### General Metrics\n\n"
        else:
            content += f"### {subsection.replace('_', ' ').title()}\n\n"

        content += f"**Count:** {len(metrics)} metrics\n\n"

        # Group related metrics
        metric_groups = group_related_metrics(metrics)

        for group_name, group_metrics in sorted(metric_groups.items()):
            # Count base metrics and std_dev variants
            base_count = sum(1 for m in group_metrics if ".std_dev" not in m)
            std_dev_count = sum(1 for m in group_metrics if ".std_dev" in m)

            # Proper pluralization
            value_word = "value" if base_count == 1 else "values"
            std_dev_word = "std_dev" if std_dev_count == 1 else "std_devs"

            content += f"**{group_name}:** ({base_count} {value_word}"
            if std_dev_count > 0:
                content += f" / {std_dev_count} {std_dev_word}"
            content += ")\n"

            # Sort metrics by replacing .std_dev with _ to group them with their base
            # This ensures .std_dev variants appear right after their base metrics
            sorted_metrics = sorted(group_metrics, key=lambda m: m.replace(".std_dev", "_"))

            # List all metrics in this group
            for metric in sorted_metrics:
                content += f"- `{metric}`\n"

                # Check if we have a description for this specific metric
                metric_desc = get_metric_description(metric, descriptions)
                if metric_desc:
                    # Format description as sub-bullets
                    desc_lines = metric_desc.split("\n")
                    for i, line in enumerate(desc_lines):
                        if line.strip():
                            # First line of description
                            if i == 0:
                                content += f"  - {line}\n"
                            # Interpretation or other formatted lines
                            elif line.strip().startswith("**"):
                                content += f"  - {line}\n"
                            # Continuation of previous line
                            else:
                                content += f"    {line}\n"
                    content += "\n"

            content += "\n"

        content += "\n"

    readme_path = section_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(content)

    return readme_path


def load_metric_descriptions(script_dir):
    """Load metric descriptions from YAML file."""
    yaml_path = os.path.join(script_dir, "metric_descriptions.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def get_metric_description(metric, descriptions):
    """Get description for a specific metric."""
    metrics_section = descriptions.get("metrics", {})

    if metric in metrics_section and "description" in metrics_section[metric]:
        desc = metrics_section[metric]["description"].strip()

        # Add unit if specified
        if "unit" in metrics_section[metric]:
            desc += f" (Unit: {metrics_section[metric]['unit']})"

        # Add interpretation if specified
        if "interpretation" in metrics_section[metric]:
            desc += f"\n\n**Interpretation:** {metrics_section[metric]['interpretation']}"

        return desc

    # Check patterns for auto-generated descriptions
    if "patterns" in descriptions:
        for pattern, pattern_info in descriptions["patterns"].items():
            if pattern.startswith("*") and metric.endswith(pattern[1:]):
                base_metric = metric[: -len(pattern[1:])]
                base_desc = get_metric_description(base_metric, descriptions)
                if base_desc:
                    return base_desc.split("\n")[0] + pattern_info.get("suffix", "")

    return None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_file = os.path.join(script_dir, "wandb_metrics.csv")

    root_dir = get_repo_root()
    output_dir = os.path.join(root_dir, "docs", "wandb", "metrics")

    if not os.path.exists(metrics_file):
        print(f"Error: {metrics_file} not found!")
        print("Please run collect_metrics.py first to generate the metrics file.")
        return

    print(f"Parsing metrics from {metrics_file}...")
    sections = parse_metrics_file(metrics_file)

    print("Loading metric descriptions...")
    descriptions = load_metric_descriptions(script_dir)

    print(f"Found {len(sections)} sections")
    print(f"Generating documentation in {output_dir}/")

    main_readme = generate_main_readme(sections, output_dir, descriptions)
    print(f"Created: {main_readme}")

    for section, subsections in sections.items():
        section_readme = generate_section_readme(section, subsections, output_dir, descriptions)
        print(f"Created: {section_readme}")

    print("\nDocumentation generation complete!")
    print(f"Main README: {output_dir}/README.md")
    print(f"Section docs: {output_dir}/<section>/README.md")
    print("\nTo view the documentation, navigate to:")
    print(f"  {output_dir}/")

    # Check if metric_descriptions.yaml exists
    yaml_path = os.path.join(script_dir, "metric_descriptions.yaml")
    if not os.path.exists(yaml_path):
        print("\nNote: No metric_descriptions.yaml found.")
        print(f"Create {yaml_path} to add custom descriptions for metrics.")


if __name__ == "__main__":
    main()
