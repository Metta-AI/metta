#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "wandb>=0.16.0",
#     "requests>=2.31.0",
# ]
# ///

import os
from collections import defaultdict
from pathlib import Path


def parse_metrics_file(filepath):
    """Parse the wandb_metrics.txt file and return organized metrics."""
    sections = defaultdict(lambda: defaultdict(list))

    with open(filepath, "r") as f:
        lines = f.readlines()

    # Parse metrics, skipping header and footer
    for line in lines:
        line = line.strip()

        # Skip empty lines, headers, and footer
        if (
            not line
            or line.startswith("=")
            or line.startswith("WandB Metrics for")
            or line.startswith("Total metrics:")
        ):
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


def generate_main_readme(sections, output_dir):
    """Generate the main README.md file."""
    content = """# WandB Metrics Documentation

This directory contains comprehensive documentation for all metrics logged to Weights & Biases (WandB) during
Metta training runs.

## Overview

Our WandB logging captures detailed metrics across multiple categories to monitor training progress, agent behavior,
environment dynamics, and system performance.

## Metric Categories

| Section | Description | Metric Count |
|---------|-------------|--------------|
"""

    # Sort sections by metric count
    section_counts = [(s, sum(len(metrics) for metrics in subs.values())) for s, subs in sections.items()]
    section_counts.sort(key=lambda x: x[1], reverse=True)

    for section, count in section_counts:
        desc = get_section_description(section)
        content += f"| [`{section}/`](./{section}/) | {desc} | {count} |\n"

    content += f"\n**Total Metrics:** {sum(c for _, c in section_counts)}\n"

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
./collect_metrics.py <run_id>  # Fetches metrics to wandb_metrics.txt
./generate_docs.py             # Regenerates documentation
```
"""

    # Create output directory
    readme_path = Path(output_dir) / "README.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)

    with open(readme_path, "w") as f:
        f.write(content)

    return readme_path


def get_section_description(section):
    """Get a description for each section."""
    descriptions = {
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
    return descriptions.get(section, "Metrics for " + section.replace("_", " "))


def generate_section_readme(section, subsections, output_dir):
    """Generate README for a specific section."""
    section_dir = Path(output_dir) / section
    section_dir.mkdir(parents=True, exist_ok=True)

    # Analyze all metrics in this section
    all_metrics = []
    for metrics in subsections.values():
        all_metrics.extend(metrics)

    patterns = analyze_metric_patterns(all_metrics)

    content = f"""# {section.replace("_", " ").title()} Metrics

## Overview

{get_section_description(section)}

**Total metrics in this section:** {len(all_metrics)}

## Subsections

"""

    # Document subsections
    for subsection, metrics in sorted(subsections.items()):
        if subsection == "general":
            content += "### General Metrics\n\n"
        else:
            content += f"### {subsection.replace('_', ' ').title()}\n\n"

        content += f"**Count:** {len(metrics)} metrics\n\n"

        # Group related metrics
        metric_groups = group_related_metrics(metrics)

        if len(metric_groups) > 5:
            # Just show summary for large groups
            content += "**Metric Groups:**\n"
            for group_name, group_metrics in sorted(metric_groups.items()):
                content += f"- `{group_name}` ({len(group_metrics)} metrics)\n"
        else:
            # Show detailed breakdown for small groups
            for group_name, group_metrics in sorted(metric_groups.items()):
                content += f"**{group_name}:**\n"
                for metric in sorted(group_metrics)[:10]:  # Limit to 10 examples
                    content += f"- `{metric}`\n"
                if len(group_metrics) > 10:
                    content += f"- ... and {len(group_metrics) - 10} more\n"
                content += "\n"

        content += "\n"

    # Add interpretation guide based on section
    content += get_section_interpretation_guide(section, patterns)

    readme_path = section_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(content)

    return readme_path


def group_related_metrics(metrics):
    """Group metrics by their base name."""
    groups = defaultdict(list)

    for metric in metrics:
        # Extract base metric name (before statistics suffixes)
        base = metric
        for suffix in [
            ".avg",
            ".std_dev",
            ".min",
            ".max",
            ".first_step",
            ".last_step",
            ".rate",
            ".updates",
            ".activity_rate",
        ]:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break

        # Further group by removing the section prefix
        parts = base.split("/")
        if len(parts) > 2:
            group_key = "/".join(parts[2:])
        else:
            group_key = parts[-1]

        groups[group_key].append(metric)

    return dict(groups)


def get_section_interpretation_guide(section, patterns):
    """Get interpretation guide for specific sections."""
    guides = {
        "env_agent": """
## Interpretation Guide

### Action Metrics
- `action.<action_type>.success/failed` - Track success rates for different actions
- `action.<action_type>.agent` - Breakdown by individual agents
- Look for high failure rates to identify problematic behaviors

### Item Interactions
- `<item>.gained/lost` - Track item acquisition and loss
- `<item>.stolen/stolen_from` - Monitor PvP dynamics
- `<item>.get/put` - Environmental interactions

### Combat Metrics
- `attack.win/loss` - Combat outcomes
- `attack.blocked` - Defensive success
- `attack.own_team` - Friendly fire incidents

### Key Patterns to Watch
1. **Action Success Rates**: Low success rates may indicate poor policy or difficult environments
2. **Item Flow**: Track how items move between agents
3. **Combat Balance**: Monitor win/loss ratios across agents
""",
        "env_timing_per_epoch": """
## Interpretation Guide

### Timing Categories
- `msec/` - Raw millisecond timings
- `frac/` - Fraction of total time
- `active_frac/` - Fraction of active (non-idle) time

### Key Operations
- `step` - Environment step execution
- `reset` - Episode reset operations
- `_initialize_c_env` - C++ environment initialization
- `process_episode_stats` - Statistics processing

### Performance Analysis
1. **Bottlenecks**: Look for operations with high `msec` values
2. **Efficiency**: Check `thread_idle` for CPU utilization
3. **Variability**: High `std_dev` values indicate inconsistent performance
""",
        "losses": """
## Interpretation Guide

### Loss Components
- `policy_loss` - Actor loss for action selection
- `value_loss` - Critic loss for value estimation
- `entropy` - Policy entropy (exploration)
- `approx_kl` - KL divergence (policy stability)

### Training Health Indicators
1. **Convergence**: Decreasing losses over time
2. **Stability**: Low variance in loss values
3. **Exploration**: Maintain reasonable entropy
4. **Policy Updates**: Monitor `approx_kl` and `clipfrac`
""",
    }

    return guides.get(
        section,
        """
## Interpretation Guide

Monitor these metrics for:
- Trends over time
- Anomalies or spikes
- Correlations with training performance
- System resource usage patterns
""",
    )


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_file = os.path.join(script_dir, "wandb_metrics.txt")
    output_dir = os.path.join(script_dir, "docs", "metrics")

    if not os.path.exists(metrics_file):
        print(f"Error: {metrics_file} not found!")
        print("Please run collect_metrics.py first to generate the metrics file.")
        return

    print(f"Parsing metrics from {metrics_file}...")
    sections = parse_metrics_file(metrics_file)

    print(f"Found {len(sections)} sections")
    print(f"Generating documentation in {output_dir}/")

    # Generate main README
    main_readme = generate_main_readme(sections, output_dir)
    print(f"Created: {main_readme}")

    # Generate section READMEs
    for section, subsections in sections.items():
        section_readme = generate_section_readme(section, subsections, output_dir)
        print(f"Created: {section_readme}")

    print("\nDocumentation generation complete!")
    print(f"Main README: {output_dir}/README.md")
    print(f"Section docs: {output_dir}/<section>/README.md")
    print("\nTo view the documentation, navigate to:")
    print(f"  {output_dir}/")


if __name__ == "__main__":
    main()
