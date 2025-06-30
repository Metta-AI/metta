#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "wandb>=0.16.0",
#     "requests>=2.31.0",
#     "pyyaml>=6.0",
# ]
# ///

import os
from collections import defaultdict
from pathlib import Path

import yaml


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

        for group_name, group_metrics in sorted(metric_groups.items()):
            content += f"**{group_name}:**\n"

            # Check if we have a description for the primary metric
            primary_metric = f"{section}/{subsection}/{group_name}".replace("/general/", "/")
            if subsection == "general":
                primary_metric = f"{section}/{group_name}"

            metric_desc = get_metric_description(primary_metric, descriptions)

            # List the metrics
            for metric in sorted(group_metrics)[:10]:  # Limit to 10 examples
                content += f"- `{metric}`\n"
            if len(group_metrics) > 10:
                content += f"- ... and {len(group_metrics) - 10} more\n"

            # Add description if available
            if metric_desc:
                content += f"\n{metric_desc}\n"

            content += "\n"

        content += "\n"

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


def load_metric_descriptions(script_dir):
    """Load metric descriptions from YAML file."""
    yaml_path = os.path.join(script_dir, "metric_descriptions.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def get_metric_description(metric, descriptions):
    """Get description for a specific metric."""
    if metric in descriptions and "description" in descriptions[metric]:
        desc = descriptions[metric]["description"]

        # Add unit if specified
        if "unit" in descriptions[metric]:
            desc += f" (Unit: {descriptions[metric]['unit']})"

        # Add interpretation if specified
        if "interpretation" in descriptions[metric]:
            desc += f"\n\n**Interpretation:** {descriptions[metric]['interpretation']}"

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
    metrics_file = os.path.join(script_dir, "wandb_metrics.txt")
    output_dir = os.path.join(script_dir, "docs", "metrics")

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

    # Generate main README
    main_readme = generate_main_readme(sections, output_dir)
    print(f"Created: {main_readme}")

    # Generate section READMEs
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
