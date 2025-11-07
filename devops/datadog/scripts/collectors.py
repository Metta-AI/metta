"""Datadog collector registry and configuration.

Single source of truth for all collector classes, metadata, and execution order.
"""

from devops.datadog.collectors.asana.collector import AsanaCollector
from devops.datadog.collectors.ec2.collector import EC2Collector
from devops.datadog.collectors.github.collector import GitHubCollector
from devops.datadog.collectors.health_fom.collector import HealthFomCollector
from devops.datadog.collectors.kubernetes.collector import KubernetesCollector
from devops.datadog.collectors.skypilot.collector import SkypilotCollector
from devops.datadog.collectors.wandb.collector import WandBCollector

COLLECTOR_REGISTRY = {
    "github": {
        "class": GitHubCollector,
        "source": "github-collector",
        "description": "GitHub metrics",
    },
    "skypilot": {
        "class": SkypilotCollector,
        "source": "skypilot-collector",
        "description": "Skypilot job metrics",
    },
    "asana": {
        "class": AsanaCollector,
        "source": "asana-collector",
        "description": "Asana project metrics",
    },
    "ec2": {
        "class": EC2Collector,
        "source": "ec2-collector",
        "description": "AWS EC2 metrics",
    },
    "wandb": {
        "class": WandBCollector,
        "source": "wandb-collector",
        "description": "WandB training metrics",
    },
    "kubernetes": {
        "class": KubernetesCollector,
        "source": "kubernetes-collector",
        "description": "Kubernetes efficiency metrics",
    },
    "health_fom": {
        "class": HealthFomCollector,
        "source": "health-fom-collector",
        "description": "Health FoM metrics",
    },
}

COLLECTOR_RUN_ORDER = [
    "github",
    "kubernetes",
    "ec2",
    "skypilot",
    "wandb",
    "asana",
    "health_fom",
]
