from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timedelta, timezone

from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v1.api.hosts_api import HostsApi
from datadog_api_client.v1.api.metrics_api import MetricsApi
from datadog_api_client.v1.api.usage_metering_api import UsageMeteringApi

from metta.common.datadog.config import datadog_config
from softmax.aws.secrets_manager import get_secretsmanager_secret

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def get_configuration() -> Configuration:
    configuration = Configuration()
    configuration.server_variables["site"] = os.environ.get("DD_SITE", datadog_config.DD_SITE)

    api_key = get_secretsmanager_secret("datadog/api-key", require_exists=False)
    app_key = get_secretsmanager_secret("datadog/app-key", require_exists=False)

    if not api_key:
        raise RuntimeError("Missing Datadog API key from AWS Secrets Manager (datadog/api-key)")
    if not app_key:
        raise RuntimeError("Missing Datadog App key from AWS Secrets Manager (datadog/app-key)")

    configuration.api_key["apiKeyAuth"] = api_key
    configuration.api_key["appKeyAuth"] = app_key
    return configuration


def get_infrastructure_hosts(api_client: ApiClient) -> None:
    logger.info("\n=== Infrastructure Hosts ===")
    hosts_api = HostsApi(api_client)

    response = hosts_api.list_hosts()
    hosts = response.get("host_list", [])

    logger.info(f"Total infrastructure hosts: {len(hosts)}")

    k8s_hosts = []
    other_hosts = []

    for host in hosts:
        name = host.get("name", "unknown")
        tags = host.get("tags_by_source", {})
        all_tags = []
        for source_tags in tags.values():
            all_tags.extend(source_tags)

        is_k8s = any("kube" in tag or "kubernetes" in tag for tag in all_tags)
        namespace_tags = [t for t in all_tags if t.startswith("kube_namespace:")]

        host_info = {
            "name": name,
            "namespaces": [t.split(":")[1] for t in namespace_tags],
            "tags": all_tags[:10],
        }

        if is_k8s:
            k8s_hosts.append(host_info)
        else:
            other_hosts.append(host_info)

    logger.info(f"\nK8s hosts: {len(k8s_hosts)}")
    for h in k8s_hosts[:10]:
        ns_str = ", ".join(h["namespaces"]) if h["namespaces"] else "no namespace tag"
        logger.info(f"  - {h['name']} ({ns_str})")
    if len(k8s_hosts) > 10:
        logger.info(f"  ... and {len(k8s_hosts) - 10} more")

    logger.info(f"\nNon-K8s hosts: {len(other_hosts)}")
    for h in other_hosts[:5]:
        logger.info(f"  - {h['name']}")
    if len(other_hosts) > 5:
        logger.info(f"  ... and {len(other_hosts) - 5} more")

    jobs_ns_hosts = [h for h in k8s_hosts if "jobs" in h["namespaces"]]
    orchestrator_ns_hosts = [h for h in k8s_hosts if "orchestrator" in h["namespaces"]]

    logger.info(f"\nHosts with 'jobs' namespace: {len(jobs_ns_hosts)}")
    for h in jobs_ns_hosts:
        logger.info(f"  - {h['name']}")

    logger.info(f"\nHosts with 'orchestrator' namespace: {len(orchestrator_ns_hosts)}")
    for h in orchestrator_ns_hosts[:5]:
        logger.info(f"  - {h['name']}")
    if len(orchestrator_ns_hosts) > 5:
        logger.info(f"  ... and {len(orchestrator_ns_hosts) - 5} more")


def get_usage_summary(api_client: ApiClient, days: int = 7) -> None:
    logger.info(f"\n=== Usage Summary (last {days} days) ===")
    usage_api = UsageMeteringApi(api_client)

    end_date = datetime.now(timezone.utc)
    end_month = end_date.replace(day=1)
    start_month = (end_date - timedelta(days=days)).replace(day=1)

    try:
        summary = usage_api.get_usage_summary(
            start_month=start_month,
            end_month=end_month,
        )

        if summary:
            logger.info("\nMonthly Usage Summary:")
            infra_hosts = getattr(summary, "infra_host_top99p", None) or 0
            apm_hosts = getattr(summary, "apm_host_top99p", None) or 0
            apm_fargate = getattr(summary, "apm_fargate_count_avg", None) or 0

            logger.info(f"  Infrastructure hosts (99th percentile): {infra_hosts}")
            logger.info(f"  APM hosts (99th percentile): {apm_hosts}")
            logger.info(f"  APM Fargate tasks (avg): {apm_fargate}")

            if infra_hosts > 50:
                logger.info("\n  WARNING: High infrastructure host count!")
            if apm_hosts > 50:
                logger.info("\n  WARNING: High APM host count!")
        else:
            logger.info("No usage data available")
    except Exception as e:
        logger.warning(f"Could not get usage summary (may need admin permissions): {e}")


def check_container_metrics(api_client: ApiClient, namespace: str) -> None:
    logger.info(f"\n=== Container Metrics for namespace '{namespace}' ===")
    metrics_api = MetricsApi(api_client)

    end_time = int(datetime.now(timezone.utc).timestamp())
    start_time = end_time - 3600

    query = f"avg:kubernetes.cpu.usage.total{{kube_namespace:{namespace}}}"

    try:
        result = metrics_api.query_metrics(
            _from=start_time,
            to=end_time,
            query=query,
        )

        series = result.get("series", [])
        if series:
            logger.info(f"Found {len(series)} metric series")
            logger.info("WARNING: Container metrics ARE being collected for this namespace!")
            logger.info("containerExcludeMetrics may not be working correctly.")
        else:
            logger.info("No container metrics found")
            logger.info("containerExcludeMetrics appears to be working correctly.")
    except Exception as e:
        logger.error(f"Failed to query metrics: {e}")


def get_apm_traces_summary(api_client: ApiClient) -> None:
    logger.info("\n=== APM Service Summary ===")

    logger.info("Note: To see APM services, check the Datadog UI:")
    logger.info("  https://app.datadoghq.com/apm/services")
    logger.info("\nLook for services like:")
    logger.info("  - eval-worker (from orchestrator namespace)")
    logger.info("  - Any services from jobs namespace (should be none)")


def main():
    parser = argparse.ArgumentParser(description="Verify Datadog billing configuration")
    parser.add_argument("--days", type=int, default=7, help="Number of days to check usage")
    args = parser.parse_args()

    logger.info("Datadog Billing Verification")
    logger.info("=" * 50)

    configuration = get_configuration()

    with ApiClient(configuration) as api_client:
        get_infrastructure_hosts(api_client)
        get_usage_summary(api_client, days=args.days)
        check_container_metrics(api_client, "jobs")
        check_container_metrics(api_client, "orchestrator")
        get_apm_traces_summary(api_client)

    logger.info("\n" + "=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    logger.info("""
Based on code analysis:
- Episode Runner (jobs ns): NO DD_AGENT_HOST -> Should NOT cause APM charges
- Eval Worker (orchestrator ns): HAS DD_AGENT_HOST -> DOES cause APM charges

Key files:
- dispatcher.py:105-113 - Episode runner env vars (no Datadog)
- k8s.py:93-110 - Eval worker env vars (has DD_AGENT_HOST)
- datadog-values.yaml:42 - containerExcludeMetrics setting
""")


if __name__ == "__main__":
    main()
