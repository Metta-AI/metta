#!/usr/bin/env python3
import argparse
import os
import sys

import sky

sys.path.insert(0, ".")

from devops.skypilot.utils import dashboard_url
from metta.util.colorama import blue, green, yellow


def get_existing_clusters():
    request_id = sky.status()
    cluster_records = sky.get(request_id)
    return cluster_records


def get_next_name(cluster_records):
    names = [record["name"] for record in cluster_records]
    username = os.environ["USER"]
    for i in range(1, 100):
        name = f"{username}-sandbox-{i}"
        if name not in names:
            return name
    raise ValueError("No available sandbox name found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--git-ref", type=str, default=None)
    parser.add_argument("--new", action="store_true")
    args = parser.parse_args()

    existing_clusters = get_existing_clusters()
    if existing_clusters and not args.new:
        print(f"You already have {len(existing_clusters)} sandbox(es) running:")
        for cluster in existing_clusters:
            message = ""
            if cluster["status"].name == "INIT":
                message = " (launching)"
            elif cluster["status"].name == "STOPPED":
                message = " (stopped)"
            print(f"  {yellow(cluster['name'])}{message}")
        print(f"\nLaunch an additional sandbox: {green('./devops/skypilot/launch_sandbox.py --new')}")
        first_cluster_name = existing_clusters[0]["name"]
        print(f"Connect to a sandbox: {green(f'sky status {first_cluster_name} && ssh {first_cluster_name}')}")
        return

    cluster_name = get_next_name(existing_clusters)
    print(f"Launching {cluster_name}...")

    task = sky.Task.from_yaml("./devops/skypilot/config/sandbox.yaml")
    request_id = sky.launch(task, cluster_name=cluster_name)

    short_request_id = request_id.split("-")[0]
    print(blue(f"Cluster {cluster_name} is launching"))
    print(
        f"Follow the launch logs with: {green(f'sky api logs {short_request_id}')}, "
        f"or on the dashboard: {blue(f'{dashboard_url()}')}"
    )
    print(f"When the sandbox is ready, connect with: {green(f'sky status {cluster_name} && ssh {cluster_name}')}")


if __name__ == "__main__":
    main()
