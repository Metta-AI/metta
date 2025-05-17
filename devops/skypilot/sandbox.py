#!/usr/bin/env python3
import argparse
import os
import sys

import sky

sys.path.insert(0, ".")

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

        print("\nLaunch an additional sandbox:")
        print(f"  {green('./devops/skypilot/launch_sandbox.py --new')}")

        first_stopped_cluster_name = next(
            (cluster["name"] for cluster in existing_clusters if cluster["status"].name == "STOPPED"), None
        )
        if first_stopped_cluster_name:
            print("Restart a stopped sandbox:")
            print(f"  {green(f'sky start {first_stopped_cluster_name}')}")

        first_cluster_name = existing_clusters[0]["name"]
        print("Connect to a sandbox:")
        print(f"  {green(f'ssh {first_cluster_name}')}")

        print("Delete a sandbox:")
        print(f"  {green(f'sky down {first_cluster_name}')}")

        return

    cluster_name = get_next_name(existing_clusters)
    print(f"Launching {blue(cluster_name)}... This will take a few minutes.")

    task = sky.Task.from_yaml("./devops/skypilot/config/sandbox.yaml")
    request_id = sky.launch(task, cluster_name=cluster_name, idle_minutes_to_autostop=48 * 60)
    sky.stream_and_get(request_id)

    print("\nConnect to the sandbox:")
    print(f"  {green(f'sky status {cluster_name} && ssh {cluster_name}')}")
    print("The cluster will be automatically stopped after 48 hours. If you want to disable autostops, run:")
    print(f"  {green(f'sky autostop --cancel {cluster_name}')}")


if __name__ == "__main__":
    main()
