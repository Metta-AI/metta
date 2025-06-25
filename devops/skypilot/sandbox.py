#!/usr/bin/env -S uv run
import argparse
import os
import subprocess

import sky
import sky.cli

from metta.common.util.colorama import blue, green, yellow


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
    parser.add_argument("--gpus", type=int, default=1, help="Number of L4 GPUs to use.")
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
        print(f"  {green('./devops/skypilot/sandbox.py --new')}")

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

    autostop_hours = 48

    # Launch the cluster
    task = sky.Task.from_yaml("./devops/skypilot/config/sandbox.yaml")
    task.set_resources_override({"accelerators": f"L4:{args.gpus}"})

    request_id = sky.launch(task, cluster_name=cluster_name, idle_minutes_to_autostop=autostop_hours * 60)
    sky.stream_and_get(request_id)

    # Cluster is up but the setup job is still starting
    print("Waiting for setup job to start...")
    setup_result = sky.tail_logs(cluster_name, job_id=1, follow=True)
    if setup_result != 0:
        print(f"Setup job failed with exit code {setup_result}")
        return

    # Force ssh setup
    subprocess.run(["sky", "status", cluster_name], check=True)

    print("\nSandbox is ready. Connect to the sandbox:")
    print(f"  {green(f'ssh {cluster_name}')}")
    print(
        f"The cluster will be automatically stopped after {autostop_hours} hours. "
        "If you want to disable autostops, run:"
    )
    print(f"  {green(f'sky autostop --cancel {cluster_name}')}")


if __name__ == "__main__":
    main()
