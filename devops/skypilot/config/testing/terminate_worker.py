# terminate_worker.py
import random

import boto3


def terminate_random_worker(cluster_name, region="us-west-2"):
    """Terminate a random worker node in the cluster."""
    ec2 = boto3.client("ec2", region_name=region)

    # Find instances with cluster tag
    response = ec2.describe_instances(
        Filters=[
            {"Name": "tag:skypilot-cluster-name", "Values": [cluster_name]},
            {"Name": "instance-state-name", "Values": ["running"]},
        ]
    )

    instances = []
    for r in response["Reservations"]:
        for i in r["Instances"]:
            # Skip master node (rank 0)
            if i.get("Tags", {}).get("skypilot-node-rank", "0") != "0":
                instances.append(i["InstanceId"])

    if instances:
        target = random.choice(instances)
        print(f"Terminating instance: {target}")
        ec2.terminate_instances(InstanceIds=[target])
        return target
    else:
        print("No worker instances found")
        return None
