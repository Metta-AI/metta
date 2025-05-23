import re
import sys

import sky

from metta.util.colorama import blue


def print_tip(text: str):
    print(blue(text), file=sys.stderr)


def dashboard_url():
    url = sky.server.common.get_server_url()
    # strip username and password from server_url
    url = re.sub("https://.*@", "https://", url)
    return url


def launch_task(task: sky.Task, dry_run=False):
    if dry_run:
        print_tip("DRY RUN.")
        print_tip("Tip: Pipe this command to `| yq -P .` to get the pretty yaml config.\n")
        print(task.to_yaml_config())
        return

    request_id = sky.jobs.launch(task)

    print(f"Request ID: {request_id}")
    (job_id, _) = sky.get(request_id)

    print("\nJob submitted successfully!")

    # Note: direct urls don't work in skypilot dashboard yet, this always opens clusters list.
    # Hopefully this will be fixed soon.
    job_url = f"{dashboard_url()}/dashboard/jobs/{job_id}"
    print(f"Open {blue(job_url)} to track your job.")
    print("To sign in, use credentials from your ~/.skypilot/config.yaml file.")
