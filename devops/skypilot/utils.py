import re
import sys

import sky

from metta.util.colorama import blue, bold, green


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

    print(green(f"Submitted sky.jobs.launch request: {request_id}"))

    short_request_id = request_id.split("-")[0]

    print(f"- Check logs with: {bold(f'sky api logs {short_request_id}')}")
    print(f"- Or, visit: {bold(f'{dashboard_url()}/api/stream?request_id={short_request_id}')}")
    print("  - To sign in, use credentials from your ~/.skypilot/config.yaml file.")
    print(f"- To cancel the request, run: {bold(f'sky api cancel {short_request_id}')}")
