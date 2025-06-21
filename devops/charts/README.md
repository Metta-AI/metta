This directory contains helm charts that are used in the Metta project.

## Configuration

Make sure your `~/.kube/config` is configured.

Install necessary tools with `./devops/macos/setup_machine.py --devops`.

## Updating charts

Edit some charts and/or `helmfile.yaml`.

Then run `helmfile apply`.

Refer to [helmfile documentation](https://helmfile.readthedocs.io/en/latest/) for more details.
