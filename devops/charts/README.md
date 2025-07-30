This directory contains helm charts that are used in the Metta project.

## Configuration

Make sure your `~/.kube/config` is configured.

Install necessary tools with `./devops/macos/setup_machine.py --devops`.

## Updating charts

### Helmfile-powered charts

Most of the infrastructure is described by `helmfile.yaml`.

This file describes several Helm charts:

- core services such as `ingress-nginx` and `cert-manager`
- `skypilot` chart (which is our fork of the upstream SkyPilot helm chart)

How to update infra charts:

1. Edit some charts and/or `helmfile.yaml`.
2. Then run `helmfile apply`.

Refer to [helmfile documentation](https://helmfile.readthedocs.io/en/latest/) for more details.

### Observatory

Observatory charts (`./observatory` and `./observatory-backend`) are deployed by CI/CD pipelines powered by GitHub
Actions.

Image tag values are updated on each build, so we can't describe these charts with all up-to-date values statically in
`helmfile.yaml`.
