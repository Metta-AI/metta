OpenTofu (Terraform) configs are managed by [Spacelift](https://spacelift.io/).

This directory contains "stacks": invididual Terraform configurations for different parts of the infrastructure.

Note that `spacelift` stack configures Spacelift, but not the individual stacks in it (see `spacelift/README.md` for details).

# Creating new stacks

- read https://docs.spacelift.io/concepts/stack/creating-a-stack
- use OpenTofu instead of Terraform
- if you need to deploy resources to AWS, attach the integration in Stack Settings -> Integrations
- you might want to switch on "Enable local preview" in Stack Setting -> Behavior for easier testing (`spacectl stack local-preview` command in CLI)
