OpenTofu (Terraform) configs are managed by [Spacelift](https://spacelift.io/).

This directory contains "stacks": Terraform configurations for different parts of the infrastructure.

Note that `spacelift` stack configures Spacelift, but not the individual stacks in it (see `spacelift/README.md` for details).

# Creating new stacks

- read https://docs.spacelift.io/concepts/stack/creating-a-stack
- use OpenTofu instead of Terraform
- if you need to deploy resources to AWS, don't forget to "Attach cloud" during stack creation, or later in Stack Settings -> Integrations
- you might want to switch on "Enable local preview" in Stack Settings -> Behavior for easier testing (`spacectl stack local-preview` command in CLI)

When you first create a stack in a PR, you might need to use the branch's PR in the stack's settings. Don't forget to update the stack's settings when the PR is merged.
