# This file should contain the reference to all Spacelift stacks.
# When you add a new `devops/tf/` directory, you should usually add a new resource here.

locals {
  stacks = [
    { name = "shared-efs" },
    { name = "spacelift" }, # register this very stack too
  ]
}

resource "spacelift_stack" "stacks" {
  for_each     = { for stack in local.stacks : stack.name => stack }
  name         = each.value.name
  space_id     = "root"
  repository   = "metta"
  branch       = "main"
  project_root = "devops/tf/${each.value.name}"

  terraform_version            = "1.9.1"
  terraform_workflow_tool      = "OPEN_TOFU"
  terraform_smart_sanitization = true

  enable_well_known_secret_masking = true
  github_action_deploy             = false
}

# Allow each stack to deploy to AWS
resource "spacelift_aws_integration_attachment" "aws_integrations" {
  for_each       = { for stack in local.stacks : stack.name => stack }
  integration_id = spacelift_aws_integration.softmax.id
  stack_id       = spacelift_stack.stacks["${each.value.name}"].id
  read           = true
  write          = true
}

# Note: stack-specific secrets are configured manually in the Spacelift UI.
# (In stack settings, under "Environment variables", add the variables one by one when needed.)

import {
  to = spacelift_stack.stacks["shared-efs"]
  id = "shared-efs"
}
