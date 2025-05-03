# This file should contain the reference to all Spacelift stacks.
# When you add a new `devops/tf/` directory, you should usually add a new resource here.

locals {
  stacks = [
    { name = "shared-efs" },
    { name = "spacelift" }, # register this very stack too
  ]
}

resource "spacelift_stack" "spacelift" {
  name         = "spacelift"
  space_id     = "root"
  repository   = "metta"
  branch       = "main"
  project_root = "devops/tf/spacelift"

  terraform_version            = "1.9.1"
  terraform_workflow_tool      = "OPEN_TOFU"
  terraform_smart_sanitization = true

  enable_well_known_secret_masking = true
  enable_local_preview             = true
  github_action_deploy             = false

  # spacelift stack uses this for managing other stacks
  administrative = true
}

resource "spacelift_stack" "shared_efs" {
  name         = "shared-efs"
  space_id     = "root"
  repository   = "metta"
  branch       = "main"
  project_root = "devops/tf/shared-efs"

  terraform_version            = "1.9.1"
  terraform_workflow_tool      = "OPEN_TOFU"
  terraform_smart_sanitization = true

  enable_well_known_secret_masking = true
  github_action_deploy             = false
}

# Allow each stack to deploy to AWS
resource "spacelift_aws_integration_attachment" "spacelift" {
  integration_id = spacelift_aws_integration.softmax.id
  stack_id       = spacelift_stack.spacelift.id
  read           = true
  write          = true
}

resource "spacelift_aws_integration_attachment" "shared_efs" {
  integration_id = spacelift_aws_integration.softmax.id
  stack_id       = spacelift_stack.shared_efs.id
  read           = true
  write          = true
}

# Note: stack-specific secrets are configured manually in the Spacelift UI.
# (In stack settings, under "Environment variables", add the variables one by one when needed.)

import {
  to = spacelift_stack.spacelift
  id = "spacelift"
}

import {
  to = spacelift_stack.shared_efs
  id = "efs"
}
