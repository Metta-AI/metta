# This file should contain the reference to all Spacelift stacks.
# When you add a new `devops/tf/` directory, you should usually add a new resource here.

resource "spacelift_stack" "shared_efs" {
  name     = "shared-efs"
  space_id = "root"

  namespace    = "Metta-AI"
  repository   = "metta"
  branch       = "main"
  project_root = "devops/tf/shared-efs"

  terraform_version            = "1.9.1"
  terraform_workflow_tool      = "OPEN_TOFU"
  terraform_smart_sanitization = true

  enable_well_known_secret_masking = true
  github_action_deploy             = false
}

# If the stack needs to deploy to AWS, we need to attach the AWS integration to it.
resource "spacelift_aws_integration_attachment" "shared_efs" {
  integration_id = spacelift_aws_integration.softmax.id
  stack_id       = spacelift_stack.shared_efs.id
  read           = true
  write          = true
}
