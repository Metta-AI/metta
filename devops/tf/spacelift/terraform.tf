terraform {
  required_providers {
    spacelift = { source = "spacelift-io/spacelift" }
  }
}


# TODO - instead of configuring this here, we could take it from AWS Secrets Manager
resource "spacelift_environment_variable" "TF_VAR_tailscale_api_key" {
  name        = "TF_VAR_tailscale_api_key"
  description = "Generated manually in https://login.tailscale.com/admin/settings/keys on 2025-05-02, expires in 90 days. Regenerate and update if necessary. (TODO: switch to Tailscale OAuth)"
  write_only  = true
  stack_id    = spacelift_stack.shared_efs.id
  value       = var.tailscale_api_key
}
