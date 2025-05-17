resource "tailscale_tailnet_key" "mettabox" {
  reusable      = true
  preauthorized = true
  description   = "Mettabox key"
  tags          = ["tag:mettabox"]
  depends_on    = [tailscale_acl.acl] # won't work until the tag is mentioned in ACL
}

output "mettabox_tailscale_key" {
  value = nonsensitive(tailscale_tailnet_key.mettabox.key)
}
