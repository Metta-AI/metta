resource "tailscale_tailnet_key" "mettabox" {
  reusable      = true
  preauthorized = true
  description   = "Mettabox key"
  tags          = ["tag:mettabox"]
  depends_on    = [tailscale_acl.acl] # won't work until the tag is mentioned in ACL
}

output "mettabox_tailscale_key" {
  # make this available through spacelift UI - we apply it to mettaboxes manually
  value = nonsensitive(tailscale_tailnet_key.mettabox.key)
}

data "tailscale_devices" "mettaboxes" {
  name_prefix = "metta"
}

resource "tailscale_device_key" "disable_mettabox_expiry" {
  for_each = {
    for device in data.tailscale_devices.mettaboxes.devices :
    device.id => device if contains(device.tags, "tag:mettabox")
  }
  device_id           = each.value.id
  key_expiry_disabled = true
}
