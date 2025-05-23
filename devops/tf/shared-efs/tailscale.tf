resource "tailscale_tailnet_key" "efs_proxy" {
  reusable      = false # we only need this once, when EC2 instance is created
  preauthorized = true
  expiry        = 3600
  description   = "EFS proxy key"
  tags          = ["tag:efs-proxy"]
  depends_on    = [tailscale_acl.acl] # won't work until the tag is mentioned in ACL
}

# Note: there can be only one ACL per tailnet. So this stack is the only one
# that can be used to manage Tailscale ACL.

resource "tailscale_acl" "acl" {
  acl = jsonencode({
    tagOwners : {
      "tag:efs-proxy" : ["autogroup:admin"]
      "tag:mettabox" : ["autogroup:admin"]
    },
    autoApprovers : {
      routes : {
        "0.0.0.0/0" : ["tag:efs-proxy"],
        "::/0" = ["tag:efs-proxy"],
      }
    },
    acls : [
      # DISABLED - we use the default permissive ACL for now
      # {
      #   action = "accept",
      #   src    = ["autogroup:member"],
      #   dst = [
      #     # covers the proxy’s EFS routes
      #     # note: app connector docs recommend "autogroup:internet:*" here, but it doesn't work, I think because EFS drive is on the local network
      #     "*:2049",
      #     # allow ssh (we don't have SSH keys right now, so this is not used)
      #     "tag:efs-proxy:22"
      #   ],
      # },

      { "action" : "accept", "src" : ["*"], "dst" : ["*:*"] },
    ]
    nodeAttrs = [
      {
        target = ["*"],
        app = {
          "tailscale.com/app-connectors" = [
            {
              name       = "AWS-EFS",
              connectors = ["tag:efs-proxy"],
              domains = [
                # add one entry per region where you have mount-targets
                "*.efs.${var.aws_zone}.amazonaws.com",
              ]
            }
          ]
        }
      }
    ]
  })
}
