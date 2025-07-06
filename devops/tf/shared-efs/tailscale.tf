# Note: there can be only one ACL per tailnet. So this stack is the only one
# that can be used to manage Tailscale ACL.

resource "tailscale_acl" "acl" {
  acl = jsonencode({
    tagOwners : {
      "tag:mettabox" : ["autogroup:admin"]
    },
    acls : [
      # we use the default permissive ACL for now
      { "action" : "accept", "src" : ["*"], "dst" : ["*:*"] },
    ]
  })
}
