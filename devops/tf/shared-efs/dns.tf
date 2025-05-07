# Internal DNS namespace
# Used to bind skypilot-api.softmax
resource "aws_service_discovery_private_dns_namespace" "dns" {
  name        = "softmax"
  description = "Softmax private namespace"
  vpc         = var.vpc_id
}

