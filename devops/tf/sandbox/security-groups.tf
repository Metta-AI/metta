# Security group for researcher sandbox instances
resource "aws_security_group" "sandbox" {
  name        = "sandbox-instances"
  description = "Security group for researcher sandbox EC2 instances"
  vpc_id      = module.sandbox_vpc.vpc_id

  tags = merge(local.tags, {
    Name = "sandbox-instances"
  })
}

# Allow SSH inbound from specified CIDR blocks
resource "aws_vpc_security_group_ingress_rule" "ssh" {
  for_each = toset(var.allowed_ssh_cidrs)

  security_group_id = aws_security_group.sandbox.id
  description       = "Allow SSH access from researchers"

  from_port   = 22
  to_port     = 22
  ip_protocol = "tcp"
  cidr_ipv4   = each.value

  tags = merge(local.tags, {
    Name = "sandbox-ssh-inbound-${each.key}"
  })
}

# Allow all outbound traffic (for pip installs, git clone, etc.)
resource "aws_vpc_security_group_egress_rule" "all" {
  security_group_id = aws_security_group.sandbox.id
  description       = "Allow all outbound traffic"

  ip_protocol = "-1"
  cidr_ipv4   = "0.0.0.0/0"

  tags = merge(local.tags, {
    Name = "sandbox-all-outbound"
  })
}
