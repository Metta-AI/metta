data "aws_ssm_parameter" "al2023_arm" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64"
}

resource "aws_security_group" "proxy" {
  vpc_id = var.vpc_id

  # allow Session Manager & Tailscale control traffic out
  egress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # allow access to EFS
  egress {
    from_port   = 2049
    to_port     = 2049
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # direct WireGuard paths
  # (optional? o3 marked this as optional)
  ingress {
    from_port   = 41641
    to_port     = 41641
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow ssh
  # There are no keys but you can connect through AWS Console
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Create EC2 instance to run as a proxy
resource "aws_instance" "proxy" {
  ami           = data.aws_ssm_parameter.al2023_arm.value
  instance_type = "t4g.micro"

  tags = {
    Name = "EFSProxy"
  }

  user_data = <<-EOF
    #!/bin/bash

    # enable ip forwarding
    echo 'net.ipv4.ip_forward = 1' | tee -a /etc/sysctl.d/99-tailscale.conf
    echo 'net.ipv6.conf.all.forwarding = 1' | tee -a /etc/sysctl.d/99-tailscale.conf
    sysctl -p /etc/sysctl.d/99-tailscale.conf

    # install tailscale
    yum install -y jq
    curl -fsSL https://tailscale.com/install.sh | sh

    # automatically register because we pre-generated the key
    tailscale up --hostname=efs-proxy --auth-key=${tailscale_tailnet_key.efs_proxy.key} --advertise-connector --advertise-tags=tag:efs-proxy
    systemctl enable --now tailscaled

    # install efs utils for local debugging
    yum install -y amazon-efs-utils
  EOF

  vpc_security_group_ids = [aws_security_group.proxy.id]
}
