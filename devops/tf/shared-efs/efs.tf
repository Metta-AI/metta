# create EFS
resource "aws_efs_file_system" "efs" {
  tags = {
    Name = "Shared"
  }
}

# allow access to EFS
resource "aws_security_group" "allow_efs_access" {
  vpc_id = var.vpc_id

  ingress {
    from_port   = 2049
    to_port     = 2049
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# find the subnet that was used for ec2 instance
# (note: aws_default_subnet resources don't create new subnets)
resource "aws_default_subnet" "proxy_subnet" {
  availability_zone = aws_instance.proxy.availability_zone
}

# make EFS mountable (probably already allowed in production)
resource "aws_efs_mount_target" "proxy" {
  file_system_id = aws_efs_file_system.efs.id
  subnet_id      = aws_default_subnet.proxy_subnet.id

  security_groups = [aws_security_group.allow_efs_access.id]
}

resource "aws_ssm_parameter" "efs_url" {
  name           = "/shared-efs/url"
  type           = "String"
  insecure_value = aws_efs_mount_target.proxy.dns_name
}

