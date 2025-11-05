variable "region" {
  type        = string
  default     = "us-east-1"
  description = "AWS region for sandbox resources"
}

variable "vpc_cidr" {
  type        = string
  default     = "10.100.0.0/16"
  description = "CIDR block for sandbox VPC (isolated from main VPC at 10.0.0.0/16)"
}

variable "environment" {
  type        = string
  default     = "production"
  description = "Environment name (production, staging, etc.)"
}

variable "allowed_ssh_cidrs" {
  type        = list(string)
  default     = ["0.0.0.0/0"]
  description = "CIDR blocks allowed to SSH into sandbox instances. Default allows all (researchers connect from various IPs)."
}

variable "default_instance_type" {
  type        = string
  default     = "g5.12xlarge"
  description = "Default EC2 instance type for sandboxes (4x A10G GPUs)"
}

variable "sandbox_ami_id" {
  type        = string
  default     = ""
  description = "AMI ID for sandbox instances (leave empty to use latest Ubuntu 22.04)"
}
