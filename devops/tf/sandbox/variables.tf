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
