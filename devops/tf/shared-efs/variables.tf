variable "aws_zone" {
  type    = string
  default = "us-east-1"
}

variable "tailscale_tailnet" {
  type    = string
  default = "stem.ai"
}

variable "vpc_id" {
  type        = string
  description = "The ID of the VPC to deploy the proxy to"
  default     = "vpc-021c19429aaf206eb" # default VPC in us-east-1
}

# Secrets
variable "tailscale_api_key" {
  type = string
}
