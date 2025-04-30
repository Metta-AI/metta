variable "tailscale_api_key" {
  type = string
}

variable "tailscale_tailnet" {
  type = string
}

variable "vpc_id" {
  type        = string
  description = "The ID of the VPC to deploy the proxy to"
}
