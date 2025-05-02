terraform {
  required_providers {
    tailscale = { source = "tailscale/tailscale" }
  }
}

provider "aws" {
  region = var.aws_zone
}

provider "tailscale" {
  api_key = var.tailscale_api_key
  tailnet = var.tailscale_tailnet
}

