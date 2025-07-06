terraform {
  required_providers {
    tailscale = { source = "tailscale/tailscale" }
  }
}

provider "aws" {
  region = "us-east-1"
}

provider "tailscale" {
  tailnet = var.tailscale_tailnet

  oauth_client_id     = var.tailscale_oauth_client_id
  oauth_client_secret = var.tailscale_oauth_client_secret
}

