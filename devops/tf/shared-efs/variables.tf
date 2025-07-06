variable "tailscale_tailnet" {
  type    = string
  default = "stem.ai"
}

# Secrets
variable "tailscale_oauth_client_id" {
  type = string
}

variable "tailscale_oauth_client_secret" {
  type      = string
  sensitive = true
}
