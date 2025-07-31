variable "region" {
  default = "us-east-1"
}

variable "cluster_name" {
  default = "main"
}

variable "cluster_version" {
  default = "1.32"
}

variable "oauth_secret_arn" {
  default = "arn:aws:secretsmanager:us-east-1:751442549699:secret:softmax-infra-oauth-rLFTw6"
}

variable "oauth_secret_namespaces" {
  default     = ["skypilot", "monitoring", "observatory"]
  description = "The secret with google oauth credentials will be created in these namespaces"
}
