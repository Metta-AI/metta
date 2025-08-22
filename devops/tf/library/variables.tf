variable "region" {
  type    = string
  default = "us-east-1"
}

variable "db_instance_class" {
  type    = string
  default = "db.t3.micro"
}

variable "db_allocated_storage" {
  type    = number
  default = 20 # GiB
}

variable "db_postgres_version" {
  type    = string
  default = "17.5"
}

variable "oauth_secret_arn" {
  default = "arn:aws:secretsmanager:us-east-1:751442549699:secret:softmax-infra-oauth-rLFTw6"
}

variable "library_secrets_arn" {
  default = "arn:aws:secretsmanager:us-east-1:751442549699:secret:softmax-library-secrets-IzNjL8"
}

variable "amplify_github_access_token" {
  type      = string
  sensitive = true

  # https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/amplify_app#access_token-1
  description = "One-time only access token for Amplify to create a webhook"
}

variable "s3_buckets" {
  type    = list(string)
  default = ["softmax-library", "softmax-library-dev"]
}

variable "main_s3_bucket" {
  type    = string
  default = "softmax-library"
}

variable "domain" {
  type    = string
  default = "library.softmax-research.net"
}
