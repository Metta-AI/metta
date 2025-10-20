variable "region" {
  type    = string
  default = "us-east-1"
}

variable "eks_cluster_name" {
  type    = string
  default = "main"
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

variable "worker_secret_name" {
  type = string
  # must match the library helm chart
  default = "softmax-library-worker-secrets"
}

variable "frontend_secret_name" {
  type = string
  # must match the library helm chart
  default = "softmax-library-frontend-secrets"
}

variable "ses_from_name" {
  type        = string
  description = "Display name for SES email notifications"
  default     = "Softmax Library"
}

variable "ses_from_email" {
  type        = string
  description = "Email address for SES email notifications"
  default     = "library@softmax.com"
}
