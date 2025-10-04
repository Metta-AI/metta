variable "region" {
  type    = string
  default = "us-east-1"
}

variable "eks_cluster_name" {
  type    = string
  default = "main"
}

variable "db_postgres_version" {
  description = "The version of PostgreSQL to use"
  type        = string
  default     = "17.5"
}

variable "db_instance_class" {
  description = "The instance class for the RDS database"
  type        = string
  default     = "db.t3.micro"
}

variable "db_allocated_storage" {
  description = "The allocated storage for the RDS database (in GB)"
  type        = number
  default     = 20
}

variable "oauth_secret_arn" {
  description = "ARN of the AWS Secrets Manager secret containing OAuth credentials"
  default = "arn:aws:secretsmanager:us-east-1:751442549699:secret:softmax-infra-oauth-rLFTw6"
}

variable "frontend_secret_name" {
  description = "Name of the Kubernetes secret for frontend environment variables"
  type        = string
  default     = "softmax-login-frontend-secrets"
}

variable "domain" {
  type    = string
  default = "login.softmax-research.net"
}
