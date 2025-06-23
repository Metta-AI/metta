variable "zone_domain" {
  default = "softmax-research.net"
}

variable "subdomain" {
  default = "skypilot-api"
}

variable "lambda_ai_secret_arn" {
  default = "arn:aws:secretsmanager:us-east-1:751442549699:secret:LambdaAI-qo6uuJ"
}

variable "jobs_bucket" {
  default = "skypilot-jobs"
}
