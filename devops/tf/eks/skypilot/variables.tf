variable "lambda_ai_secret_arn" {
  default = "arn:aws:secretsmanager:us-east-1:751442549699:secret:LambdaAI-qo6uuJ"
}

variable "jobs_bucket" {
  default = "skypilot-jobs"
}

variable "db_instance_class" {
  type    = string
  default = "db.m8gd.large"
}

variable "db_allocated_storage" {
  type    = number
  default = 500 # GiB
}

variable "db_postgres_version" {
  type    = string
  default = "17.5"
}
