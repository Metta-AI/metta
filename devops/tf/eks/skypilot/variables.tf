variable "zone_domain" {
    default = "softmax"
}

variable "subdomain" {
    default = "skypilot"
}


variable "lambda_ai_secret_arn" {
    default = "arn:aws:secretsmanager:us-east-1:751442549699:secret:LambdaAI-qo6uuJ"
}

variable "namespace" {
    default = "skypilot"
}

variable "ssm_parameter_name" {
    default = "/skypilot/api_url"
}
