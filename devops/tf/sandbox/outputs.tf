output "vpc_id" {
  description = "ID of the sandbox VPC"
  value       = module.sandbox_vpc.vpc_id
}

output "public_subnet_ids" {
  description = "IDs of public subnets where sandbox instances will be launched"
  value       = module.sandbox_vpc.public_subnets
}

output "security_group_id" {
  description = "ID of the security group for sandbox instances"
  value       = aws_security_group.sandbox.id
}

output "instance_profile_name" {
  description = "Name of the IAM instance profile for sandbox instances"
  value       = aws_iam_instance_profile.sandbox.name
}

output "instance_role_arn" {
  description = "ARN of the IAM role for sandbox instances"
  value       = aws_iam_role.sandbox_instance.arn
}

output "sandbox_manager_user_name" {
  description = "Name of the IAM user for sandbox manager service"
  value       = aws_iam_user.sandbox_manager.name
}

output "sandbox_manager_access_key_id" {
  description = "Access key ID for sandbox manager (store in k8s secret)"
  value       = aws_iam_access_key.sandbox_manager.id
  sensitive   = true
}

output "sandbox_manager_secret_access_key" {
  description = "Secret access key for sandbox manager (store in k8s secret)"
  value       = aws_iam_access_key.sandbox_manager.secret
  sensitive   = true
}

# Output configuration values for FastAPI service
output "sandbox_config" {
  description = "Configuration values to be used by sandbox-manager FastAPI service"
  value = {
    vpc_id                = module.sandbox_vpc.vpc_id
    subnet_id             = module.sandbox_vpc.public_subnets[0] # Use first AZ
    security_group_id     = aws_security_group.sandbox.id
    instance_profile      = aws_iam_instance_profile.sandbox.name
    region                = var.region
    default_instance_type = var.default_instance_type
  }
}
