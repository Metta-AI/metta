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

# Output configuration values for FastAPI service
output "sandbox_config" {
  description = "Configuration values to be used by sandbox-manager FastAPI service"
  value = {
    vpc_id            = module.sandbox_vpc.vpc_id
    subnet_id         = length(module.sandbox_vpc.public_subnets) > 0 ? module.sandbox_vpc.public_subnets[0] : null
    security_group_id = aws_security_group.sandbox.id
    instance_profile  = aws_iam_instance_profile.sandbox.name
    region            = var.region
  }
}
