provider "aws" {
  region = "us-east-1"
}

# Create ECR repository
resource "aws_ecr_repository" "metta" {
  name                 = "metta"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Add lifecycle policy to clean up old images
resource "aws_ecr_lifecycle_policy" "metta" {
  repository = aws_ecr_repository.metta.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 30 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 30
      }
      action = {
        type = "expire"
      }
    }]
  })
}
