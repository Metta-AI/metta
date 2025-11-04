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

# Create shared ECR repository for all cronjobs (dashboard metrics, PR similarity cache, etc.)
resource "aws_ecr_repository" "cronjobs" {
  name                 = "metta-cronjobs"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Add lifecycle policy to clean up old cronjob images
resource "aws_ecr_lifecycle_policy" "cronjobs" {
  repository = aws_ecr_repository.cronjobs.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 30 images across all cronjobs"
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

# Replicate to other regions
data "aws_caller_identity" "current" {}

resource "aws_ecr_replication_configuration" "regions" {
  replication_configuration {
    rule {
      dynamic "destination" {
        for_each = var.replication_regions
        content {
          region      = destination.value
          registry_id = data.aws_caller_identity.current.account_id
        }
      }
    }
  }
}
