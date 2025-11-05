# S3 bucket for researcher sandbox outputs (training checkpoints, logs, etc.)
resource "aws_s3_bucket" "sandbox_outputs" {
  bucket = "softmax-sandbox-outputs"

  tags = merge(local.tags, {
    Name        = "sandbox-outputs"
    Description = "Researcher sandbox training outputs, checkpoints, and logs"
  })
}

# Enable versioning for safety (researchers can recover accidentally deleted files)
resource "aws_s3_bucket_versioning" "sandbox_outputs" {
  bucket = aws_s3_bucket.sandbox_outputs.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Lifecycle policy to clean up old versions and reduce costs
resource "aws_s3_bucket_lifecycle_configuration" "sandbox_outputs" {
  bucket = aws_s3_bucket.sandbox_outputs.id

  rule {
    id     = "cleanup-old-versions"
    status = "Enabled"

    # Delete old versions after 30 days
    noncurrent_version_expiration {
      noncurrent_days = 30
    }

    # Delete incomplete multipart uploads after 7 days
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  rule {
    id     = "expire-old-outputs"
    status = "Enabled"

    # Move to cheaper storage after 90 days
    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }

    # Delete after 180 days (6 months)
    expiration {
      days = 180
    }
  }
}

# Block public access (only researchers via IAM can access)
resource "aws_s3_bucket_public_access_block" "sandbox_outputs" {
  bucket = aws_s3_bucket.sandbox_outputs.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Server-side encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "sandbox_outputs" {
  bucket = aws_s3_bucket.sandbox_outputs.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
