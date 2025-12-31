resource "aws_s3_bucket" "skypilot_jobs" {
  bucket = var.jobs_bucket
}
