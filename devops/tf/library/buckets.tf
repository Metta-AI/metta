resource "aws_s3_bucket" "buckets" {
  for_each = toset(var.s3_buckets)
  bucket   = each.value
}
