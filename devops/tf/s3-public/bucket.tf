locals {
  bucket_name              = "softmax-public"
}

resource "aws_s3_bucket" "softmax_public" {
  bucket = local.bucket_name
}

resource "aws_s3_bucket_policy" "softmax_public" {
  bucket = aws_s3_bucket.softmax_public.id
  policy = jsonencode(
    {
      Statement = [
        {
          Action    = "s3:GetObject"
          Effect    = "Allow"
          Principal = "*"
          Resource  = "${aws_s3_bucket.softmax_public.arn}/*"
          Sid       = "AllowPublicRead"
        }
      ]
      Version = "2012-10-17"
    }
  )
}

resource "aws_s3_bucket_cors_configuration" "softmax_public" {
  bucket = aws_s3_bucket.softmax_public.id

  cors_rule {
    allowed_headers = [
      "*",
    ]
    allowed_methods = [
      "GET",
      "HEAD",
    ]
    allowed_origins = [
      "*",
    ]
    expose_headers = [
      "ETag",
      "x-amz-meta-custom-header",
    ]
  }
}

import {
  to = aws_s3_bucket.softmax_public
  id = "softmax-public"
}
