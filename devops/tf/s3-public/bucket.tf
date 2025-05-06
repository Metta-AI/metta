locals {
  bucket_name              = "softmax-public"
  david_bloomin_account_id = "767406518141"
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
        },
        {
          Action = "s3:*"
          Effect = "Allow"
          Principal = {
            AWS = [
              "arn:aws:iam::${local.david_bloomin_account_id}:role/ecsTaskExecutionRole",
              "arn:aws:iam::${local.david_bloomin_account_id}:role/s3_sync",
              "arn:aws:iam::${local.david_bloomin_account_id}:root",
            ]
          }
          Resource = [
            aws_s3_bucket.softmax_public.arn,
            "${aws_s3_bucket.softmax_public.arn}/*",
          ]
          Sid = "FullAccessFromDavidBloominAccount"
        },
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
