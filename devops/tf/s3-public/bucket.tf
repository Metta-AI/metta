resource "aws_s3_bucket" "softmax_public" {
  bucket = "softmax-public"
}

import {
  to = aws_s3_bucket.softmax_public
  id = "softmax-public"
}
