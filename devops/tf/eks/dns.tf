# private zone for softmax
resource "aws_default_vpc" "default" {}

resource "aws_route53_zone" "softmax" {
  name = "softmax"
  vpc {
    vpc_id = aws_default_vpc.default.id
  }
}
