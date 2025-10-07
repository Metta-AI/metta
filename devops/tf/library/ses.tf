# AWS SES Configuration for Email Notifications
# This module sets up AWS Simple Email Service for sending email notifications
# for user mentions, comments, and replies in the library application.

# Email identity for sending notifications
# Note: This requires DNS verification before emails can be sent
resource "aws_ses_email_identity" "library_notifications" {
  email = "noreply@${var.domain}"
}

# Domain identity for better deliverability and DKIM signing
resource "aws_ses_domain_identity" "library" {
  domain = var.domain
}

# DKIM signing for improved email authentication
resource "aws_ses_domain_dkim" "library" {
  domain = aws_ses_domain_identity.library.domain
}

# IAM user dedicated to SES SMTP access
resource "aws_iam_user" "ses_smtp" {
  name = "softmax-library-ses-smtp"
  path = "/system/"

  tags = {
    Purpose     = "SES SMTP access for library notifications"
    Application = "softmax-library"
  }
}

# Policy allowing the IAM user to send emails via SES
resource "aws_iam_user_policy" "ses_smtp" {
  name = "ses-smtp-access"
  user = aws_iam_user.ses_smtp.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ses:SendEmail",
          "ses:SendRawEmail"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "ses:FromAddress" = "noreply@${var.domain}"
          }
        }
      }
    ]
  })
}

# Generate access key for SMTP authentication
# The secret will be automatically converted to SMTP password format
resource "aws_iam_access_key" "ses_smtp" {
  user = aws_iam_user.ses_smtp.name
}

# Configuration set for tracking email metrics and events
resource "aws_ses_configuration_set" "library" {
  name = "softmax-library-notifications"

  delivery_options {
    tls_policy = "Require"
  }

  reputation_metrics_enabled = true
}

# Event destination for tracking bounces and complaints
resource "aws_ses_event_destination" "cloudwatch" {
  name                   = "cloudwatch-metrics"
  configuration_set_name = aws_ses_configuration_set.library.name
  enabled                = true
  matching_types         = ["send", "reject", "bounce", "complaint", "delivery"]

  cloudwatch_destination {
    default_value  = "default"
    dimension_name = "ses:configuration-set"
    value_source   = "emailHeader"
  }
}

# Reference existing Route53 hosted zone for the parent domain
# This zone is managed by external-dns from the EKS cluster
data "aws_route53_zone" "library" {
  name         = "softmax-research.net"
  private_zone = false
}

# Route53 record for SES domain verification
resource "aws_route53_record" "ses_verification" {
  zone_id = data.aws_route53_zone.library.zone_id
  name    = "_amazonses.${var.domain}"
  type    = "TXT"
  ttl     = 600
  records = [aws_ses_domain_identity.library.verification_token]
}

# Route53 records for DKIM signing (3 records)
resource "aws_route53_record" "ses_dkim" {
  count   = 3
  zone_id = data.aws_route53_zone.library.zone_id
  name    = "${aws_ses_domain_dkim.library.dkim_tokens[count.index]}._domainkey.${var.domain}"
  type    = "CNAME"
  ttl     = 600
  records = ["${aws_ses_domain_dkim.library.dkim_tokens[count.index]}.dkim.amazonses.com"]
}

# Convert AWS secret access key to SES SMTP password format
# SES requires a special password derived from the IAM secret using AWS SigV4
data "external" "ses_smtp_password" {
  program = [
    "python3",
    "${path.module}/scripts/convert_ses_password.py",
    aws_iam_access_key.ses_smtp.secret
  ]
}

# Local values for constructing SES connection details
locals {
  ses_smtp_endpoint = "email-smtp.${var.region}.amazonaws.com"
  ses_from_email    = "noreply@${var.domain}"

  # SES SMTP credentials from IAM access key
  ses_smtp_username = aws_iam_access_key.ses_smtp.id
  # SES SMTP password converted from AWS secret using SigV4 algorithm
  ses_smtp_password = data.external.ses_smtp_password.result.password
}
