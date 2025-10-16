output "postgres_endpoint" {
  value = aws_db_instance.postgres.endpoint
}

output "postgres_url" {
  value = nonsensitive(local.postgres_url)
}

# SES DNS Records - Add these to Cloudflare for email verification and DKIM
output "ses_domain_verification_record" {
  description = "TXT record to add to Cloudflare for SES domain verification"
  value = {
    name  = "_amazonses.${local.ses_domain}"
    type  = "TXT"
    value = aws_ses_domain_identity.library.verification_token
    ttl   = 600
  }
}

output "ses_dkim_records" {
  description = "CNAME records to add to Cloudflare for DKIM signing (add all 3)"
  value = [
    for idx in range(3) : {
      name  = "${aws_ses_domain_dkim.library.dkim_tokens[idx]}._domainkey.${local.ses_domain}"
      type  = "CNAME"
      value = "${aws_ses_domain_dkim.library.dkim_tokens[idx]}.dkim.amazonses.com"
      ttl   = 600
    }
  ]
}
