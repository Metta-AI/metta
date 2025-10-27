# Library Infrastructure

Terraform configuration for the Softmax Library application infrastructure.

## Components

- **PostgreSQL Database**: RDS instance for application data
- **S3 Buckets**: Storage for user uploads and assets
- **AWS SES**: Email notifications for mentions, comments, and replies

## SES Email Setup

The SES configuration sends emails from `library@softmax.com` (configurable via `ses_from_email` variable).

### DNS Configuration

After Spacelift applies the changes, add these DNS records to **Cloudflare** (not Route53):

1. View outputs in:

   - **Spacelift UI**: Stack → Resources → Outputs tab
   - **GitHub**: PR comment from Spacelift shows outputs after apply
   - **CLI**:

   ```bash
   spacectl stack output library
   ```

2. Look for `ses_domain_verification_record` (1 TXT record) and `ses_dkim_records` (3 CNAME records)

3. Add all 4 records to the `softmax.com` domain in Cloudflare

Verification typically completes within 5-30 minutes.

### Variables

- `ses_from_email`: Email address for notifications (default: `library@softmax.com`)
- `ses_from_name`: Display name for emails (default: `Softmax Library`)
