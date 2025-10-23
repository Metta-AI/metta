# Dashboard Templates

This directory contains exported JSON files from Datadog dashboards, used as references when migrating to Terraform.

## Usage

Export a dashboard using the export script:

```bash
# From the devops/datadog directory
./export_dashboard.py abc-123-def > templates/my_dashboard.json
```

These JSON files are:
- **Reference only** - Use them to understand widget structure when writing Terraform
- **Gitignored** - Won't be committed to version control (they can be large and change frequently)
- **Temporary** - Delete after successfully migrating the dashboard to Terraform

## After Migration

Once you've successfully converted a dashboard to Terraform:

1. Import the dashboard: `terraform import datadog_dashboard.name dashboard-id`
2. Verify with `terraform plan` (should show no changes)
3. Delete the JSON template file
4. Document the migration in your PR
