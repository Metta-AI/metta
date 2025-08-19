# Database Reset and Import Scripts

This directory contains scripts to reset your library database and import fresh data from various sources.

## Quick Start

### Basic Reset (Fast)

```bash
pnpm quick-reset
```

- Resets database to fresh state
- Imports papers from Asana
- Takes ~6 seconds

### Full Reset (Complete)

```bash
pnpm reset-and-import
```

- Resets database to fresh state
- Imports papers from Asana
- Extracts authors from papers
- Enriches author metadata
- Extracts institutions from papers
- Shows database summary
- Takes ~30-60 seconds

### Preview Changes (Safe)

```bash
pnpm reset-and-import --dry-run
```

- Shows what will be executed without making changes
- Good for verifying prerequisites and planned steps

## Script Options

### reset-and-import-all.ts

The comprehensive reset and import script with full data processing.

**Options:**

- `--dry-run` - Preview execution without making changes
- `--skip-reset` - Skip database reset, only run import scripts

**Example Usage:**

```bash
# Full reset and import
pnpm reset-and-import

# Preview what would happen
pnpm reset-and-import --dry-run

# Only run imports without resetting database
pnpm reset-and-import --skip-reset
```

### quick-reset.ts

A lightweight script for fast development cycles.

**What it does:**

- Resets database to clean state
- Imports papers from Asana
- Skips author/institution processing for speed

**Example Usage:**

```bash
pnpm quick-reset
```

## Prerequisites

### Required Environment Variables

- `DATABASE_URL` - PostgreSQL connection string
- `ASANA_API_KEY` - For importing papers from Asana project

### Optional Environment Variables

- `ANTHROPIC_API_KEY` - For LLM-enhanced institution extraction

### Setup

1. Copy `.env.local.example` to `.env.local` (if not already done)
2. Add your API keys to `.env.local`
3. Ensure PostgreSQL is running
4. Run `pnpm prisma generate` to ensure Prisma client is ready

## What Gets Imported

### Papers from Asana

- Paper titles, abstracts, links
- Custom fields (topics, domains, etc.)
- Tags and metadata
- Asana task links for reference

### Author Information

- Extracted from paper metadata
- Author names and affiliations
- Additional metadata from author enrichment

### Institution Data

- University names, research labs, companies
- Extracted from paper text using LLM parsing
- Institutional affiliations for authors

## Troubleshooting

### Common Issues

**Database Connection Error**

```
Error: Environment variable not found: DATABASE_URL
```

→ Check your `.env.local` file has `DATABASE_URL` set

**Asana Import Fails**

```
Error: ASANA_API_KEY environment variable is required
```

→ Add your Asana API key to `.env.local`

**Institution Extraction Issues**

```
Warning: ANTHROPIC_API_KEY not found, falling back to mock parsing
```

→ Add your Anthropic API key for better institution extraction, or ignore if rule-based parsing is sufficient

**Database Table Missing Error**

```
Error: The table `public.paper` does not exist in the current database
```

→ This indicates the database schema wasn't applied after reset. The scripts now automatically run `prisma db push` after reset to fix this.

### Verification

After running scripts, verify the data:

```bash
# View database contents
tsx scripts/show-database-contents.ts

# Open Prisma Studio for detailed view
pnpm db:studio
```

## Development Workflow

### Daily Development

```bash
pnpm quick-reset  # Fast reset for testing
pnpm dev          # Start development server
```

### Full Data Refresh

```bash
pnpm reset-and-import  # Complete data pipeline
pnpm dev               # Start with full dataset
```

### Safe Testing

```bash
pnpm reset-and-import --dry-run  # Preview changes
pnpm reset-and-import            # Execute if preview looks good
```

## Script Execution Flow

### Full Reset (`reset-and-import`)

1. **Prerequisites Check** - Verify environment variables
2. **Database Reset** - `prisma migrate reset --force`
3. **Schema Application** - `prisma db push` (recreate tables)
4. **Asana Import** - Import papers and metadata
5. **Author Extraction** - Parse authors from papers
6. **Author Enrichment** - Add metadata to authors (optional)
7. **Institution Extraction** - Extract institutional affiliations
8. **Verification** - Show database summary

### Quick Reset (`quick-reset`)

1. **Database Reset** - `prisma migrate reset --force`
2. **Schema Application** - `prisma db push` (recreate tables)
3. **Asana Import** - Import papers and metadata
4. **Summary** - Quick completion report

## Integration with Existing Scripts

These reset scripts orchestrate existing individual scripts:

- `pnpm pull-asana-papers` - Asana import
- `pnpm extract-authors` - Author extraction
- `pnpm enrich-metadata` - Author enrichment
- `pnpm extract-institutions` - Institution extraction

You can still run these individually if needed for targeted updates.
