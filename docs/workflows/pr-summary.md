# PR Newsletter Workflow

## Overview

This workflow automatically generates newsletters summarizing merged pull requests and posts them to Discord. It uses a
two-phase approach with Google's Gemini AI models and implements smart caching to minimize API calls.

## Schedule

The newsletter runs on a **Monday/Wednesday/Friday** schedule at 5 PM PST (1 AM GMT the following day):

- **Monday**: Analyzes PRs from the last 3 days (covering the weekend)
- **Wednesday**: Analyzes PRs from the last 2 days
- **Friday**: Analyzes PRs from the last 2 days

### Schedule Configuration

```yaml
schedule:
  - cron: '0 1 * * 2,4,6' # Tue/Thu/Sat at 1 AM GMT = Mon/Wed/Fri at 5 PM PST
```

The cron runs on Tuesday/Thursday/Saturday GMT because 1 AM GMT corresponds to 5 PM PST the previous day.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   GitHub API    │────▶│   PR Digest      │────▶│ Gemini Analysis │
│  (Fetch PRs)    │     │ (with caching)   │     │ (AI Summaries)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
                                                  ┌─────────────────┐
                                                  │ Discord Webhook │
                                                  │    Posting      │
                                                  └─────────────────┘
```

## Key Features

### Smart Caching System

- **PR-level caching**: Individual PR summaries are cached to avoid re-analyzing
- **Incremental fetching**: Only new PRs since last run are fetched from GitHub
- **Cache pruning**: Automatic removal of entries older than 60 days
- **Per-PR summary files**: Each PR gets its own summary file in `pr-summaries/`

### Variable Lookback Period

The workflow automatically adjusts the lookback period based on the day:

- Monday runs analyze 3 days of PRs (to cover the weekend)
- Wednesday and Friday runs analyze 2 days of PRs

### Newsletter Continuity

- References previous newsletters for context
- Tracks recent shout-outs to distribute recognition
- Maintains narrative continuity across newsletters
- Shows progression and trends over time

### AI-Powered Analysis

- **Phase 1**: Individual PR summaries using `gemini-2.5-flash` (fast, efficient)
- **Phase 2**: Collection synthesis using `gemini-2.5-pro` (comprehensive analysis)
- Parallel processing for efficiency
- Model-aware caching (regenerates if model changes)

## Manual Triggering

You can manually trigger the workflow with custom parameters:

```yaml
workflow_dispatch:
  inputs:
    days_to_scan: '1' # Options: 1, 7, 14, 30
    force_refresh: false # Bypass cache
    skip_discord: false # Test mode without posting
```

## Configuration

### Required Secrets

```yaml
GITHUB_TOKEN        # GitHub API access (usually automatic)
GEMINI_API_KEY      # Google AI API key
DISCORD_WEBHOOK_URL # Discord webhook for posting
```

### Repository Variables (Optional)

```yaml
PR_NEWSLETTER_HISTORY_DAYS # Override default days for scheduled runs
```

## Output Files

The workflow generates several artifacts:

- `discord_summary_output.txt` - Formatted message for Discord
- `collection_summary_output.txt` - Full collection analysis
- `pr_summary_data.json` - Structured data for all PRs
- `pr-summaries/pr_*.txt` - Individual PR summaries (cached)
- `pr_digest_output.json` - Raw PR data from GitHub

## Discord Output Format

The newsletter includes:

1. **Statistics** - PR counts by category and impact level
2. **Development Focus** - Main themes and priorities
3. **Key Achievements** - Significant features and improvements
4. **Technical Health** - Code quality and project evolution
5. **Notable PRs** - 3-4 most impactful PRs with links
6. **API Changes** - Any internal API modifications
7. **Shout Outs** - Recognition for exceptional contributions
8. **Closing Thoughts** - Rotating creative elements (haiku, reflection, etc.)

## Cache Management

The workflow uses multiple caching strategies:

1. **GitHub Actions Cache** - Preserves `pr-summaries/` directory between runs
2. **PR Digest Cache** - Tracks which PRs have been fetched
3. **Summary Cache** - Stores AI-generated summaries per PR

Cache keys include run numbers to ensure proper invalidation while maintaining history.

## Performance Notes

- Processes up to 100 PRs efficiently with parallel AI analysis
- Typical run time: 2-5 minutes depending on PR count
- API costs minimized through aggressive caching
- Rate limiting handled automatically

## Custom Actions

### 1. Fetch Artifacts Action (`.github/actions/fetch-artifacts/`)

**Purpose**: Downloads zipped artifacts from previous successful workflow runs.

**Key Features**:

- Searches through workflow history for matching artifacts
- Downloads artifacts as ZIP files for local processing
- Supports pattern matching for artifact names
- Excludes current run to avoid self-reference

**Inputs**:

- `workflow-name`: Name of the workflow file (e.g., "generate-newsletter.yml")
- `artifact-name-pattern`: Pattern to match (e.g., "newsletter-\*")
- `num-artifacts`: Number of artifacts to collect (default: 5)
- `output-directory`: Where to save downloads (default: "downloaded-artifacts")

### 2. Discord Webhook Action (`.github/actions/discord-webhook/`)

**Purpose**: Posts content to Discord with automatic message splitting.

**Key Features**:

- Automatic splitting at 2000 character Discord limit
- Smart boundaries (paragraphs/lines) for readability
- Rate limiting protection (0.5s delay between messages)
- Sanitizes @everyone/@here mentions for security
- Supports both direct content and file input

**Inputs**:

- `webhook-url`: Discord webhook URL (required)
- `content`: Direct content to post
- `content-file`: Path to file containing content
- `suppress-embeds`: Whether to suppress link embeds (default: true)

## File Structure

```
.github/
├── actions/
│   ├── discord-webhook/
│   │   ├── action.yml
│   │   └── discord_webhook.py
│   └── fetch-artifacts/
│       ├── action.yml
│       └── fetch_artifacts.py
├── scripts/
│   ├── create_pr_digest.py      # Fetches PRs from GitHub API
│   ├── gemini_analyze_pr.py     # Individual PR analysis
│   ├── gemini_analyze_pr_digest.py  # Orchestrates full analysis
│   └── gemini_client.py         # Gemini AI client wrapper
└── workflows/
    └── generate-newsletter.yml   # Main workflow file
```

## Troubleshooting

### No PRs Found

If no PRs are found for the time period, the workflow creates minimal output files and posts a simple notification.

### Cache Issues

To force a complete refresh:

1. Manually trigger with `force_refresh: true`
2. Or delete the `pr-summaries/` directory in your repository

### Time Zone Confusion

Remember: The cron schedule runs in GMT. The day-of-week detection happens at runtime:

- Tuesday 1 AM GMT = Monday 5 PM PST
- Thursday 1 AM GMT = Wednesday 5 PM PST
- Saturday 1 AM GMT = Friday 5 PM PST

## Example Manual Runs

### Test without posting to Discord

```bash
gh workflow run generate-newsletter.yml -f skip_discord=true
```

### Generate a weekly summary

```bash
gh workflow run generate-newsletter.yml -f days_to_scan=7
```

### Force refresh all caches

```bash
gh workflow run generate-newsletter.yml -f force_refresh=true
```
