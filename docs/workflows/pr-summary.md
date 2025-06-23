# PR Summary Workflow Documentation

## Overview

This workflow automatically generates weekly summaries of merged pull requests and posts them to Discord. It uses a two-phase approach with Google's Gemini AI models and implements smart caching to minimize API calls.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   GitHub API    │────▶│ PR Digest Action │────▶│ Summary Script  │
│  (Fetch PRs)    │     │ (with caching)   │     │ (Gemini AI)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
                                                  ┌─────────────────┐
                                                  │ Discord Webhook │
                                                  │    Action       │
                                                  └─────────────────┘
```

## Components

### 1. Main Workflow (`.github/workflows/pr_summary.yml`)

**Schedule**: Runs every Friday at 7 PM UTC (`0 19 * * 5`)

**Manual Triggers**:
- `days_to_scan`: Number of days to analyze (7, 14, or 30)
- `force_refresh`: Force cache refresh
- `skip_discord`: Skip Discord posting for testing (boolean)

**Steps**:
1. Fetch PR digest using custom action
2. Generate AI summary using Gemini
3. Display summary in workflow logs (always)
4. Post summary to Discord (conditional)
5. Upload artifacts for debugging

### 2. PR Digest Action (`.github/actions/pr-digest/`)

**Purpose**: Fetches merged PRs with intelligent caching to reduce GitHub API calls.

**Files**:
- `action.yml`: Action metadata and configuration
- `fetch_pr_digest.py`: Main script (note: file name has underscore, not hyphen)

**Key Features**:
- **Incremental caching**: Only fetches new PRs since last run
- **Cache pruning**: Removes entries older than 60 days
- **Rate limit handling**: Automatic retry with backoff
- **Diff truncation**: Limits diff size to prevent token overflow

**Inputs**:
- `github-token`: GitHub API access token
- `repository`: Target repository (default: current)
- `days`: Lookback period (default: 7)
- `diff-limit`: Max diff characters (default: 20,000)
- `force-refresh`: Bypass cache (default: false)

**Outputs**:
- `digest-file`: Path to JSON with PR data
- `pr-count`: Number of PRs found
- `cache-stats`: Cache hit/miss statistics
- `date-range`: Date range scanned

### 3. Summary Generation Script (`.github/scripts/generate_pr_summary.py`)

**Two-Phase AI Approach**:

**Phase 1**: Individual PR Summaries
- Model: `gemini-2.5-flash-preview-05-20` (fast, efficient)
- Generates concise technical summaries for each PR
- Parallel processing with configurable workers
- Per-PR caching to avoid regeneration

**Phase 2**: Consolidated Summary
- Model: `gemini-2.5-pro-preview-05-06` (powerful, synthesis)
- Synthesizes individual summaries into cohesive report
- Structured format with executive summary, API changes, and detailed breakdown

**Features**:
- Model-aware caching (regenerates if model changes)
- Configurable via environment variables
- Special prompts for Gemini 2.5 models with thinking capabilities
- Intermediate output saving for debugging

### 4. Discord Webhook Action (`.github/actions/discord-webhook/`)

**Purpose**: Posts content to Discord with automatic message splitting.

**Files**:
- `action.yml`: Action metadata and configuration
- `discord_webhook.py`: Main posting script

**Key Features**:
- **Automatic splitting**: Handles Discord's 2000 character limit
- **Smart boundaries**: Splits at paragraphs/lines for readability
- **Rate limiting**: 0.5s delay between messages
- **Security**: Sanitizes @everyone/@here mentions
- **Error handling**: Proper HTTP error reporting

## Configuration

### Required Secrets

```yaml
GITHUB_TOKEN        # GitHub API access (usually automatic)
GEMINI_API_KEY      # Google AI API key
DISCORD_WEBHOOK_URL # Discord webhook for posting
```

### Environment Variables

**For PR Digest**:
```bash
REPOSITORY="owner/repo"    # Target repository
DAYS_TO_SCAN="7"          # Lookback period
DIFF_LIMIT="20000"        # Max diff size
FORCE_REFRESH="false"     # Cache bypass
```

**For Summary Generation**:
```bash
PHASE1_MODEL="gemini-2.5-flash-preview-05-20"  # PR summaries
PHASE2_MODEL="gemini-2.5-pro-preview-05-06"    # Final synthesis
USE_PARALLEL="true"                             # Parallel processing
MAX_WORKERS="5"                                 # Parallel workers
```

## Caching Strategy

### PR Digest Cache

**Structure**:
```json
{
  "version": "1.0",
  "repository": "owner/repo",
  "last_updated": "2024-01-01T00:00:00Z",
  "last_pr_merged_at": "2024-01-01T00:00:00Z",
  "pr_digests": [
    {
      "number": 123,
      "title": "PR Title",
      "body": "Description",
      "merged_at": "2024-01-01T00:00:00Z",
      "html_url": "https://github.com/...",
      "diff": "...",
      "author": "username",
      "labels": ["bug", "enhancement"]
    }
  ]
}
```

**Cache Key**: `pr-digest-{repository}-week-{week_number}-{year}`

### PR Summary Cache

**Structure**:
```json
{
  "model": "gemini-2.5-flash-preview-05-20",
  "summaries": {
    "123-2024-01-01T00:00:00Z": {
      "number": 123,
      "title": "PR Title",
      "url": "https://github.com/...",
      "summary": "AI-generated summary..."
    }
  }
}
```

**Cache Key**: `{pr_number}-{merged_at}` (invalidated on model change)

## Output Format

The final Discord message follows this structure:

```markdown
## Summary of changes from YYYY-MM-DD to YYYY-MM-DD

**Executive Summary:**
A concise paragraph summarizing the week's key technical developments...

**Internal API Changes:**
- Change description [#123](https://github.com/...)
- Another change [#456](https://github.com/...)

**Detailed Breakdown:**

### Feature Category
- PR description with technical details [#789](https://github.com/...)
- Related PR [#012](https://github.com/...)

### Bug Fixes
- Fix description [#345](https://github.com/...)
```

## Error Handling

1. **GitHub API Rate Limiting**: Automatic retry with exponential backoff
2. **Gemini API Errors**: Graceful degradation with error placeholders
3. **Discord Rate Limiting**: Built-in delays and retry logic
4. **Cache Corruption**: Falls back to full refresh

## Performance Optimizations

1. **Incremental Updates**: Only fetch new PRs since last run
2. **Parallel Processing**: Phase 1 summaries generated concurrently
3. **Smart Caching**: Two-level caching reduces API calls
4. **Diff Truncation**: Prevents token overflow in AI models

## Usage Examples

### Manual Trigger with 14-day Lookback
```yaml
workflow_dispatch:
  inputs:
    days_to_scan: "14"
    force_refresh: false
    skip_discord: false
```

### Test Mode (No Discord)
```yaml
workflow_dispatch:
  inputs:
    days_to_scan: "7"
    force_refresh: false
    skip_discord: true
```

### Force Full Refresh
```yaml
workflow_dispatch:
  inputs:
    days_to_scan: "7"
    force_refresh: true
    skip_discord: false
```

## File Structure Summary

```
.github/
├── actions/
│   ├── discord-webhook/
│   │   ├── action.yml
│   │   └── discord_webhook.py
│   └── pr-digest/
│       ├── action.yml
│       └── fetch_pr_digest.py
├── docs/
│   └── workflows/
│       └── pr-summary.md
├── scripts/
│   └── generate_pr_summary.py
└── workflows/
    └── pr_summary.yml
```
