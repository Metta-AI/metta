# Fileset Export Tool

Export code filesets to Google Drive for Asana AI consumption.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Google Drive API Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Google Drive API
4. Go to "Credentials" → "Create Credentials" → "OAuth 2.0 Client IDs"
5. Choose "Desktop application"
6. Download the credentials JSON file
7. Save it as `tools/gdrive_upload/credentials.json`

### 3. Directory Structure

```
tools/gdrive_upload/
├── export_fileset.py      # Main CLI script
├── aggregate.py           # File discovery and aggregation
├── gdrive_io.py          # Google Drive operations
├── filesets.yml          # Configuration file (committed)
├── requirements.txt      # Python dependencies
├── README.md             # This documentation
├── credentials.json      # Google OAuth credentials (YOU ADD - gitignored)
├── token.json           # Auto-generated after first auth (gitignored)
├── manifest.json        # Auto-generated state file (gitignored)
└── filesets.local.yml   # Optional local overrides (gitignored)
```

**Important**: The repository includes `filesets.yml` with sensible defaults, but three files are gitignored and user-specific:
- `credentials.json` - Your Google OAuth credentials
- `token.json` - Auto-generated authentication token
- `manifest.json` - Your personal Drive file IDs and state

## Usage

### Export a single fileset

```bash
cd /path/to/your/repo
python tools/gdrive_upload/export_fileset.py --fileset metta-readme
```

### Export all filesets

```bash
python tools/gdrive_upload/export_fileset.py --all
```

### Dry run (preview without uploading)

```bash
python tools/gdrive_upload/export_fileset.py --fileset mettagrid --dry-run
```

### Verbose logging

```bash
python tools/gdrive_upload/export_fileset.py --fileset metta-readme --verbose
```

## Configuration

### Fileset Configuration
Edit `tools/gdrive_upload/filesets.yml` to customize:
- **Drive settings**: Folder name where documents are stored
- **Filesets**: Include/exclude patterns for each fileset

The committed `filesets.yml` includes sensible defaults for the Metta repository structure.

### Local Overrides (Optional)
Create `tools/gdrive_upload/filesets.local.yml` to override settings without modifying the committed configuration:

```yaml
# Override drive folder name
drive:
  parent_folder_name: "My Custom Exports"

# Add personal filesets or modify existing ones
filesets:
  my-custom-fileset:
    includes: ["my-experiments/**"]
    excludes: ["**/*.log"]
```

The tool will load `filesets.yml` first, then merge any settings from `filesets.local.yml`.

## First Run

1. The first time you run the script, it will open a browser for OAuth authentication
2. Grant the requested permissions
3. The token will be saved for future runs
4. A "Metta Exports" folder will be created in your Google Drive
5. Documents will be created with public "anyone with link" viewing permissions

## Output

Each fileset becomes a single Google Doc containing:

- Header with fileset name, timestamp, and git commit SHA
- Table of contents with file paths and sizes
- Full content of each file with syntax highlighting
- Automatic skipping of binary files and files > 500KB

## Troubleshooting

### Authentication Issues

- Delete `tools/gdrive_upload/token.json` and re-run to re-authenticate
- Check that `tools/gdrive_upload/credentials.json` is valid OAuth 2.0 desktop credentials

### Permission Issues

- If "anyone with link" permissions fail, your Google Workspace may restrict sharing
- Documents will still be created but will be private to your account
- Consider using "Publish to web" feature manually for such documents

### Size Limits

- Individual files > 500KB are skipped
- Total aggregated content is limited to 8MB
- Large filesets may need include/exclude pattern adjustments

### API Limits

- Google Drive API has rate limits
- The tool includes exponential backoff retry logic
- Very large operations may need to be broken up

## File States

The `manifest.json` file tracks:

- **file_id**: Google Drive file ID (stable across updates)
- **web_link**: Public viewing URL
- **last_hash**: SHA-256 of content (for change detection)

Documents are updated in-place when content changes, preserving the same URL.
