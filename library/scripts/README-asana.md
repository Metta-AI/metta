# Asana Paper Collection Script

This script pulls research papers from an Asana project and imports them into the library database.

## Setup

### 1. Asana Configuration

1. **Get your Asana Personal Access Token**:
   - Go to [Asana My Apps](https://app.asana.com/0/my-apps)
   - Create a new Personal Access Token
   - Copy the token and add it to your `.env.local` file as `ASANA_API_KEY`

2. **Find your Project ID**:
   - Navigate to your papers project in Asana
   - Look at the URL: `https://app.asana.com/0/PROJECT_ID/...`
   - Copy the PROJECT_ID and add it to your `.env.local` file as `ASANA_PAPERS_PROJECT_ID`

3. **Optional: Configure Custom Fields**:
   - If your Asana tasks use custom fields for paper metadata (arXiv ID, paper links, abstracts), you can configure these field IDs in your `.env.local` file
   - To find field IDs, you can run the script once and examine the output, or use Asana's API explorer

### 2. Environment Variables

Copy `.env.example` to `.env.local` and configure:

```bash
# Required
ASANA_API_KEY=your-asana-personal-access-token
ASANA_PAPERS_PROJECT_ID=your-papers-project-id

# Optional
ASANA_WORKSPACE_ID=your-workspace-id
ASANA_PAPER_LINK_FIELD_ID=paper-link-field-id
ASANA_ARXIV_ID_FIELD_ID=arxiv-id-field-id
ASANA_ABSTRACT_FIELD_ID=abstract-field-id
```

### 3. Asana Project Structure

The script expects your Asana project to contain tasks where:

- **Task Name**: The paper title
- **Task Description**: Paper abstract (optional)
- **Custom Fields**: Paper metadata like arXiv ID, paper links, etc.
- **Tags**: Paper categories/topics
- **Completed Tasks**: Are skipped (assuming they're already processed)

## Usage

Run the script:

```bash
pnpm pull-asana-papers
```

The script will:

1. Connect to Asana using your API key
2. Fetch all tasks from the specified project
3. Extract paper metadata from each task
4. Import papers into the database (skipping duplicates)
5. Show a summary of imported, skipped, and error papers

## Data Mapping

| Asana Field | Database Field | Notes |
|-------------|----------------|-------|
| Task Name | `title` | Required |
| Task Description | `abstract` | Optional |
| Custom Fields | `link`, `externalId` | Extracted based on field names/IDs |
| Tags | `tags` | Array of tag names |
| Task URL | Not stored | Could be added as metadata |
| Created Date | `createdAt` | Automatically set |

## Paper Sources

The script automatically detects paper sources:

- **arXiv**: If arXiv ID or arXiv.org URL is found
- **bioRxiv**: If biorxiv.org URL is found
- **External**: If other URL is found
- **Asana**: Default for tasks without external links

## Duplicate Detection

Papers are considered duplicates if they have the same:
- External ID (e.g., arXiv ID)
- Link URL
- Title

Duplicates are skipped during import.

## Troubleshooting

### "ASANA_API_KEY environment variable is required"
- Make sure you've set `ASANA_API_KEY` in your `.env.local` file
- Verify the token is valid by testing it in Asana's API explorer

### "ASANA_PAPERS_PROJECT_ID environment variable is required"
- Make sure you've set the project ID in your `.env.local` file
- Double-check the project ID from your Asana project URL

### "Asana API Error: 403"
- Your API token doesn't have access to the project
- Make sure you're a member of the project or have appropriate permissions

### "No papers to import"
- Check that your project has tasks
- Verify the tasks aren't all marked as completed
- Check that task names aren't empty

## Custom Fields

For better data extraction, you can set up custom fields in your Asana project:

1. **Paper Link**: URL to the paper (arXiv, journal, etc.)
2. **arXiv ID**: Just the arXiv identifier (e.g., "2301.12345")
3. **Abstract**: Full paper abstract if not in task description

Configure the field IDs in your `.env.local` to help the script find these fields automatically.
