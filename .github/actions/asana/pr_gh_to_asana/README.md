# PR to Asana Integration with VCR

This GitHub Action integrates Pull Requests with Asana tasks using VCR (Video Cassette Recorder) for HTTP request recording and replaying.

## VCR Integration

The codebase uses VCR to record HTTP interactions with external APIs (GitHub and Asana) and replay them during testing. This provides several benefits:

- **Consistent Testing**: HTTP responses are recorded and replayed, ensuring consistent test results
- **Offline Development**: Once recorded, tests can run without internet connectivity
- **Performance**: No need to make real API calls during testing
- **Security**: Sensitive data (tokens, headers) is filtered out of recordings

## Configuration

VCR is configured with the following settings:

- **Cassette Library**: `cassettes/` directory
- **Record Mode**: `once` (record if cassette doesn't exist, replay if it does)
- **Match On**: URI and method
- **Filtered Headers**: Authorization, Accept, X-GitHub-Api-Version
- **Filtered Query Parameters**: access_token
- **Compressed Response**: Decoded for readability

## Usage

### Recording New Cassettes

To record new HTTP interactions:

1. Delete existing cassettes (if any)
2. Run the script with real API tokens
3. VCR will record all HTTP interactions
4. Cassettes will be saved in `cassettes/` directory

### Replaying Cassettes

Once cassettes are recorded:

1. The script will automatically use recorded responses
2. No real API calls will be made
3. Tests will run faster and more reliably

### Cassette Files

The following cassette files are created:

- `pr_{repo}_{pr_number}.yaml` - GitHub PR data
- `validate_task_{gid}.yaml` - Asana task validation
- `search_tasks.yaml` - Asana task search
- `create_task.yaml` - Asana task creation
- `update_task_{gid}.yaml` - Asana task updates
- `ensure_github_url_in_task.yaml` - GitHub URL attachment
- `get_comments_{task_id}.yaml` - Asana comments
- `update_comment_{story_id}.yaml` - Comment updates
- `add_comment_{review_id}.yaml` - New comments
- `project_tasks_{project_id}.yaml` - Project tasks

## Security

Sensitive information is automatically filtered:

- Authorization headers
- GitHub API version headers
- Access tokens in query parameters
- Personal information in responses

## Dependencies

- `vcrpy>=6.0.0` - VCR library
- `requests>=2.31.0` - HTTP client

## Development

When modifying the code:

1. Update VCR configuration if needed
2. Delete relevant cassettes to re-record
3. Test with real API calls first
4. Commit cassettes for consistent testing
