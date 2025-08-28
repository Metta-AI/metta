# Codebot MCP Server

This document describes how to use Codebot's Model Context Protocol (MCP) server to integrate AI-powered development
assistance with MCP clients like Claude Desktop.

## What is MCP?

The Model Context Protocol (MCP) is a standardized protocol that enables AI applications to interact with external
tools, data sources, and services through MCP servers. It provides a consistent interface for tools, resources, and
prompts.

## Installation

First, install codebot with MCP support:

```bash
pip install -e .
```

The MCP server requires the `mcp` package, which is included in the project dependencies.

## Starting the MCP Server

To start the MCP server:

```bash
codebot mcp-server
```

For verbose logging:

```bash
codebot mcp-server -v
```

The server will start and communicate via standard input/output (stdio), which is the standard transport mechanism for
MCP servers.

## Available Tools

The Codebot MCP server provides the following tools:

### 1. `summarize`

Generate an AI-powered summary of code files.

**Parameters:**

- `paths` (array of strings, optional): List of file or directory paths to analyze. Defaults to current directory.
- `token_limit` (integer, optional): Maximum tokens for the summary (default: 2000)
- `no_cache` (boolean, optional): Bypass cache and generate fresh summary (default: false)

**Example:**

```json
{
  "paths": ["src/", "tests/"],
  "token_limit": 4000,
  "no_cache": true
}
```

### 2. `context`

Show the context that would be sent to AI commands.

**Parameters:**

- `paths` (array of strings, optional): List of file or directory paths to analyze. Defaults to current directory.

## Available Resources

The server provides these MCP resources:

- `codebot://summaries`: AI-generated code summaries and analysis results (text/markdown)
- `codebot://context`: Context information for AI commands (text/plain)

## Integration with Claude Desktop

To use the Codebot MCP server with Claude Desktop, add the following configuration to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "codebot": {
      "command": "codebot",
      "args": ["mcp-server"]
    }
  }
}
```

### Configuration File Location

The configuration file location depends on your operating system:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/claude/claude_desktop_config.json`

### Full Configuration Example

```json
{
  "mcpServers": {
    "codebot": {
      "command": "codebot",
      "args": ["mcp-server"],
      "env": {
        "PYTHONPATH": "/path/to/your/codebot/directory"
      }
    }
  }
}
```

## Usage Examples

Once configured, you can use Codebot tools in Claude Desktop:

### Generate a Code Summary

"Use the summarize tool to analyze my Python files in the src/ directory with a 3000 token limit."

### View Code Context

"Use the context tool to show me what context would be sent for the current directory."

### Get Recent Summary

"Read the codebot://summaries resource to see the latest code analysis."

## Development Mode

For development, you can run the MCP server directly:

```bash
cd /path/to/your/project
python -m codebot.mcp_server
```

Or test it interactively using the MCP Inspector tool if available.

## Troubleshooting

### Common Issues

1. **Server not starting**: Check that all dependencies are installed (`pip install -e .`)
2. **Command not found**: Ensure codebot is in your PATH or use the full path to the executable
3. **Permission errors**: Make sure the codebot executable has proper permissions

### Debugging

Enable verbose logging to see detailed information:

```bash
codebot mcp-server -v
```

The server logs to stderr, so you can redirect output to a file:

```bash
codebot mcp-server 2> mcp_server.log
```

### Testing the Server

You can test the server manually using stdio. Start the server and send JSON-RPC messages:

```bash
codebot mcp-server
```

Then send an initialization message followed by tool calls (see MCP specification for exact message format).

## Architecture

The MCP server is implemented in `codebot/mcp_server.py` and consists of:

- **CodebotMCPServer**: Main server class that handles MCP protocol
- **Tools**: Async functions that implement codebot commands
- **Resources**: Access to generated summaries and context information
- **Transport**: Uses stdio for communication with MCP clients

The server leverages the existing codebot command infrastructure, particularly the `run_summarize_command` function and
context gathering from codeclip.

## Security Considerations

- The MCP server runs with the same permissions as the user who starts it
- File access is limited to the current working directory and subdirectories
- No network access is required for basic functionality
- Generated summaries are stored in `.codebot/summaries/` directory

## Future Enhancements

Planned features include:

- **Additional Tools**: `test` (generate comprehensive tests), `debug-tests` (debug failing tests), and `refactor` (safe
  code refactoring)
- **Transport Mechanisms**: Support for HTTP, WebSocket, and other transport options
- **Authentication**: Authorization for remote deployments
- **Git Integration**: Integration with git operations and CI/CD workflows
- **Extended Resources**: Test results, refactoring reports, and other resource types
