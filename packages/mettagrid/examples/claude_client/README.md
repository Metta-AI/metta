# Claude-Powered MettaGrid RPC Client

This example demonstrates using Claude (Anthropic's AI assistant) to play MettaGrid via the RPC server.

## Architecture

```
┌─────────────┐
│   Claude    │
│     API     │
└──────┬──────┘
       │ (action decisions)
       │
┌──────▼──────────┐
│  Python Client  │
│  (this code)    │
└──────┬──────────┘
       │ (protobuf over TCP)
       │
┌──────▼──────────┐
│  RPC Server     │
│  (C++/Bazel)    │
└──────┬──────────┘
       │
┌──────▼──────────┐
│ MettaGridEngine │
│   (C++ Core)    │
└─────────────────┘
```

## Prerequisites

1. **RPC Server**: Built RPC server binary
   ```bash
   bazel build //cpp:mettagrid_rpc_server
   ```

2. **Anthropic API Key**: You have two options:

   **Option A: Environment Variable**
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-your-key-here"
   ```

   **Option B: .env File** (Recommended for development)
   ```bash
   # Copy the example file
   cp .env.example .env

   # Edit .env and add your API key
   # ANTHROPIC_API_KEY=sk-ant-your-key-here
   ```

   Get your API key from: https://console.anthropic.com/

3. **Python Dependencies**:
   ```bash
   pip install anthropic protobuf
   ```

## Usage

### 1. Start the RPC Server

In one terminal:
```bash
cd packages/mettagrid
bazel run //cpp:mettagrid_rpc_server
```

The server will start on `127.0.0.1:5858` by default.

### 2. Run the Claude Client

In another terminal:
```bash
cd packages/mettagrid
python examples/claude_client/main.py
```

### Options

```bash
# Run multiple episodes
python examples/claude_client/main.py --episodes 5

# Use different Claude model
python examples/claude_client/main.py --model claude-3-opus-20240229

# Connect to different server
python examples/claude_client/main.py --host 192.168.1.100 --port 9000

# Quiet mode (less verbose)
python examples/claude_client/main.py --quiet
```

## How It Works

1. **Game Setup**: The client creates a simple 10x10 grid with walls around the border and an agent in the center.

2. **Game Loop**:
   - Get current game state from RPC server
   - Send state to Claude API for action decision
   - Parse Claude's response to get action index
   - Send action to RPC server
   - Repeat until episode ends

3. **Actions**: The agent can take 7 actions:
   - 0: noop (do nothing)
   - 1: move forward
   - 2: move backward
   - 3: move left
   - 4: move right
   - 5: rotate left
   - 6: rotate right

## Files

- `rpc_client.py` - Low-level RPC communication (length-prefixed protobuf)
- `game_config.py` - Simple game configuration builder
- `claude_player.py` - Claude API integration for action selection
- `main.py` - Main game loop and CLI
- `test_rpc_connection.py` - RPC connection test (no Claude API required)

## Example Output

```
Claude-powered MettaGrid RPC Client
============================================================
Server: 127.0.0.1:5858
Model: claude-3-5-sonnet-20241022
Episodes: 1
============================================================

Initialized Claude player with model: claude-3-5-sonnet-20241022
Connected to RPC server at 127.0.0.1:5858

************************************************************
Episode 1/1
************************************************************

============================================================
Starting episode with 1 agent(s)
Max steps: 100
============================================================

Step   1/100 | Action: 1 | Reward: -0.010 | Total: -0.010
Step   2/100 | Action: 5 | Reward: +0.000 | Total: -0.010
Step   3/100 | Action: 1 | Reward: +0.000 | Total: -0.010
...

============================================================
Episode complete!
Total steps: 100
Total reward: -0.450
============================================================

Game 'claude_game_0' deleted successfully
Disconnected from RPC server
```

## Testing

You can test the RPC connection without requiring a Claude API key:

```bash
# Start the RPC server first
bazel run //cpp:mettagrid_rpc_server

# In another terminal, run the test script
cd packages/mettagrid/examples/claude_client
uv run --with protobuf --with numpy test_rpc_connection.py
```

This will verify that:
- The RPC client can connect to the server
- Games can be created and deleted
- State can be queried and updated

Expected output:
```
Testing RPC connection...
Connected to RPC server at 127.0.0.1:5858

Creating game 'test_game'...
Game 'test_game' created successfully
Fetching initial state...
  - Observations: 150 bytes
  - Rewards: 4 bytes
  - Terminals: 1 bytes

Stepping game with noop action (0)...
  - Observations: 150 bytes
  - Rewards: 4 bytes

Deleting game 'test_game'...
Game 'test_game' deleted successfully

✓ All RPC operations successful!
Disconnected from RPC server
```

## Notes

- The current implementation uses a very simple game with minimal state information
- Claude currently makes decisions based on limited context (just step number and last reward)
- Future enhancements could include:
  - Sending visual observations as images
  - Providing more detailed state descriptions
  - Implementing memory/history
  - Multi-agent coordination
