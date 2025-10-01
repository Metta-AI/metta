# ClaudePolicy - AI Agent using Anthropic's Claude

ClaudePolicy enables Claude AI to play CoGames by integrating with the Anthropic API.

## Installation

The `anthropic` package is included in the cogames dependencies. If you need to install it separately:

```bash
uv add anthropic
# or
pip install anthropic
```

## Usage

### Option 1: Using Environment Variable

Set your Anthropic API key as an environment variable:

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

Then use ClaudePolicy in your code:

```python
from cogames.policy import ClaudePolicy
from mettagrid import MettaGridEnv

env = MettaGridEnv(env_cfg=your_config)
policy = ClaudePolicy(env)
```

### Option 2: Using Configuration File

Create a `claude_settings.yaml` file (see `claude_settings.yaml.example` for template):

```yaml
api_key: "sk-ant-api03-..."
prompt: "Custom system prompt for Claude"
model: "claude-sonnet-4-5-20250929"
```

Then load it with `--policy-data`:

```bash
cogames play --game your_game --policy cogames.policy.ClaudePolicy --policy-data claude_settings.yaml
```

### Option 3: Direct API Key

Pass the API key directly when creating the policy:

```python
policy = ClaudePolicy(env, api_key="sk-ant-api03-...")
```

## Configuration Options

- **api_key**: Your Anthropic API key (required if not in environment)
- **prompt**: Custom system prompt to guide Claude's behavior (optional)
- **model**: Claude model to use (default: `claude-sonnet-4-5-20250929`)

Available models:
- `claude-sonnet-4-5-20250929` (recommended, good balance of speed and quality)
- `claude-opus-4-5-20250929` (highest quality, slower)
- `claude-3-5-sonnet-20241022` (previous generation)

## Command Line Usage

### Play with Claude Agent

```bash
# Using environment variable
export ANTHROPIC_API_KEY="sk-ant-api03-..."
cogames play --game cogs_vs_clips_2x2 --policy cogames.policy.ClaudePolicy

# Using config file
cogames play --game cogs_vs_clips_2x2 --policy cogames.policy.ClaudePolicy --policy-data claude_settings.yaml
```

## How It Works

1. **Observation Processing**: Claude receives formatted observations including:
   - Observation shape and statistics
   - Available actions
   - Available resources
   - Current game state

2. **Action Generation**: Claude generates actions in the format `action: X, argument: Y`
   - If parsing fails, a random action is used as fallback

3. **Error Handling**:
   - API errors fall back to random actions
   - Invalid actions are automatically constrained to valid action space

## Customizing Claude's Behavior

Edit the `prompt` field in your configuration to customize how Claude plays:

```yaml
prompt: |
  You are a strategic AI agent in a resource gathering game.

  Priorities:
  1. Gather wood efficiently
  2. Cooperate with other agents
  3. Avoid conflicts over resources

  When responding, provide an action in the format: action: X, argument: Y
```

## Performance Considerations

- **API Latency**: Each action requires an API call (~100-500ms)
- **Cost**: Each action costs tokens (typically $0.003 per 1000 tokens)
- **Rate Limits**: Subject to Anthropic's rate limits

For high-speed gameplay, consider:
- Using faster models like Sonnet
- Reducing the max_tokens parameter
- Caching common observations

## Troubleshooting

**"Anthropic API key not provided"**
- Set `ANTHROPIC_API_KEY` environment variable
- Or pass `api_key` parameter
- Or use `--policy-data` with a config file

**"anthropic package not found"**
```bash
uv add anthropic
```

**Actions not working correctly**
- Check that Claude's responses are in the format: `action: X, argument: Y`
- Review the prompt to ensure it guides Claude to respond correctly
- Check the logs for parsing errors

## Example

```python
import os
from cogames.policy import ClaudePolicy
from mettagrid import MettaGridEnv
from cogames.cogs_vs_clips.scenarios import games

# Get a game config
game_config = games()["cogs_vs_clips_2x2"]

# Create environment
env = MettaGridEnv(env_cfg=game_config)

# Create Claude policy
policy = ClaudePolicy(
    env,
    api_key=os.environ["ANTHROPIC_API_KEY"],
    prompt="You are a cooperative AI agent. Focus on efficiency and teamwork."
)

# Create per-agent policies
agent_policies = [policy.agent_policy(i) for i in range(env.num_agents)]

# Run episode
obs, _ = env.reset()
for agent_id, agent_policy in enumerate(agent_policies):
    agent_policy.reset()

for step in range(100):
    actions = [agent_policies[i].step(obs[i]) for i in range(env.num_agents)]
    obs, rewards, dones, truncated, info = env.step(actions)

    if all(dones) or all(truncated):
        break
```

## License

Same as the parent cogames package.
