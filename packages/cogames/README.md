CoGames is a collection of environments and scenarios for multi-agent cooperative and competitive games.

## Installation

```bash
uv pip install cogames
```

## Usage

```bash
# Help
cogames --help

# List games and scenarios
cogames games                      # List all available games
cogames games [game]               # Describe a game and list its scenarios
cogames scenario [game] [scenario] # Describe a specific scenario

# Play a game
cogames play [game] [scenario]
cogames make-scenario [game]

# Train a policy
cogames train [game] [scenario]

# Evaluate a policy
cogames evaluate [game] [scenario] [policy]
```
