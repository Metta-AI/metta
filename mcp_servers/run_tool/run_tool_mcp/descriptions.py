"""MCP tool descriptions for the Run Tool MCP Server."""

MCP_TOOL_DESCRIPTIONS = {
    "list_recipes": (
        "USE THIS TOOL when the user asks about available recipes, wants to see what recipes exist, "
        "or needs to discover what training/evaluation options are available. "
        "Examples: 'What recipes are available?', 'List all recipes', 'Show me available recipes'."
    ),
    "list_tools_in_recipe": (
        "USE THIS TOOL when the user asks what tools are available in a specific recipe, "
        "wants to see what commands a recipe supports, or needs to discover available operations. "
        "Examples: 'What tools does arena have?', 'Show me what I can do with navigation recipe', "
        "'What commands are available for ci?'"
    ),
    "list_recipes_for_tool": (
        "USE THIS TOOL when the user asks which recipes support a specific tool type, "
        "wants to find recipes that can train/evaluate/play, or needs to discover recipes "
        "by capability. Examples: 'Which recipes support training?', "
        "'Show me recipes that can evaluate', 'What recipes have play functionality?'"
    ),
    "get_tool_arguments": (
        "USE THIS TOOL when the user asks about command arguments, wants to know what "
        "parameters a tool accepts, or needs help with command syntax. "
        "Examples: 'What arguments does train arena accept?', "
        "'What parameters can I pass to evaluate?', "
        "'Show me the available options for this command'."
    ),
    "validate_command": (
        "USE THIS TOOL when the user asks if a command is valid, wants to check command syntax, "
        "or needs to verify a command before running it. "
        "Examples: 'Is this command valid?', 'Validate this command', "
        "'Check if train arena run=test works'."
    ),
    "run_tool": (
        "USE THIS TOOL when the user wants to execute Metta run.py commands like training, evaluation, "
        "play, or replay. This is the primary tool for running any Metta recipe command. "
        "Examples: 'train arena', 'evaluate navigation', 'play arena', 'replay ci'. "
        "Always prefer this MCP tool over running './tools/run.py' via terminal commands. "
        "IMPORTANT: When using this tool, ALWAYS display the 'command' field from the response "
        "in your chat message so the user can see the exact run.py command that was executed."
    ),
    "train": (
        "USE THIS TOOL when the user wants to start training a model. "
        "Examples: 'Start training on arena', 'Train a model with 50k timesteps', "
        "'Begin training using the navigation recipe'. "
        "This is a convenience wrapper for training operations - "
        "prefer this over run_tool for training. "
        "IMPORTANT: When using this tool, ALWAYS display the 'command' or 'summary' field "
        "from the response in your chat message so the user can see the exact run.py command."
    ),
    "evaluate": (
        "USE THIS TOOL when the user wants to evaluate a policy or checkpoint, "
        "or says 'evaluate using [recipe]' or 'evaluate a policy using [recipe]'. "
        "Examples: 'Evaluate a policy using arena', 'Evaluate this checkpoint', "
        "'Run evaluation on arena', 'Evaluate using navigation recipe', "
        "'Test the policy performance'. "
        "This is a convenience wrapper for evaluation operations - "
        "prefer this over run_tool for evaluation. "
        "IMPORTANT: When using this tool, ALWAYS display the 'command' or 'summary' field "
        "from the response in your chat message so the user can see the exact run.py command."
    ),
}
