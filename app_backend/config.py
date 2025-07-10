import os

stats_db_uri = os.getenv("STATS_DB_URI", "postgres://postgres:password@127.0.0.1/postgres")
debug_user_email = os.getenv("DEBUG_USER_EMAIL")
debug = os.getenv("DEBUG", "").lower() in ["true", "1", "yes"]

host = os.getenv("HOST", "127.0.0.1")
port = int(os.getenv("PORT", "8000"))

# LLM API configuration
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# Optional API URL overrides
openai_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1/chat/completions")
openrouter_api_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1/chat/completions")
anthropic_api_url = os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages")

# Optional model name overrides
openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
openrouter_model = os.getenv("OPENROUTER_MODEL", "openrouter/auto")

# Git client configuration
git_client_mode = os.getenv("GIT_CLIENT_MODE", "auto")
