"""CLI login functionality for CoGames."""

from cogames.auth import BaseCLIAuthenticator

# Default CoGames server URL
DEFAULT_COGAMES_SERVER = "https://beta.softmax.com/api"


class CoGamesAuthenticator(BaseCLIAuthenticator):
    """CLI Authenticator for CoGames, storing tokens in cogames.yaml under 'login_tokens' key."""

    def __init__(self, auth_server_url: str):
        super().__init__(
            auth_server_url=auth_server_url,
            token_file_name="cogames.yaml",
            token_storage_key="login_tokens",  # Nested under 'login_tokens' key
            extra_uris={},  # No extra URIs for CoGames
        )


def perform_login(auth_server_url: str, force: bool = False, timeout: int = 300) -> bool:
    """Perform CoGames login authentication.

    Args:
        auth_server_url: URL of the CoGames authentication server
        force: If True, get a new token even if one already exists
        timeout: Authentication timeout in seconds

    Returns:
        True if authentication successful, False otherwise
    """
    authenticator = CoGamesAuthenticator(auth_server_url=auth_server_url)

    # Check if we already have a token
    if authenticator.has_saved_token() and not force:
        return True

    # Perform authentication
    return authenticator.authenticate(timeout=timeout)
