import cogames.auth

observatory_auth_config = cogames.auth.AuthConfigReaderWriter(
    token_file_name="config.yaml", token_storage_key="observatory_tokens"
)
