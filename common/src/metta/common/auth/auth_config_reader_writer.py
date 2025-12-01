from cogames.auth import AuthConfigReaderWriter

observatory_auth_config = AuthConfigReaderWriter(token_file_name="config.yaml", token_storage_key="observatory_tokens")
