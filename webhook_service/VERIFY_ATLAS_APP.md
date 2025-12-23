# Verification Prompt for Codex

## Context

Nishad suggested using `asana/atlas_app` from AWS Secrets Manager for service authentication instead of a Personal
Access Token (PAT). He said:

> "can do get_secretsmanager_secret("asana/atlas_app", require_exists=False) gives you a client_id and client_secret. i
> think this is the preferred way to connect for a service instead of a token"

He also mentioned the roster project ID: `1209948553419016`

## Verification Tasks

Please verify the following:

### 1. Check atlas_app Secret Structure

**Task:** Verify what's in the `asana/atlas_app` secret in AWS Secrets Manager.

**How to check:**

- Use `get_secretsmanager_secret("asana/atlas_app", require_exists=False)` (from `softmax.aws.secrets_manager`)
- Or use boto3 directly: `boto3.client('secretsmanager').get_secret_value(SecretId='asana/atlas_app')`
- Parse the JSON and list all keys

**Expected findings:**

- Should have `client_id` and `client_secret` fields
- May or may not have a `token` field

### 2. Test OAuth client_credentials Grant

**Task:** Test if Asana's OAuth endpoint supports `client_credentials` grant type using the atlas_app credentials.

**How to test:**

```python
import httpx
import json

# Get credentials from atlas_app secret
client_id = "..."
client_secret = "..."

# Try OAuth client_credentials flow
response = httpx.post(
    "https://app.asana.com/-/oauth_token",
    data={
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }
)
print(f"Status: {response.status_code}")
print(f"Body: {response.text}")
```

**Expected result:**

- Should get a 400 error with "unsupported_grant_type"
- Asana only supports `authorization_code` and `refresh_token` grant types

### 3. Verify Roster Project ID

**Task:** Check if the roster project ID in our config matches what Nishad said.

**How to check:**

- Look in `webhook_service/src/github_webhook/config.py`
- Find `ASANA_ROSTER_PROJECT_GID` default value
- Compare with Nishad's value: `1209948553419016`

**Expected result:**

- Should match: `1209948553419016`

### 4. Check Existing Code Usage

**Task:** See how `asana_bugs.py` uses atlas_app credentials.

**How to check:**

- Read `devops/stable/asana_bugs.py`
- Look at `_get_asana_credentials()` function
- See what it does with atlas_app secret

**Expected findings:**

- It tries to get a token from the secret first
- If no token, tries OAuth client_credentials (which fails)
- Falls back to environment variables

## Questions to Answer

1. **Does atlas_app secret have client_id and client_secret?** (Yes/No)
2. **Does OAuth client_credentials grant work?** (Yes/No - should be No)
3. **What error does Asana return for client_credentials?** (Exact error message)
4. **Does roster project ID match?** (Yes/No)
5. **Is there a different way to use atlas_app for service auth?** (Check Asana docs or existing code)

## Current Status

- ✅ Code loads atlas_app credentials from AWS
- ❌ OAuth client_credentials doesn't work (Asana limitation)
- ⚠️ Falls back to PAT (which is from deactivated account)
- ❓ Need clarification from Nishad on how to use atlas_app

## Files to Check

- `webhook_service/src/github_webhook/config.py` - Configuration loading
- `webhook_service/src/github_webhook/asana_integration.py` - Authentication logic
- `devops/stable/asana_bugs.py` - Example of atlas_app usage
- `softmax/src/softmax/aws/secrets_manager.py` - Secret retrieval function

## Expected Conclusion

After verification, you should confirm:

1. atlas_app has client_id/client_secret (but no token)
2. OAuth client_credentials is NOT supported by Asana
3. Roster project ID is correct
4. We need to ask Nishad: "How do we use atlas_app for service authentication if client_credentials doesn't work?"
