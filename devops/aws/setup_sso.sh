#!/bin/bash

# 1. Create AWS config
mkdir -p ~/.aws
cat > ~/.aws/config << 'EOL'
[default]
region = us-east-1
[profile stem]
sso_session = stem-sso
sso_account_id = 767406518141
sso_role_name = PowerUserAccess
region = us-east-1
[sso-session stem-sso]
sso_start_url = https://stemai.awsapps.com/start/
sso_region = us-east-1
sso_registration_scopes = sso:account:access
EOL

# 2. Log in to AWS SSO
echo "Logging in to AWS SSO..."
aws sso login --profile stem

# 3. Set AWS_PROFILE
export AWS_PROFILE=stem

# 4. Test the configuration
echo "Testing AWS access..."
aws s3 ls

# 5. Add profile to .bashrc if it's not already there
if ! grep -q "export AWS_PROFILE=stem" ~/.bashrc; then
    echo "export AWS_PROFILE=stem" >> ~/.bashrc
    echo "Added AWS_PROFILE to .bashrc"
else
    echo "AWS_PROFILE already in .bashrc"
fi

echo "Setup complete! Please restart your terminal or run 'source ~/.bashrc' to apply changes."
