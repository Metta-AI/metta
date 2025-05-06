#!/bin/bash -e

# Set up SSO session. There is a bug in awscli that prevents us from setting
# sso this way, so we have to do it manually.
# aws configure set sso-session.softmax-sso.sso_start_url https://softmaxx.awsapps.com/start/
# aws configure set sso-session.softmax-sso.sso_region us-east-1
# aws configure set sso-session.softmax-sso.sso_registration_scopes sso:account:access

# Check if the SSO session already exists
if ! grep -q "\[sso-session softmax-sso\]" ~/.aws/config; then
    echo "Adding SSO session configuration..."

    # Create a temporary file with the new config
    cat >> ~/.aws/config << EOF

[sso-session softmax-sso]
sso_start_url = https://softmaxx.awsapps.com/start/
sso_region = us-east-1
sso_registration_scopes = sso:account:access
EOF
    echo "SSO session added successfully."
else
    echo "SSO session already exists. in ~/.aws/config"
fi

# Set up root profile
aws configure set profile.softmax-root.region us-east-1
aws configure set profile.softmax-root.output json


# Set up softmax-db profile
aws configure set profile.softmax-db.sso_session softmax-sso
aws configure set profile.softmax-db.sso_account_id 767406518141
aws configure set profile.softmax-db.sso_role_name PowerUserAccess
aws configure set profile.softmax-db.region us-east-1

# Set up softmax-db-admin profile
aws configure set profile.softmax-db-admin.sso_session softmax-sso
aws configure set profile.softmax-db-admin.sso_account_id 767406518141
aws configure set profile.softmax-db-admin.sso_role_name AdministratorAccess
aws configure set profile.softmax-db-admin.region us-east-1

# Set up softmax profile
aws configure set profile.softmax.sso_session softmax-sso
aws configure set profile.softmax.sso_account_id 751442549699
aws configure set profile.softmax.sso_role_name PowerUserAccess
aws configure set profile.softmax.region us-east-1

# Set up softmax-admin profile
aws configure set profile.softmax-admin.sso_session softmax-sso
aws configure set profile.softmax-admin.sso_account_id 751442549699
aws configure set profile.softmax-admin.sso_role_name AdministratorAccess
aws configure set profile.softmax-admin.region us-east-1

echo "AWS profiles have been configured successfully."
echo "Login to AWS using: aws sso login --profile softmax-sso"
