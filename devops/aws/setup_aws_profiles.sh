#!/bin/bash -e

# Parse command line arguments
RESET_CONFIG=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --reset)
      RESET_CONFIG=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--reset]"
      exit 1
      ;;
  esac
done

# Function to completely clear AWS configuration and cache
clear_aws_completely() {
  echo "Completely clearing AWS configuration and cache..."

  # Remove entire AWS directory
  echo "Removing entire ~/.aws directory..."
  rm -rf ~/.aws/

  # Recreate the directory
  mkdir -p ~/.aws

  echo "AWS configuration completely cleared and reset."
}

# Function to check if a valid SSO token exists
check_sso_token() {
  # Path to the SSO cache directory
  SSO_CACHE_DIR=~/.aws/sso/cache

  # Check if the directory exists
  if [ ! -d "$SSO_CACHE_DIR" ]; then
    return 1
  fi

  # Look for valid token files (non-empty JSON files with unexpired tokens)
  for token_file in "$SSO_CACHE_DIR"/*.json; do
    if [ -f "$token_file" ]; then
      # Check if the token contains our SSO URL
      if grep -q "softmaxx.awsapps.com" "$token_file" 2> /dev/null; then
        # Check if token is not expired
        expiration=$(grep -o '"expiresAt": "[^"]*"' "$token_file" | cut -d '"' -f 4)
        if [ -n "$expiration" ]; then
          # Convert expiration to timestamp
          expiry_timestamp=$(date -d "$expiration" +%s 2> /dev/null || date -j -f "%Y-%m-%dT%H:%M:%SZ" "$expiration" +%s 2> /dev/null)
          current_timestamp=$(date +%s)

          if [ $current_timestamp -lt $expiry_timestamp ]; then
            echo "Valid SSO token found in $token_file (expires at $expiration)"
            return 0
          fi
        fi
      fi
    fi
  done

  return 1
}

# Function to initialize AWS SSO configuration
initialize_aws_config() {
  echo "Initializing AWS configuration..."

  # Check if the SSO session already exists
  if ! grep -q "\[sso-session softmax-sso\]" ~/.aws/config 2> /dev/null; then
    echo "Adding SSO session configuration..."
    mkdir -p ~/.aws
    # Create a temporary file with the new config
    cat >> ~/.aws/config << EOF
[sso-session softmax-sso]
sso_start_url = https://softmaxx.awsapps.com/start/
sso_region = us-east-1
sso_registration_scopes = sso:account:access
EOF
    echo "SSO session added successfully."
  else
    echo "SSO session already exists in ~/.aws/config"
  fi

  # Set up root profile
  aws configure set profile.softmax-root.region us-east-1
  aws configure set profile.softmax-root.output json

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
  grep -q '^export AWS_PROFILE=' ~/.zshrc 2> /dev/null || echo -e '\nexport AWS_PROFILE=softmax' >> ~/.zshrc
}

# Check if we're in CI or Docker
if [ -f /.dockerenv ]; then
  export IS_DOCKER=true
else
  export IS_DOCKER=false
fi

# Main execution logic
if [ -z "$CI" ] && [ "$IS_DOCKER" = "false" ]; then

  if [ -f ~/.aws/config ]; then
    if grep -q '^\[profile softmax-db\]' ~/.aws/config; then
      echo "Removing softmax-db profile..."
      sed -i.bak '/^\[profile softmax-db\]/,/^\[/{/^\[profile softmax-db\]/d;/^\[/!d;}' ~/.aws/config
    fi

    if grep -q '^\[profile softmax-db-admin\]' ~/.aws/config; then
      echo "Removing softmax-db-admin profile..."
      sed -i.bak '/^\[profile softmax-db-admin\]/,/^\[/{/^\[profile softmax-db-admin\]/d;/^\[/!d;}' ~/.aws/config
    fi
  fi

  # Handle reset if requested
  if [ "$RESET_CONFIG" = true ]; then
    clear_aws_completely
    echo "Proceeding with fresh AWS SSO setup..."
    initialize_aws_config
  else
    # Check if we already have a valid token
    if check_sso_token; then
      echo "Valid SSO token already exists for softmax-sso"
      echo "Skip initialization as token is valid."
      echo "Use --reset if you need to completely reset your AWS configuration."
      exit 0
    else
      echo "No valid SSO token found, initializing AWS config..."
      initialize_aws_config
    fi
  fi

  echo "Running AWS SSO login..."
  aws sso login --profile softmax

  # Verify token was created successfully
  if check_sso_token; then
    echo "SSO login successful! Token created for softmax-sso."
  else
    echo "WARNING: SSO login completed but valid token not found. There might be an issue."
  fi
else
  # Always initialize in CI/Docker environments, but don't attempt login
  initialize_aws_config
  echo "Login to AWS using: aws sso login --profile softmax"
fi
