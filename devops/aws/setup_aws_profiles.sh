#!/bin/bash -e

# Function to completely clear AWS configuration and cache
clear_aws_completely() {
  echo "Completely clearing AWS configuration and cache..."
  
  # Create backup if config exists
  if [ -f ~/.aws/config ]; then
    echo "Creating backup of existing AWS config..."
    cp ~/.aws/config ~/.aws/config.backup.$(date +%Y%m%d_%H%M%S)
  fi
  
  if [ -f ~/.aws/credentials ]; then
    echo "Creating backup of existing AWS credentials..."
    cp ~/.aws/credentials ~/.aws/credentials.backup.$(date +%Y%m%d_%H%M%S)
  fi
  
  # Remove entire AWS directory
  echo "Removing entire ~/.aws directory..."
  rm -rf ~/.aws/
  
  # Recreate the directory
  mkdir -p ~/.aws
  
  echo "AWS configuration completely cleared and reset."
}

# Function to check and update AWS_PROFILE in shell configuration
check_and_update_aws_profile() {
  local shell_config=""
  
  # Determine which shell config file to use
  if [ -n "$ZSH_VERSION" ]; then
    shell_config="$HOME/.zshrc"
  elif [ -n "$BASH_VERSION" ]; then
    shell_config="$HOME/.bashrc"
  else
    # Default to .bashrc if we can't determine
    shell_config="$HOME/.bashrc"
  fi
  
  echo "Checking AWS_PROFILE configuration in $shell_config..."
  
  # Check if AWS_PROFILE is already set correctly
  if grep -q "^export AWS_PROFILE=softmax" "$shell_config" 2>/dev/null; then
    echo "✓ AWS_PROFILE=softmax is already correctly set in $shell_config"
  elif grep -q "^export AWS_PROFILE=" "$shell_config" 2>/dev/null; then
    # AWS_PROFILE exists but with different value
    current_value=$(grep "^export AWS_PROFILE=" "$shell_config" | cut -d'=' -f2)
    echo "⚠ AWS_PROFILE is set to $current_value in $shell_config"
    echo "  Updating to AWS_PROFILE=softmax..."
    
    # Use sed to update the existing line
    if [[ "$OSTYPE" == "darwin"* ]]; then
      # macOS sed
      sed -i '' 's/^export AWS_PROFILE=.*/export AWS_PROFILE=softmax/' "$shell_config"
    else
      # Linux sed
      sed -i 's/^export AWS_PROFILE=.*/export AWS_PROFILE=softmax/' "$shell_config"
    fi
    echo "✓ Updated AWS_PROFILE to softmax in $shell_config"
  else
    # AWS_PROFILE doesn't exist, add it
    echo "Adding AWS_PROFILE=softmax to $shell_config..."
    echo -e '\n# AWS Profile for softmax SSO' >> "$shell_config"
    echo 'export AWS_PROFILE=softmax' >> "$shell_config"
    echo "✓ Added AWS_PROFILE=softmax to $shell_config"
  fi
  
  # Check current environment and update for this session
  if [ "$AWS_PROFILE" = "softmax" ]; then
    echo "✓ AWS_PROFILE is correctly set to 'softmax' in current session"
  else
    echo "⚠ Current AWS_PROFILE is set to '${AWS_PROFILE:-<unset>}'"
    echo "  Setting AWS_PROFILE=softmax for current session..."
    export AWS_PROFILE=softmax
    echo "✓ AWS_PROFILE set to softmax for current session"
  fi
  
  echo "  Note: Changes to $shell_config will apply to new terminal sessions"
  echo "  or you can run 'source $shell_config' to apply now"
}

# Function to check and update AWS_PROFILE in shell configuration
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

# Function to initialize AWS SSO configuration from scratch
initialize_aws_config() {
  echo "Setting up fresh AWS configuration..."

  # Create the complete config file from scratch
  cat > ~/.aws/config << EOF
[sso-session softmax-sso]
sso_start_url = https://softmaxx.awsapps.com/start/
sso_region = us-east-1
sso_registration_scopes = sso:account:access

[profile softmax-root]
region = us-east-1
output = json

[profile softmax-db]
sso_session = softmax-sso
sso_account_id = 767406518141
sso_role_name = PowerUserAccess
region = us-east-1

[profile softmax-db-admin]
sso_session = softmax-sso
sso_account_id = 767406518141
sso_role_name = AdministratorAccess
region = us-east-1

[profile softmax]
sso_session = softmax-sso
sso_account_id = 751442549699
sso_role_name = PowerUserAccess
region = us-east-1

[profile softmax-admin]
sso_session = softmax-sso
sso_account_id = 751442549699
sso_role_name = AdministratorAccess
region = us-east-1
EOF

  echo "AWS configuration file created successfully."
  
  # Check and update AWS_PROFILE in shell config
  check_and_update_aws_profile
}

# Always clear cache to ensure fresh authentication

# Check if we're in CI or Docker
if [ -f /.dockerenv ]; then
  export IS_DOCKER=true
else
  export IS_DOCKER=false
fi

# Main execution logic
if [ -z "$CI" ] && [ "$IS_DOCKER" = "false" ]; then
  
  # Clear cache if requested
  if [ "$FORCE_CLEAR" = true ]; then
    clear_aws_cache
  fi
  
  # Always completely clear AWS config and start fresh
  clear_aws_completely
  
  # Initialize AWS config and proceed with fresh login
  echo "Setting up completely fresh AWS SSO configuration..."
  
  initialize_aws_config

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