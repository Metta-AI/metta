#!/bin/bash

# Parse command line arguments
USE_ROOT=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --root) USE_ROOT=true; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Create AWS directory if it doesn't exist
mkdir -p ~/.aws

if [ "$USE_ROOT" = true ]; then
    # Setup for root account
    echo "Setting up AWS configuration for root account..."

    # Prompt for AWS access key and secret
    echo "Please enter your AWS root account credentials:"
    read -p "AWS Access Key ID: " AWS_ACCESS_KEY_ID
    read -sp "AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
    echo

    # Create AWS credentials file
    cat > ~/.aws/credentials << EOL
[default]
aws_access_key_id = $AWS_ACCESS_KEY_ID
aws_secret_access_key = $AWS_SECRET_ACCESS_KEY
region = us-east-1
EOL

    # Create AWS config file
    cat > ~/.aws/config << 'EOL'
[default]
region = us-east-1
output = json
EOL

    # Test the configuration
    echo "Testing AWS access with root credentials..."
    aws s3 ls

    # Unset AWS_PROFILE if it's set
    if [ -n "$AWS_PROFILE" ]; then
        echo "Unsetting AWS_PROFILE environment variable"
        unset AWS_PROFILE
    fi

    # Remove AWS_PROFILE from shell config files
    for rc_file in ~/.bashrc ~/.zshrc; do
        if [ -f "$rc_file" ]; then
            if grep -q "export AWS_PROFILE=" "$rc_file"; then
                sed -i.bak '/export AWS_PROFILE=/d' "$rc_file"
                echo "Removed AWS_PROFILE from $rc_file"
            fi
        fi
    done

    echo "Root account setup complete!"

    # Create a helper script to open the AWS console
    cat > ~/.aws/console.sh << 'EOL'
#!/bin/bash
# Helper script to open the AWS console in the browser
aws_signin_url="https://signin.aws.amazon.com/console"
open "$aws_signin_url" 2>/dev/null || xdg-open "$aws_signin_url" 2>/dev/null || echo "Could not open browser. Please visit $aws_signin_url manually."
EOL
    chmod +x ~/.aws/console.sh

    # Add an alias for the console command
    for rc_file in ~/.bashrc ~/.zshrc; do
        if [ -f "$rc_file" ]; then
            if ! grep -q "alias aws-console=" "$rc_file"; then
                echo "alias aws-console='~/.aws/console.sh'" >> "$rc_file"
                echo "Added aws-console alias to $rc_file"
            fi
        fi
    done

    echo "You can open the AWS Console in your browser by typing 'aws-console'"
    echo "IMPORTANT: To apply changes immediately, run: source ~/.bashrc (or source ~/.zshrc)"
else
    # Original SSO setup
    # 1. Create AWS config with both default and stem profiles
    cat > ~/.aws/config << 'EOL'
[default]
region = us-east-1
output = json

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

    # 3. Verify the profile works
    echo "Testing AWS access with stem profile..."
    if ! aws s3 ls --profile stem; then
        echo "Error: Could not access AWS with the stem profile."
        echo "Please check your SSO configuration and try again."
        exit 1
    fi

    # 4. Set AWS_PROFILE in the current shell
    export AWS_PROFILE=stem
    echo "Set AWS_PROFILE=stem in current shell"

    # 5. Add profile to .bashrc if it's not already there
    if ! grep -q "export AWS_PROFILE=stem" ~/.bashrc; then
        echo "export AWS_PROFILE=stem" >> ~/.bashrc
        echo "Added AWS_PROFILE to .bashrc"
    else
        echo "AWS_PROFILE already in .bashrc"
    fi

    # 6. Add profile to .zshrc if it's not already there
    if ! grep -q "export AWS_PROFILE=stem" ~/.zshrc; then
        echo "export AWS_PROFILE=stem" >> ~/.zshrc
        echo "Added AWS_PROFILE to .zshrc"
    else
        echo "AWS_PROFILE already in .zshrc"
    fi

    echo "SSO setup complete!"
    echo "IMPORTANT: To apply changes immediately, run: source ~/.bashrc (or source ~/.zshrc)"
fi
