#!/bin/sh
# PATH modification utilities adapted from uv installer (MIT licensed)
# https://github.com/astral-sh/uv
#
# These functions handle cross-platform PATH configuration for shell scripts

# Function to get HOME directory with fallback
get_home() {
    if [ -n "${HOME:-}" ]; then
        echo "$HOME"
    elif [ -n "${USER:-}" ]; then
        getent passwd "$USER" 2>/dev/null | cut -d: -f6 || echo "/home/$USER"
    else
        getent passwd "$(id -un)" 2>/dev/null | cut -d: -f6 || echo "/home/$(id -un)"
    fi
}

# Function to replace absolute paths with $HOME variable for portability
replace_home_with_var() {
    local _path="$1"
    local _home="${HOME:-$(get_home)}"
    
    if [ -n "$_home" ]; then
        echo "$_path" | sed "s,^$_home,\$HOME,"
    else
        echo "$_path"
    fi
}

# Function to detect user's shell
detect_shell() {
    local detected_shell=""
    
    # First, check the current shell
    if [ -n "$BASH_VERSION" ]; then
        detected_shell="bash"
    elif [ -n "$ZSH_VERSION" ]; then
        detected_shell="zsh"
    else
        # Fallback to checking $SHELL
        case "$SHELL" in
            */bash) detected_shell="bash" ;;
            */zsh) detected_shell="zsh" ;;
            */fish) detected_shell="fish" ;;
            */sh) detected_shell="sh" ;;
            *) detected_shell="sh" ;;  # Default fallback
        esac
    fi
    
    echo "$detected_shell"
}

# Write environment script that safely adds to PATH
write_env_script_sh() {
    local _install_dir="$1"
    local _env_script_path="$2"
    
    cat > "$_env_script_path" << EOF
#!/bin/sh
# add binaries to PATH if they aren't added yet
# affix colons on either side of \$PATH to simplify matching
case ":\${PATH}:" in
    *:"$_install_dir":*)
        ;;
    *)
        # Prepending path in case a system-installed binary needs to be overridden
        export PATH="$_install_dir:\$PATH"
        ;;
esac
EOF
    chmod +x "$_env_script_path"
}

# Write fish environment script
write_env_script_fish() {
    local _install_dir="$1"
    local _env_script_path="$2"
    
    cat > "$_env_script_path" << EOF
if not contains "$_install_dir" \$PATH
    # Prepending path in case a system-installed binary needs to be overridden
    set -x PATH "$_install_dir" \$PATH
end
EOF
    chmod +x "$_env_script_path"
}

# Add source line to shell profile if not already present
add_to_profile() {
    local _profile="$1"
    local _line="$2"
    
    if [ -f "$_profile" ]; then
        if ! grep -F "$_line" "$_profile" > /dev/null 2>&1; then
            echo >> "$_profile"
            echo "$_line" >> "$_profile"
            return 0
        fi
    fi
    return 1
}

# Get the appropriate shell config file for the current shell
get_shell_config() {
    local shell_type="$1"
    local config_file=""
    
    case "$shell_type" in
        bash)
            # On macOS, .bash_profile is preferred for login shells
            # On Linux, .bashrc is typically used
            if [ "$(uname -s)" = "Darwin" ]; then
                if [ -f "$HOME/.bash_profile" ]; then
                    config_file="$HOME/.bash_profile"
                elif [ -f "$HOME/.profile" ]; then
                    config_file="$HOME/.profile"
                else
                    config_file="$HOME/.bash_profile"
                fi
            else
                if [ -f "$HOME/.bashrc" ]; then
                    config_file="$HOME/.bashrc"
                elif [ -f "$HOME/.profile" ]; then
                    config_file="$HOME/.profile"
                else
                    config_file="$HOME/.bashrc"
                fi
            fi
            ;;
        zsh)
            # Check for ZDOTDIR
            local zdotdir="${ZDOTDIR:-$HOME}"
            config_file="$zdotdir/.zshrc"
            ;;
        fish)
            config_file="$HOME/.config/fish/conf.d/metta.fish"
            ;;
        sh|*)
            config_file="$HOME/.profile"
            ;;
    esac
    
    echo "$config_file"
}

# Apply PATH modifications to all relevant shell config files
# Uses a "shotgun" approach for bash to ensure coverage
apply_path_modifications() {
    local _bin_dir="$1"
    local _env_script="$2"
    local _fish_env_script="$3"
    local _shell_type="$4"
    
    local _env_script_expr=$(replace_home_with_var "$_env_script")
    local _fish_env_script_expr=$(replace_home_with_var "$_fish_env_script")
    local _modified=0
    
    case "$_shell_type" in
        bash|sh)
            # Try multiple bash config files
            for _profile in "$HOME/.profile" "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.bash_login"; do
                if [ -f "$_profile" ]; then
                    if add_to_profile "$_profile" ". \"$_env_script_expr\""; then
                        echo "Updated $_profile"
                        _modified=1
                    fi
                fi
            done
            
            # If none exist, create .profile
            if [ "$_modified" = "0" ]; then
                if add_to_profile "$HOME/.profile" ". \"$_env_script_expr\""; then
                    echo "Created $HOME/.profile"
                    _modified=1
                fi
            fi
            ;;
            
        zsh)
            local _zdotdir="${ZDOTDIR:-$HOME}"
            
            if add_to_profile "$_zdotdir/.zshrc" ". \"$_env_script_expr\""; then
                echo "Updated $_zdotdir/.zshrc"
                _modified=1
            fi
            
            # On macOS, also check .zprofile
            if [ "$(uname -s)" = "Darwin" ] && [ -f "$_zdotdir/.zprofile" ]; then
                if add_to_profile "$_zdotdir/.zprofile" ". \"$_env_script_expr\""; then
                    echo "Updated $_zdotdir/.zprofile"
                    _modified=1
                fi
            fi
            ;;
            
        fish)
            # Create fish config directory
            mkdir -p "$HOME/.config/fish/conf.d"
            
            # Create a dedicated config file for metta
            local _fish_config="$HOME/.config/fish/conf.d/metta.fish"
            echo "source \"$_fish_env_script_expr\"" > "$_fish_config"
            echo "Created $_fish_config"
            _modified=1
            ;;
    esac
    
    return $([ "$_modified" = "1" ] && echo 0 || echo 1)
}

# Check if binary is shadowed by another command in PATH
check_shadowed_binary() {
    local _bin_name="$1"
    local _expected_path="$2"
    
    if command -v "$_bin_name" > /dev/null 2>&1; then
        local _actual_path=$(command -v "$_bin_name")
        if [ "$_actual_path" != "$_expected_path" ]; then
            echo "Warning: '$_bin_name' command found at $_actual_path"
            echo "The newly installed version at $_expected_path may be shadowed."
            return 1
        fi
    fi
    return 0
}

# Add to GITHUB_PATH if in CI
add_to_ci_path() {
    local _install_dir="$1"
    
    if [ -n "${GITHUB_PATH:-}" ]; then
        echo "$_install_dir" >> "$GITHUB_PATH"
        echo "Added $_install_dir to GITHUB_PATH for CI"
        return 0
    fi
    return 1
}