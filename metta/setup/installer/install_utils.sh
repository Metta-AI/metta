#!/bin/sh
# shellcheck shell=dash
# shellcheck disable=SC2039  # local is non-POSIX
#
# PATH modification utilities adapted from uv installer (MIT licensed)
# https://github.com/astral-sh/uv
#
# These functions handle cross-platform PATH configuration for shell scripts

# Some versions of ksh have no `local` keyword. Alias it to `typeset`
has_local() {
    # shellcheck disable=SC2034  # deliberately unused
    local _has_local
}

has_local 2>/dev/null || alias local=typeset

set -u

# ========== Error handling and output functions ==========

say() {
    if [ "${PRINT_QUIET:-0}" = "0" ]; then
        echo "$1"
    fi
}

say_verbose() {
    if [ "${PRINT_VERBOSE:-0}" = "1" ]; then
        echo "$1"
    fi
}

warn() {
    if [ "${PRINT_QUIET:-0}" = "0" ]; then
        local red
        local reset
        red=$(tput setaf 1 2>/dev/null || echo '')
        reset=$(tput sgr0 2>/dev/null || echo '')
        say "${red}WARN${reset}: $1" >&2
    fi
}

err() {
    if [ "${PRINT_QUIET:-0}" = "0" ]; then
        local red
        local reset
        red=$(tput setaf 1 2>/dev/null || echo '')
        reset=$(tput sgr0 2>/dev/null || echo '')
        say "${red}ERROR${reset}: $1" >&2
    fi
    exit 1
}

need_cmd() {
    if ! check_cmd "$1"
    then err "need '$1' (command not found)"
    fi
}

check_cmd() {
    command -v "$1" > /dev/null 2>&1
    return $?
}

# Run a command that should never fail
ensure() {
    if ! "$@"; then err "command failed: $*"; fi
}

# This is just for indicating that commands' results are being intentionally ignored
ignore() {
    "$@"
}

# ========== HOME directory handling ==========

# Some Linux distributions don't set HOME
# https://github.com/astral-sh/uv/issues/6965
get_home() {
    if [ -n "${HOME:-}" ]; then
        echo "$HOME"
    elif [ -n "${USER:-}" ]; then
        getent passwd "$USER" 2>/dev/null | cut -d: -f6
    else
        getent passwd "$(id -un)" 2>/dev/null | cut -d: -f6
    fi
}

# The HOME reference to show in user output
get_home_expression() {
    if [ -n "${HOME:-}" ]; then
        # shellcheck disable=SC2016
        echo '$HOME'
    else
        get_home
    fi
}

# Replaces $HOME with the variable name for display, only if $HOME is defined
replace_home() {
    local _str="$1"

    if [ -n "${HOME:-}" ]; then
        echo "$_str" | sed "s,$HOME,\$HOME,"
    else
        echo "$_str"
    fi
}

# ========== Shell detection and configuration ==========

print_home_for_script() {
    local script="$1"

    local _home
    case "$script" in
        # zsh has a special ZDOTDIR directory
        .zsh*)
            if [ -n "${ZDOTDIR:-}" ]; then
                _home="$ZDOTDIR"
            else
                _home="$(get_home)"
            fi
            ;;
        *)
            _home="$(get_home)"
            ;;
    esac

    echo "$_home"
}

# ========== Environment script creation ==========

write_env_script_sh() {
    local _install_dir_expr="$1"
    local _env_script_path="$2"
    ensure cat <<EOF > "$_env_script_path"
#!/bin/sh
# add binaries to PATH if they aren't added yet
# affix colons on either side of \$PATH to simplify matching
case ":\${PATH}:" in
    *:"$_install_dir_expr":*)
        ;;
    *)
        # Prepending path in case a system-installed binary needs to be overridden
        export PATH="$_install_dir_expr:\$PATH"
        ;;
esac
EOF
}

write_env_script_fish() {
    local _install_dir_expr="$1"
    local _env_script_path="$2"
    ensure cat <<EOF > "$_env_script_path"
if not contains "$_install_dir_expr" \$PATH
    # Prepending path in case a system-installed binary needs to be overridden
    set -x PATH "$_install_dir_expr" \$PATH
end
EOF
}

# ========== PATH modification functions ==========

add_install_dir_to_path() {
    # Edit rcfiles ($HOME/.profile) to add install_dir to $PATH
    #
    # We do this slightly indirectly by creating an "env" shell script which checks if install_dir
    # is on $PATH already, and prepends it if not. The actual line we then add to rcfiles
    # is to just source that script. This allows us to blast it into lots of different rcfiles and
    # have it run multiple times without causing problems. It's also specifically compatible
    # with the system rustup uses, so that we don't conflict with it.
    local _install_dir_expr="$1"
    local _env_script_path="$2"
    local _env_script_path_expr="$3"
    local _rcfiles="$4"
    local _shell="$5"

    local _inferred_home="$(get_home)"
    if [ -n "$_inferred_home" ]; then
        local _target
        local _home

        # Find the first file in the array that exists and choose that as our target
        for _rcfile_relative in $_rcfiles; do
            _home="$(print_home_for_script "$_rcfile_relative")"
            local _rcfile="$_home/$_rcfile_relative"

            if [ -f "$_rcfile" ]; then
                _target="$_rcfile"
                break
            fi
        done

        # If we didn't find anything, pick the first entry as the default
        if [ -z "${_target:-}" ]; then
            local _rcfile_relative
            _rcfile_relative="$(echo "$_rcfiles" | awk '{ print $1 }')"
            _home="$(print_home_for_script "$_rcfile_relative")"
            _target="$_home/$_rcfile_relative"
        fi

        # `. x` is more portable than `source x`
        local _robust_line=". \"$_env_script_path_expr\""
        local _pretty_line="source \"$_env_script_path_expr\""

        # Add the env script if it doesn't already exist
        if [ ! -f "$_env_script_path" ]; then
            say_verbose "creating $_env_script_path"
            if [ "$_shell" = "sh" ]; then
                write_env_script_sh "$_install_dir_expr" "$_env_script_path"
            else
                write_env_script_fish "$_install_dir_expr" "$_env_script_path"
            fi
        else
            say_verbose "$_env_script_path already exists"
        fi

        # Check if the line is already in the rcfile
        if ! grep -F "$_robust_line" "$_target" > /dev/null 2>/dev/null && \
           ! grep -F "$_pretty_line" "$_target" > /dev/null 2>/dev/null
        then
            # If the script now exists, add the line to source it to the rcfile
            if [ -f "$_env_script_path" ]; then
                local _line
                # Fish has deprecated `.` as an alias for `source`
                if [ "$_shell" = "fish" ]; then
                    _line="$_pretty_line"
                else
                    _line="$_robust_line"
                fi
                say_verbose "adding $_line to $_target"
                # prepend an extra newline in case the user's file is missing a trailing one
                ensure echo "" >> "$_target"
                ensure echo "$_line" >> "$_target"
                return 1
            fi
        else
            say_verbose "$_install_dir_expr already on PATH"
        fi
    fi
}

shotgun_install_dir_to_path() {
    # Edit rcfiles ($HOME/.profile) to add install_dir to $PATH
    # (Shotgun edition - write to all provided files that exist rather than just the first)
    local _install_dir_expr="$1"
    local _env_script_path="$2"
    local _env_script_path_expr="$3"
    local _rcfiles="$4"
    local _shell="$5"

    local _inferred_home="$(get_home)"
    if [ -n "$_inferred_home" ]; then
        local _found=false
        local _home

        for _rcfile_relative in $_rcfiles; do
            _home="$(print_home_for_script "$_rcfile_relative")"
            local _rcfile_abs="$_home/$_rcfile_relative"

            if [ -f "$_rcfile_abs" ]; then
                _found=true
                add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" "$_rcfile_relative" "$_shell"
            fi
        done

        # Fall through to previous "create + write to first file in list" behavior
        if [ "$_found" = false ]; then
            add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" "$_rcfiles" "$_shell"
        fi
    fi
}

add_to_ci_path() {
    # Attempt to do CI-specific rituals to get the install-dir on PATH faster
    local _install_dir="$1"

    # If GITHUB_PATH is present, then write install_dir to the file it refs.
    # After each GitHub Action, the contents will be added to PATH.
    if [ -n "${GITHUB_PATH:-}" ]; then
        ensure echo "$_install_dir" >> "$GITHUB_PATH"
        say_verbose "Added $_install_dir to GITHUB_PATH for CI"
    fi
}

check_for_shadowed_bins() {
    local _install_dir="$1"
    local _bins="$2"
    local _shadowed_bins=""

    for _bin_name in $_bins; do
        local _shadow
        _shadow="$(command -v "$_bin_name" 2>/dev/null || true)"
        if [ -n "$_shadow" ] && [ "$_shadow" != "$_install_dir/$_bin_name" ]; then
            _shadowed_bins="$_shadowed_bins $_bin_name"
        fi
    done

    echo "$_shadowed_bins"
}
