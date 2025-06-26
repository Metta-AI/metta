#!/bin/bash

cd_repo_root() {
    local current_dir="$(pwd)"
    local search_dir="$current_dir"

    # Search current directory and all parent directories
    while [[ "$search_dir" != "/" ]]; do
        if [[ -d "$search_dir/.git" ]]; then
            cd "$search_dir"
            return 0
        fi
        search_dir="$(dirname "$search_dir")"
    done

    # Check root directory as well
    if [[ -d "/.git" ]]; then
        cd "/"
        return 0
    fi

    echo "Repository root not found - no .git directory in current path or parent directories" >&2
    return 1
}

export AWS_PROFILE=softmax

# list jobs
alias jj="sky jobs queue --skip-finished"
alias jja="sky jobs queue"

# cancel ("kill") job
alias jk="sky jobs cancel -y"
alias jka="sky jobs cancel -a -y"

# get logs
alias jl="sky jobs logs"
alias jlc="sky jobs logs --controller"
alias jll='sky jobs logs $(jj | grep -A1 TASK | grep -v TASK | awk "{print \$1}")'
alias jllc='sky jobs logs --controller $(jj | grep -A1 TASK | grep -v TASK | awk "{print \$1}")'

# launch training
unalias lt 2>/dev/null
lt() {
    local original_dir="$(pwd)"
    if cd_repo_root; then
        ./devops/skypilot/launch.py "$@"
        local exit_code=$?
        cd "$original_dir"
        return $exit_code
    fi
}
