#!/bin/bash

cd_repo_root() {
    while [[ "$PWD" != "/" && ! -d ".git" ]]; do
        cd ..
    done

    if [[ ! -d ".git" ]]; then
        echo "Repository root not found - no .git directory in current path or parent directories" >&2
        return 1
    fi
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
    cd_repo_root
    local repo_result=$?
    
    if [ $repo_result -eq 0 ]; then
        ./devops/skypilot/launch.py "train" "$@"
        local exit_code=$?
    else
        local exit_code=$repo_result
    fi
    
    cd "$original_dir"
    return $exit_code
}


