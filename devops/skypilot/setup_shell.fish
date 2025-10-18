#!/usr/bin/env fish

# Fish-compatible setup for SkyPilot helpers

function cd_repo_root
  set -l starting_dir (pwd)
  while test "$PWD" != "/" -a ! -d ".git"
    cd ..
  end

  if not test -d ".git"
    echo "Repository root not found - no .git directory in current path or parent directories" >&2
    cd $starting_dir
    return 1
  end
end

# Environment
set -x AWS_PROFILE softmax

# Aliases
alias jj "sky jobs queue --skip-finished"
alias jja "sky jobs queue"

alias jk "sky jobs cancel -y"
alias jka "sky jobs cancel -a -y"

alias jl "sky jobs logs"
alias jlc "sky jobs logs --controller"

function jll
  sky jobs logs (jj | grep -A1 TASK | grep -v TASK | awk '{print $1}')
end

function jllc
  sky jobs logs --controller (jj | grep -A1 TASK | grep -v TASK | awk '{print $1}')
end

# Ensure no conflicting function exists
functions -q lt; and functions -e lt

function lt
  set -l original_dir (pwd)
  cd_repo_root
  set -l repo_result $status

  if test $repo_result -eq 0
    ./devops/skypilot/launch.py $argv
    set -l exit_code $status
  else
    set -l exit_code $repo_result
  end

  cd $original_dir
  return $exit_code
end


