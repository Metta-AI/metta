export AWS_PROFILE=softmax-db-admin

<<<<<<< HEAD
#sky
alias sky=".venv/skypilot/bin/sky"

=======
>>>>>>> 13c12a2fdf120e435aa056c95de09aa7ccaa5a87
# list jobs
alias jq="sky jobs queue --skip-finished"
alias jqa="sky jobs queue"

# cancel job
alias jk="sky jobs cancel"
alias jc="sky jobs cancel"
alias jkl="sky jobs cancel"

# get logs
alias jl="sky jobs logs"
alias jlc="sky jobs logs --controller"
alias jll='sky jobs logs $(jq | grep -A1 TASK | grep -v TASK | awk "{print \$1}")'
alias jllc='sky jobs logs --controller $(jq | grep -A1 TASK | grep -v TASK | awk "{print \$1}")'

# launch training
alias lt="./devops/skypilot/launch.sh train"
