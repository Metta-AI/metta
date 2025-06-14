export AWS_PROFILE=softmax

# list jobs
alias jq="sky jobs queue --skip-finished"
alias jqa="sky jobs queue"

# cancel ("kill") job
alias jk="sky jobs cancel"

# get logs
alias jl="sky jobs logs"
alias jlc="sky jobs logs --controller"
alias jll='sky jobs logs $(jq | grep -A1 TASK | grep -v TASK | awk "{print \$1}")'
alias jllc='sky jobs logs --controller $(jq | grep -A1 TASK | grep -v TASK | awk "{print \$1}")'

# launch training
alias lt="./devops/skypilot/launch.py train"
