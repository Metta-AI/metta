#!/bin/bash -e

ssh_cmd=$(vastai show instances  | tail -n1 | awk '{print "ssh -p " $11 " root@" $10}')
cmd="bash -c 'source ~/.bashrc; cd /workspace/metta/ ; $@'"
full_cmd="$ssh_cmd \"$cmd\""
echo $full_cmd
$full_cmd
