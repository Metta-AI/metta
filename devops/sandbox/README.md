to setup a new mac:

python devops/aws/setup_machine.py

to launch a sandbox on aws:

./devops/aws/cmd.sh launch  --cmd=sandbox --run=sandbox --job-name=<sandbox_name>

to connect to the sandbox:

./devops/aws/cmd.sh ssh --job-name=<sandbox_name>

to stop the sandbox:

./devops/aws/cmd.sh stop <sandbox_name>


