# Sandbox Management Guide

This guide provides instructions for setting up and managing sandbox environments on AWS.

## Initial Setup

To set up a new Mac machine:

```bash
python devops/aws/setup_machine.py
```

## Sandbox Operations

### Launching a Sandbox

To launch a new sandbox on AWS:

```bash
./devops/aws/cmd.sh launch --cmd=sandbox --run=sandbox --job-name=<sandbox_name>
```

### Connecting to a Sandbox

To connect to an existing sandbox:

1. First claim it in Asana: [Dev Cluster](https://app.asana.com/1/1209016784099267/project/1209353759349008/task/1210106185904866?focus=true)
2. Then connect using:

```bash
./devops/aws/cmd.sh ssh --job-name=<sandbox_name>
```

### Stopping a Sandbox

To stop a running sandbox:

```bash
./devops/aws/cmd.sh stop <sandbox_name>
```


