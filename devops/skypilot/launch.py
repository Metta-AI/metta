#!.venv/skypilot/bin/python3

import argparse
import subprocess

import sky


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", help="Command to run")
    parser.add_argument("run", help="Run ID")
    parser.add_argument("--git-ref", type=str, default=None)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    (args, cmd_args) = parser.parse_known_args()

    git_ref = args.git_ref
    if not git_ref:
        git_ref = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()

    cmd = [
        "sky",
        "jobs",
        "launch",
        "--num-nodes", str(args.nodes),
        "--cpus", f"{args.cpus}+",
        "--name", args.run,
        "./devops/skypilot/config/sk_train.yaml",
        "--env", f"METTA_RUN_ID={args.run}",
        "--env", f"METTA_CMD={args.cmd}",
        "--env", f"METTA_CMD_ARGS={' '.join(cmd_args)}",
        "--env", f"METTA_GIT_REF={git_ref}",
        "--detach-run",
        "--async",
        "--yes",
    ]

    if args.dry_run:
        print("DRY RUN:")
        print(" ".join(cmd))
    else:
        result = subprocess.run(cmd)
        exit(result.returncode)
        # task = sky.Task.from_yaml("./devops/skypilot/config/sk_train.yaml")
        # task = task.update_env(dict(
        #     METTA_RUN_ID=args.run,
        #     METTA_CMD=args.cmd,
        #     METTA_CMD_ARGS=" ".join(cmd_args),
        #     METTA_GIT_REF=git_ref,
        # ))
        # result = sky.jobs.launch(task)
        # print(result)


if __name__ == "__main__":
    main()
