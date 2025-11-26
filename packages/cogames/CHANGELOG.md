# Changelog

## Dec 3, 2025

**What's new**

Breaking change: The format for specifying a policy in `cogames` command is now updated to support passing keyword
arguments.

- cogames `-p` or `--policy` arguments should now take the form `class=...[,data=...][,proportion=...] [,kw.name=val]`

- Example: uv run cogames play -m hello_world -p class=path.to.MyPolicyClass,data=my_data/policy.pt,kw.policy_mode=fast

Update: `cogames submit` now accepts much larger checkpoint files. It makes use of presigned s3 URLs for uploading your
checkpoint data instead of going through web servers that enforced lower file size caps.

Update: `cogames submit` does a light round of validation on your proposed submission before uploading it. It ensures
that your policy is loadable (using the latest copy of `cogames` in pypi, not necessarily the one you have installed),
and can perform a step on a simple environment.
