This stack configures the Spacelift itself.

It intentionally doesn't create stacks as
[resources](https://registry.terraform.io/providers/spacelift-io/spacelift/latest/docs/resources/stack), though it
could.

The reason is that we want to deploy stacks on commits to `main` branch, but it's impossible to test stacks before the
PR is merged.
