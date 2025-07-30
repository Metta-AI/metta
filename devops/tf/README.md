# How to use Spacelift to manage Terraform

OpenTofu (Terraform) configs are managed by [Spacelift](https://spacelift.io/).

This directory contains [stacks](https://docs.spacelift.io/concepts/stack/creating-a-stack): Terraform configurations
for different parts of the infrastructure.

Note that `spacelift` stack configures Spacelift, but not the individual stacks in it (see `spacelift/README.md` for
details).

## Common workflow

Terraform works in two stages: first, `plan`, and then `apply`. Planning is relatively safe, so it can be done
automatically, or even locally without pushing to the repo.

Applying should ideally happen only from the `main` branch, but see below for the possible loopholes.

General workflow:

1. Update terraform files
2. Register new stacks in Spacelift if necessary
3. Push changes to GitHub and open a PR
4. Check the planned changes in Spacelift or GitHub checks (under **spacelift.io** section)
5. Apply the changes - use one of the approaches described below

## Creating new stacks

- Read **Creating a stack** in the [Spacelift docs](https://docs.spacelift.io/concepts/stack/creating-a-stack).
- Use the latest OpenTofu instead of Terraform.
- If the stack includes AWS resources, don't forget to "Attach cloud" during stack creation, or later in **Stack
  Settings → Integrations**.
- (Optional) Enable **Local Preview** under **Stack Settings → Behavior** for faster iteration.

When working on a PR that adds a new stack, you can enable **Run Promotion** (see
[Applying without merging](#applying-without-merging) below) and iterate on it before the PR is merged.

## Previewing plans without pushing to GitHub

You can preview changes without pushing anything:

```bash
spacectl stack local-preview
```

This sends your local changes to Spacelift and shows the planned diff.

**Enable local preview** option must be enabled in stack settings for this to work.

It's not possible to apply the stack locally without pushing to the repo.

## Applying changes

### Careful approach - applying from `main` manually

1. Iterate on the PR until the plan looks good.
2. Merge the PR.
3. Go to the stack in Spacelift and apply the latest `main` run.

### Automatic approach - applying from `main` on merge

1. Enable **Autodeploy** in the stack's settings (once).
2. Merge the PR.

### Applying without merging

Use sparingly - this makes `main` drift from deployed state, but is handy for quick tests:

1. Enable **Allow run promotion** in the stack's settings.
2. Push changes to the PR.
3. In Spacelift (or the GitHub check), find the planned run and click **Promote** (or **Deploy** on GitHub).
