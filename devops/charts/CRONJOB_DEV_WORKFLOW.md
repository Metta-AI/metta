# Dev Cronjob Workflow

Quick guide for deploying and cleaning up dev cronjob instances.

## Deploy Dev Cronjob

1. **Uncomment the dev release** in `helmfile.yaml`:

   ```yaml
   - name: pr-similarity-cache-cronjob-dev
     <<: *cronjob_template
     values:
       - schedule: "*/5 * * * *"
       - datadog:
           service: pr-similarity-cache-refresh-dev
           env: dev
   ```

2. **Commit and push** to your feature branch:

   ```bash
   git add devops/charts/helmfile.yaml
   git commit -m "chore: enable dev cronjob for testing"
   git push
   ```

3. **Trigger workflow** from your branch with dev flag:

   ```bash
   gh workflow run deploy-pr-similarity-cache-cronjob.yml \
     -f dev=true \
     -f branch=your-branch-name
   ```

   > **Note**: The `branch` parameter ensures the workflow uses your helmfile with the dev release uncommented.

4. **Monitor deployment**:

   ```bash
   kubectl get cronjobs -n monitoring | grep dev
   kubectl get jobs -n monitoring | grep dev
   kubectl logs -n monitoring job/<job-name>
   ```

5. **Manually trigger a run** (optional - don't wait for schedule):
   ```bash
   kubectl create job -n monitoring test-run --from=cronjob/pr-similarity-cache-cronjob-dev
   kubectl logs -n monitoring job/test-run -f
   ```

## Remove Dev Cronjob

1. **Re-comment the dev release** in `helmfile.yaml`:

   ```yaml
   # - name: pr-similarity-cache-cronjob-dev
   #   <<: *cronjob_template
   #   ...
   ```

2. **Commit and push** to your branch (or merge to main):

   ```bash
   git add devops/charts/helmfile.yaml
   git commit -m "chore: remove dev cronjob"
   git push
   ```

3. **Manually delete** the Helm release:
   ```bash
   helm uninstall -n monitoring pr-similarity-cache-cronjob-dev
   ```

> **Note**: Pushing changes to `helmfile.yaml` on main triggers prod deployments, but dev releases must be manually
> deleted via `helm uninstall` - they won't auto-remove when you re-comment them.

## Why This Pattern?

- **Helmfile defines structure**: Declares what cronjobs exist
- **GitHub Actions deploys**: Builds images and runs Helm with specific SHA tags
- **Dev flag creates separate instance**: Different release name, won't interfere with prod
- **Manual cleanup required**: Commenting out code doesn't delete K8s resources
