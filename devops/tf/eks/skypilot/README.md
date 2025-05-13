Terraform module for Skypilot.

`./skypilot-chart` includes the copy of [upstream chart](https://github.com/skypilot-org/skypilot/tree/master/charts/skypilot) with patches for:
1. `--host 0.0.0.0` bug
2. `cert-manager` support
3. Patch for automatically obtaining ECR credentials

Note: we should remove the chart later when those features are supported in upstream chart version.
