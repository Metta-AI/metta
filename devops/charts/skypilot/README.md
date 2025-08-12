![SkyPilot](https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/images/skypilot-wide-light-1k.png)

[![Documentation](https://img.shields.io/badge/docs-gray?logo=readthedocs&logoColor=f5f5f5)](https://docs.skypilot.co/)
[![GitHub Release](https://img.shields.io/github/release/skypilot-org/skypilot.svg)](https://github.com/skypilot-org/skypilot/releases)
[![Join Slack](https://img.shields.io/badge/SkyPilot-Join%20Slack-blue?logo=slack)](http://slack.skypilot.co)
[![Downloads](https://img.shields.io/pypi/dm/skypilot)](https://github.com/skypilot-org/skypilot/releases)

## Run AI on Any Infra — Unified, Faster, Cheaper

### [🌟 **SkyPilot Demo** 🌟: Click to see a 1-minute tour](https://demo.skypilot.co/dashboard/)

---

## Softmax patch

This is a copy of [upstream chart](https://github.com/skypilot-org/skypilot/tree/master/charts/skypilot) with patches
for:

1. `cert-manager` support
2. Patch for automatically obtaining ECR credentials and auto-detecting ECR region.

Note: we should remove the chart later when those features are supported in upstream chart version.
