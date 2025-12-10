#!/usr/bin/env bash
#
# Fetch dinky:v14 from Observatory S3 and copy it to the metta2 host (and optionally into the metta container).
#
# Defaults assume:
#   - AWS SSO profile "softmax" is logged in on this machine.
#   - SSH access to metta@metta2.
#   - Container name "metta" running on metta2.
#
# Override with env vars:
#   AWS_PROFILE, REMOTE_HOST, REMOTE_USER, REMOTE_PATH, CONTAINER_NAME, CONTAINER_PATH
set -euo pipefail

AWS_PROFILE="${AWS_PROFILE:-softmax}"
REMOTE_USER="${REMOTE_USER:-metta}"
REMOTE_HOST="${REMOTE_HOST:-metta2}"
REMOTE_PATH="${REMOTE_PATH:-/home/${REMOTE_USER}/dinky_v14.zip}"
CONTAINER_NAME="${CONTAINER_NAME:-metta}"
CONTAINER_PATH="${CONTAINER_PATH:-/workspace/policies/dinky_v14.zip}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="${HERE}/tmp"
LOCAL_ZIP="${TMP_DIR}/dinky_v14.zip"

mkdir -p "${TMP_DIR}"

echo "Resolving metta://policy/72d89b2c-3a72-4dab-b885-d7e30958f615 to S3 (using ${AWS_PROFILE})..."
S3_URI="$(
  AWS_PROFILE="${AWS_PROFILE}" python - <<'PY'
from metta.rl.metta_scheme_resolver import MettaSchemeResolver
resolver = MettaSchemeResolver()
uri = "metta://policy/72d89b2c-3a72-4dab-b885-d7e30958f615"
print(resolver.get_path_to_policy_spec_or_mpt(uri))
PY
)"

echo "Downloading ${S3_URI} -> ${LOCAL_ZIP}"
AWS_PROFILE="${AWS_PROFILE}" aws s3 cp "${S3_URI}" "${LOCAL_ZIP}"

echo "SCP to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
scp "${LOCAL_ZIP}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"

echo "Upload into container ${CONTAINER_NAME}:${CONTAINER_PATH}"
ssh "${REMOTE_USER}@${REMOTE_HOST}" "docker cp '${REMOTE_PATH}' ${CONTAINER_NAME}:'${CONTAINER_PATH}'"

echo "Done. File locations:"
echo "  Local:   ${LOCAL_ZIP}"
echo "  Remote:  ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
echo "  Container: ${CONTAINER_NAME}:${CONTAINER_PATH}"

