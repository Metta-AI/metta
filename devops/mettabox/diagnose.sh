#!/usr/bin/env bash
#
# This script gathers logs and crash-related data from the previous boot
# and packages everything into a tarball for offline inspection.
#
# Usage: Run as root (e.g., sudo ./diagnose.sh). It will:
#   1. Ensure systemd-journald persistence is enabled.
#   2. Identify the previous boot (“-1”).
#   3. Dump the full journal of the previous boot and filter for common errors.
#   4. Copy /var/log/kern.log* and /var/log/syslog* around the crash.
#   5. Capture dmesg from the previous boot.
#   6. List and copy any files in /var/crash and /var/lib/kdump.
#   7. Create a timestamped directory under $HOME and tar it into ~/crash-inspection.tar.gz.
#
# Once done, you can “scp” or otherwise retrieve ~/crash-inspection.tar.gz for analysis.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "❗️ This script must be run as root. Try: sudo $0" >&2
  exit 1
fi

# 1. Ensure persistent journaling (so older boots are retained)
if [[ ! -d /var/log/journal ]]; then
  echo "🔧 Creating /var/log/journal for persistent logs…"
  mkdir -p /var/log/journal
  systemd-tmpfiles --create --prefix /var/log/journal
  systemctl restart systemd-journald
  echo "✅ Persistent journaling enabled."
else
  echo "ℹ️ Persistent journaling already enabled (/var/log/journal exists)."
fi

# 2. Define the “previous boot” index
PREV_BOOT=-1

# 3. Create a timestamped output directory in the current user's home
TIMESTAMP="$(date '+%Y%m%d%H%M%S')"
OUTPUT_DIR="./crash-inspection-$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"
echo "📂 Created output directory: $OUTPUT_DIR"

# 4. Dump the full journal from the previous boot
echo "📝 Saving full journal of boot ${PREV_BOOT} to $OUTPUT_DIR/journal-boot-previous.log"
journalctl -b "${PREV_BOOT}" > "$OUTPUT_DIR/journal-boot-previous.log"

# 5. Grep for common kernel panic / oops / BUG messages
echo "🔍 Extracting kernel panic/oops/BUG lines to $OUTPUT_DIR/journal-boot-previous-kernel-panics.log"
journalctl -b "${PREV_BOOT}" | grep -i -E 'panic|oops|BUG|kernel panic' \
  > "$OUTPUT_DIR/journal-boot-previous-kernel-panics.log" || true

# 6. Grep for Out-Of-Memory (OOM) killer invocations
echo "🔍 Extracting OOM-kill lines to $OUTPUT_DIR/journal-boot-previous-oom.log"
journalctl -b "${PREV_BOOT}" | grep -i -E 'killed process|out of memory|oom' \
  > "$OUTPUT_DIR/journal-boot-previous-oom.log" || true

# 7. Grep for NVIDIA driver errors (NVRM / Xid)
echo "🔍 Extracting NVIDIA NVRM/Xid errors to $OUTPUT_DIR/journal-boot-previous-nvidia.log"
journalctl -b "${PREV_BOOT}" | grep -i nvidia \
  > "$OUTPUT_DIR/journal-boot-previous-nvidia.log" || true

# 8. Grep for generic “error” or “fail” lines in that boot
echo "🔍 Extracting generic ERROR/FAIL lines to $OUTPUT_DIR/journal-boot-previous-errors.log"
journalctl -b "${PREV_BOOT}" | grep -i -E 'error|fail' \
  > "$OUTPUT_DIR/journal-boot-previous-errors.log" || true

# 9. Copy /var/log/kern.log and rotations
echo "📥 Copying /var/log/kern.log* to $OUTPUT_DIR/"
cp /var/log/kern.log* "$OUTPUT_DIR/" 2>/dev/null || true

# 10. Copy /var/log/syslog and rotations
echo "📥 Copying /var/log/syslog* to $OUTPUT_DIR/"
cp /var/log/syslog* "$OUTPUT_DIR/" 2>/dev/null || true

# 11. Dump dmesg from the previous boot
echo "📝 Saving previous-boot dmesg to $OUTPUT_DIR/dmesg-boot-previous.log"
journalctl -k -b "${PREV_BOOT}" > "$OUTPUT_DIR/dmesg-boot-previous.log"

# 12. List and copy any files in /var/crash
echo "📂 Listing /var/crash contents to $OUTPUT_DIR/var-crash-list.txt"
ls -lh /var/crash > "$OUTPUT_DIR/var-crash-list.txt" 2>/dev/null || true
echo "📥 Copying /var/crash/* to $OUTPUT_DIR/"
cp /var/crash/* "$OUTPUT_DIR/" 2>/dev/null || true

# 13. List and copy any files in /var/lib/kdump
echo "📂 Listing /var/lib/kdump contents to $OUTPUT_DIR/var-lib-kdump-list.txt"
ls -lh /var/lib/kdump > "$OUTPUT_DIR/var-lib-kdump-list.txt" 2>/dev/null || true
echo "📥 Copying /var/lib/kdump/* to $OUTPUT_DIR/"
cp /var/lib/kdump/* "$OUTPUT_DIR/" 2>/dev/null || true

# 14. Package everything into a tarball
TARBALL="$HOME/crash-inspection-$TIMESTAMP.tar.gz"
echo "📦 Creating tarball: $TARBALL"
tar czf "$TARBALL" -C "$OUTPUT_DIR" .

echo -e "\n✅ Done! All logs and crash data are in:\n    $TARBALL"
echo "You can now transfer this file for offline analysis."

