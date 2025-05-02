#!/usr/bin/env bash
set -e 

# check that the user is root
if [ "$EUID" -ne 0 ]; then
    echo "This script must be run as root"
    exit 1
fi

# Adapted from `brew info amazon-efs-utils`:

# To start using Amazon EFS
echo "Symlinking mount.efs script"
mkdir -p /Library/Filesystems/efs.fs/Contents/Resources
ln -sf $(brew --prefix)/bin/mount.efs /Library/Filesystems/efs.fs/Contents/Resources/mount_efs

# To enable watchdog for TLS mounts
# (`brew info` instructions are outdated, see https://github.com/aws/homebrew-aws/issues/27)
echo "Enabling watchdog for TLS mounts"
cp $(brew --prefix)/opt/amazon-efs-utils/libexec/amazon-efs-mount-watchdog.plist /Library/LaunchAgents
launchctl unload /Library/LaunchAgents/amazon-efs-mount-watchdog.plist || true
launchctl load /Library/LaunchAgents/amazon-efs-mount-watchdog.plist

# Note: `launchctl load` warns that it's deprecated and that `launchctl bootstrap` with `LaunchDaemons` dir is the preferred way.
# But mount.efs checks for the watchdog in the LaunchAgents dir specifically.
# See also: https://github.com/aws/homebrew-aws/issues/27

echo "Creating /Volumes/metta-efs"
mkdir -p /Volumes/metta-efs
