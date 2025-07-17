#!/bin/bash

# Script to view Docker Compose logs with full-width lines (no wrapping)

# Use script command to remove TTY detection and pipe to less with -S flag
docker compose logs --follow --no-log-prefix "$@" | less -S +F