#!/bin/sh
# Pack the agent plugin package reproducibly: sorted member order, fixed
# owner and mtime, and a gzip header without name or timestamp, so the
# same content hashes identically across filesystems and machines. Used
# by both `just plugin-tarball` and the release workflow.
set -eu

OUT="${1:?usage: pack-plugin.sh <out.tar.gz>}"

# The member set has one manifest, shared with the crate build script and the
# crate staging script (scripts/plugin-package-members.txt).
members=$(sed 's/[[:space:]]*$//' "$(dirname "$0")/plugin-package-members.txt" | grep -v '^$' || true)

# shellcheck disable=SC2086 # the manifest lists whitespace-free members
tar --sort=name \
    --owner=root --group=root --numeric-owner \
    --mtime='2026-01-01 00:00:00 UTC' \
    --exclude='__pycache__' --exclude='*.pyc' \
    -cf - \
    $members \
  | gzip -n > "$OUT"

# License retention (RFC-0001:C-LICENSE-RETENTION): the plugin package packs
# the notice, asserted here.
tar -tzf "$OUT" | grep -q '^LICENSE$'
tar -tzf "$OUT" | grep -q '^plugins/inferlab/skills/inferlab/SKILL.md$'
tar -tzf "$OUT" | grep -q '^plugins/inferlab/skills/inferlab/references/capability-map.md$'
tar -tzf "$OUT" | grep -q '^plugins/inferlab/skills/inferlab/references/workspace-authoring.md$'
! tar -tzf "$OUT" | grep -q '^docs/workspace-authoring.md$'
tar -tzf "$OUT" | grep -q '^docs/backend-support.md$'
tar -tzf "$OUT" | grep -q '^.claude-plugin/marketplace.json$'
tar -tzf "$OUT" | grep -q '^.agents/plugins/marketplace.json$'
