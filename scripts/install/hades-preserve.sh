#!/usr/bin/env bash
#
# hades-preserve.sh - capture the HADES host state that a fresh OS install
# destroys, so hades-install.sh can put it back afterwards.
#
# RUN THIS BEFORE WIPING THE ROOT FILESYSTEM.
#
# WHAT THIS IS FOR
#
# On a machine where the ArangoDB data directory lives on ZFS, the wipe
# does not touch the database. What it destroys is the host identity that
# makes the surviving data usable: the secrets in /etc/hades, the systemd
# units, and the numeric UIDs and GIDs that own the files on the pool.
# ZFS records owners as numbers, not names, so a rebuilt machine that
# allocates a different UID to the `arangodb` account cannot read its own
# data directory.
#
# This script writes all of that to a tarball on a path you choose, which
# should be on one of the pools that survives the reinstall.
#
# The tarball contains ArangoDB credentials and MCP tokens. It is written
# mode 0600 and should be treated as a secret.
#
# USAGE
#
#   sudo scripts/install/hades-preserve.sh [--out DIR] [--dry-run]
#
#   --out DIR   Directory to write the archive to. Must survive the
#               reinstall. Default: /bulk-store/backups/hades-preserve
#   --dry-run   Print what would be captured, write nothing.
#
# Optionally, export ARANGO_ROOT_PASSWORD before running to also record a
# human-readable dump of ArangoDB users and their per-database grants.
# That data lives in the surviving data directory and is not needed for
# restore, but having it in plain text makes post-reinstall verification
# a diff instead of a memory test.
#
# See docs/install/reinstall-runbook.md for where this fits in the
# overall sequence.

set -euo pipefail

OUT_DIR=/bulk-store/backups/hades-preserve
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)     OUT_DIR="${2:?--out needs a directory}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ $EUID -ne 0 ]]; then
    echo "must run as root: /etc/hades/daemon.conf is root:hades 0640" >&2
    exit 1
fi

# Timestamp comes from the clock, not from git, so repeated runs on the
# same day do not overwrite each other.
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

ARCHIVE="$OUT_DIR/hades-preserve-$STAMP.tar.gz"

say()  { printf '  %s\n' "$*"; }
warn() { printf '  WARNING: %s\n' "$*" >&2; }

echo "HADES preserve"
echo "  archive: $ARCHIVE"
[[ $DRY_RUN -eq 1 ]] && echo "  (dry run: nothing will be written)"
echo

# ---------------------------------------------------------------------
# 1. /etc/hades - configs and secrets
# ---------------------------------------------------------------------
echo "[1/5] /etc/hades"
if [[ -d /etc/hades ]]; then
    mkdir -p "$STAGE/etc/hades"
    # -a preserves mode and ownership. Ownership is by name in the tar,
    # which is why section 3 records the numeric IDs separately.
    cp -a /etc/hades/. "$STAGE/etc/hades/"
    while IFS= read -r f; do say "$f"; done < <(cd /etc/hades && ls -1)
else
    warn "/etc/hades does not exist - nothing to preserve"
fi

# ---------------------------------------------------------------------
# 2. systemd units, drop-ins, sysusers, tmpfiles
# ---------------------------------------------------------------------
echo
echo "[2/5] systemd state"
mkdir -p "$STAGE/etc/systemd/system" "$STAGE/etc/sysusers.d" "$STAGE/etc/tmpfiles.d"

shopt -s nullglob
for unit in /etc/systemd/system/hades-*.service; do
    cp -a "$unit" "$STAGE/etc/systemd/system/"
    say "$(basename "$unit")"
done
for dropin_dir in /etc/systemd/system/hades-*.service.d \
                  /etc/systemd/system/arangodb3.service.d; do
    [[ -d "$dropin_dir" ]] || continue
    cp -a "$dropin_dir" "$STAGE/etc/systemd/system/"
    say "$(basename "$dropin_dir")/ ($(ls -1 "$dropin_dir" | wc -l) file(s))"
done
for f in /etc/sysusers.d/hades.conf /etc/tmpfiles.d/hades.conf; do
    [[ -f "$f" ]] || { warn "missing: $f"; continue; }
    cp -a "$f" "$STAGE${f}"
    say "$f"
done
shopt -u nullglob

# Which units were actually enabled. Restoring a unit file is not the
# same as restoring the enable state.
systemctl list-unit-files --no-legend --no-pager \
    'hades-*' 'arangodb3.service' > "$STAGE/enabled-units.txt" 2>/dev/null || true
say "enabled-units.txt"

# ---------------------------------------------------------------------
# 3. Numeric identity - the part that actually breaks
# ---------------------------------------------------------------------
echo
echo "[3/5] numeric identity"
{
    echo "# UID/GID assignments at preserve time."
    echo "# ZFS stores numeric owners. If the rebuilt machine assigns"
    echo "# different numbers, files on the surviving pools become"
    echo "# unreadable to the service that owns them. Re-create these"
    echo "# accounts with these exact IDs BEFORE installing packages that"
    echo "# would allocate them dynamically (notably arangodb3)."
    echo
    for u in arangodb hades; do
        if id "$u" >/dev/null 2>&1; then
            printf '%s_UID=%s\n' "${u^^}" "$(id -u "$u")"
            printf '%s_GID=%s\n' "${u^^}" "$(id -g "$u")"
        else
            echo "# user $u not present"
        fi
    done
    for g in arangodb hades weaver-admin; do
        if getent group "$g" >/dev/null 2>&1; then
            printf 'GROUP_%s_GID=%s\n' "$(echo "$g" | tr 'a-z-' 'A-Z_')" \
                   "$(getent group "$g" | cut -d: -f3)"
            printf 'GROUP_%s_MEMBERS=%s\n' "$(echo "$g" | tr 'a-z-' 'A-Z_')" \
                   "$(getent group "$g" | cut -d: -f4)"
        else
            echo "# group $g not present"
        fi
    done
    echo
    echo "# Observed owner of the ArangoDB data directory. This is the"
    echo "# number that matters: arangod must run as it."
    if [[ -d /var/lib/arangodb3 ]]; then
        printf 'DATADIR_UID=%s\n' "$(stat -c %u /var/lib/arangodb3)"
        printf 'DATADIR_GID=%s\n' "$(stat -c %g /var/lib/arangodb3)"
    fi
} > "$STAGE/identity.env"
sed 's/^/  /' "$STAGE/identity.env" | grep -v '^  #' | grep -v '^  $' || true

# ---------------------------------------------------------------------
# 4. Environment manifest - for reproducing versions, not for restore
# ---------------------------------------------------------------------
echo
echo "[4/5] environment manifest"
{
    echo "preserved_at: $STAMP"
    echo "hostname: $(hostname)"
    echo "kernel: $(uname -r)"
    echo "os: $(. /etc/os-release && echo "$PRETTY_NAME")"
    echo "arangodb_pkg: $(dpkg-query -W -f='${Version}' arangodb3 2>/dev/null || echo absent)"
    echo "system_python: $(python3 --version 2>&1 || echo absent)"
    echo "rustc: $(sudo -u "${SUDO_USER:-root}" sh -lc 'rustc --version' 2>/dev/null || echo absent)"
    echo "hades_bin: $(/usr/local/bin/hades --version 2>/dev/null || echo absent)"
    echo
    echo "# ZFS datasets and mountpoints. The reinstall must reproduce"
    echo "# these mountpoints or the surviving data lands in the wrong place."
    zfs list -o name,mountpoint 2>/dev/null || echo "# zfs not available"
    echo
    echo "# Filesystem of each path HADES depends on. Anything not zfs is"
    echo "# lost on wipe."
    for p in / /etc /home/todd /home/todd/olympus /usr/local/bin \
             /var/cache /var/lib/arangodb3 /opt/weaver/huggingface; do
        printf '%-32s %s\n' "$p" \
            "$(df --output=source,fstype "$p" 2>/dev/null | tail -1 || echo missing)"
    done
} > "$STAGE/manifest.txt"
say "manifest.txt"

# Optional: record ArangoDB users and grants for post-restore comparison.
if [[ -n "${ARANGO_ROOT_PASSWORD:-}" ]] && [[ -S /run/arangodb3/arangodb.sock ]]; then
    if curl -fsS --unix-socket /run/arangodb3/arangodb.sock \
            -u "root:${ARANGO_ROOT_PASSWORD}" \
            http://localhost/_api/user > "$STAGE/arango-users.json" 2>/dev/null; then
        say "arango-users.json (grants recorded for verification)"
    else
        warn "could not query ArangoDB users - check ARANGO_ROOT_PASSWORD"
        rm -f "$STAGE/arango-users.json"
    fi
else
    say "arango-users.json skipped (set ARANGO_ROOT_PASSWORD to include)"
fi

# ---------------------------------------------------------------------
# 5. Write the archive
# ---------------------------------------------------------------------
echo
echo "[5/5] archive"
if [[ $DRY_RUN -eq 1 ]]; then
    echo "  dry run: would write $ARCHIVE"
    echo "  staged contents:"
    (cd "$STAGE" && find . -type f | sed 's|^\./|    |' | sort)
    exit 0
fi

mkdir -p "$OUT_DIR"
chmod 700 "$OUT_DIR"
(cd "$STAGE" && tar czf "$ARCHIVE" .)
chmod 600 "$ARCHIVE"
ln -sfn "$(basename "$ARCHIVE")" "$OUT_DIR/latest.tar.gz"

sha256sum "$ARCHIVE" | tee "$ARCHIVE.sha256" >/dev/null
chmod 600 "$ARCHIVE.sha256"

echo "  wrote $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
echo "  sha256: $(cut -d' ' -f1 < "$ARCHIVE.sha256")"
echo "  symlink: $OUT_DIR/latest.tar.gz"
echo
echo "This archive contains ArangoDB credentials and MCP tokens. It is"
echo "mode 0600. Confirm it is on a pool that survives the reinstall:"
df --output=source,fstype "$OUT_DIR" | tail -1 | sed 's/^/  /'
echo
echo "Next: docs/install/reinstall-runbook.md"
