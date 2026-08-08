#!/usr/bin/env bash
#
# hades-install.sh - install HADES onto a machine, from scratch or after an
# OS reinstall.
#
# This automates the procedure documented in the README Install section.
# The README remains the explanation of why each step exists. This script
# is the executable form of it, plus the parts a real reinstall needs that
# a first-time install does not: restoring preserved secrets, and
# reconciling numeric UIDs against data that outlived the operating
# system.
#
# It is idempotent. Re-running against an installed machine converges the
# machine to the repository state and reports what it changed.
#
# USAGE
#
#   sudo scripts/install/hades-install.sh [options]
#
#   --from-preserve PATH   Restore secrets from a hades-preserve.sh
#                          archive (.tar.gz) or an already-extracted
#                          directory. Without this, daemon.conf is
#                          generated as a template you must fill in.
#   --arango-uid-fix MODE  What to do when the arangodb account's UID does
#                          not match the owner of the data directory:
#                            report  fail with instructions (default)
#                            adopt   move the account onto the data
#                                    directory's UID/GID (fast, needs
#                                    arangod stopped)
#                            chown   recursively chown the data directory
#                                    onto the account (slow on large data)
#   --services-dir DIR     Path substituted into the embedder and trainer
#                          units. Default: <repo>/services
#   --with-services        Also install the embedder, extractor and trainer
#                          units. Off by default: they need a GPU and a
#                          populated Python venv.
#   --no-start             Install everything but do not enable or start
#                          any unit.
#   --dry-run              Print every action, change nothing.
#
# See docs/install/reinstall-runbook.md for the full reinstall sequence,
# including what has to happen before this script can run.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

FROM_PRESERVE=""
ARANGO_UID_FIX=report
SERVICES_DIR="$REPO_ROOT/services"
WITH_SERVICES=0
NO_START=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from-preserve)  FROM_PRESERVE="${2:?--from-preserve needs a path}"; shift 2 ;;
        --arango-uid-fix) ARANGO_UID_FIX="${2:?--arango-uid-fix needs a mode}"; shift 2 ;;
        --services-dir)   SERVICES_DIR="${2:?--services-dir needs a path}"; shift 2 ;;
        --with-services)  WITH_SERVICES=1; shift ;;
        --no-start)       NO_START=1; shift ;;
        --dry-run)        DRY_RUN=1; shift ;;
        -h|--help)        sed -n '2,45p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "$ARANGO_UID_FIX" in
    report|adopt|chown) ;;
    *) echo "--arango-uid-fix must be report, adopt or chown" >&2; exit 2 ;;
esac

if [[ $EUID -ne 0 ]]; then
    echo "must run as root" >&2
    exit 1
fi

CHANGES=()
NOTES=()

step()   { printf '\n=== %s ===\n' "$*"; }
say()    { printf '  %s\n' "$*"; }
note()   { printf '  NOTE: %s\n' "$*"; NOTES+=("$*"); }
fail()   { printf '  ERROR: %s\n' "$*" >&2; exit 1; }
changed(){ CHANGES+=("$*"); }

# Every mutation goes through run(), so --dry-run is honest rather than
# approximate.
run() {
    if [[ $DRY_RUN -eq 1 ]]; then
        printf '  would run: %s\n' "$*"
    else
        "$@"
    fi
}

echo "HADES install"
say "repo:         $REPO_ROOT"
say "services dir: $SERVICES_DIR"
[[ -n "$FROM_PRESERVE" ]] && say "restoring from: $FROM_PRESERVE"
[[ $DRY_RUN -eq 1 ]] && say "(dry run: no changes will be made)"

# ---------------------------------------------------------------------
step "1. Preflight"
# ---------------------------------------------------------------------
for cmd in systemctl install getent stat; do
    command -v "$cmd" >/dev/null || fail "missing required command: $cmd"
done

command -v systemd-sysusers >/dev/null \
    || fail "systemd-sysusers not found - this installer targets systemd hosts"

if ! command -v arangod >/dev/null; then
    fail "arangod not found. Install ArangoDB first - see the README
  Prerequisites section. On a reinstall, create the arangodb account with
  its preserved UID BEFORE apt-get install, or the package will allocate a
  new one and lose access to the surviving data directory."
fi
say "arangod: $(arangod --version 2>/dev/null | head -1)"

BINARY="$REPO_ROOT/target/release/hades"
[[ -x "$BINARY" ]] || fail "no release binary at $BINARY - run: cargo build --release"
say "binary:  $("$BINARY" --version)"

# ---------------------------------------------------------------------
step "2. Restore preserved state"
# ---------------------------------------------------------------------
PRESERVE_DIR=""
if [[ -n "$FROM_PRESERVE" ]]; then
    if [[ -d "$FROM_PRESERVE" ]]; then
        PRESERVE_DIR="$FROM_PRESERVE"
    elif [[ -f "$FROM_PRESERVE" ]]; then
        PRESERVE_DIR=$(mktemp -d)
        trap 'rm -rf "$PRESERVE_DIR"' EXIT
        tar xzf "$FROM_PRESERVE" -C "$PRESERVE_DIR"
        say "extracted archive"
    else
        fail "--from-preserve path not found: $FROM_PRESERVE"
    fi

    if [[ -f "$PRESERVE_DIR/identity.env" ]]; then
        # shellcheck disable=SC1091
        source "$PRESERVE_DIR/identity.env"
        say "identity: arangodb=${ARANGODB_UID:-?}:${ARANGODB_GID:-?} datadir=${DATADIR_UID:-?}:${DATADIR_GID:-?}"
    else
        note "archive has no identity.env - UID reconciliation will use the live data directory only"
    fi
else
    say "no archive given - fresh install"
fi

# ---------------------------------------------------------------------
step "3. Reconcile the ArangoDB data directory owner"
# ---------------------------------------------------------------------
# This is the step that a reinstall onto surviving storage needs and a
# first-time install does not. ZFS records file owners as numbers. The
# arangodb3 package allocates its system account dynamically, so a
# rebuilt machine frequently ends up with an arangodb account whose UID
# does not match the data it is supposed to own.
DATADIR=/var/lib/arangodb3
if [[ -d "$DATADIR" ]] && id arangodb >/dev/null 2>&1; then
    disk_uid=$(stat -c %u "$DATADIR")
    disk_gid=$(stat -c %g "$DATADIR")
    acct_uid=$(id -u arangodb)
    acct_gid=$(id -g arangodb)

    if [[ "$disk_uid" == "$acct_uid" && "$disk_gid" == "$acct_gid" ]]; then
        say "owner matches: $DATADIR is ${disk_uid}:${disk_gid}, arangodb is ${acct_uid}:${acct_gid}"
    else
        say "MISMATCH: $DATADIR is owned by ${disk_uid}:${disk_gid}"
        say "          the arangodb account is ${acct_uid}:${acct_gid}"
        case "$ARANGO_UID_FIX" in
            report)
                fail "arangod cannot read its own data directory.

  Two remedies, pick one and re-run with the matching flag:

    --arango-uid-fix adopt
        Move the arangodb account onto ${disk_uid}:${disk_gid}. Fast,
        independent of data size. Requires arangod to be stopped.
        Preferred when the data directory is large.

    --arango-uid-fix chown
        Recursively chown $DATADIR to ${acct_uid}:${acct_gid}. Simple,
        but rewrites metadata for every file - slow on a large store.

  Doing nothing is not an option: arangod will fail to start."
                ;;
            adopt)
                if systemctl is-active --quiet arangodb3.service; then
                    say "stopping arangodb3 before moving the account"
                    run systemctl stop arangodb3.service
                fi
                run groupmod -g "$disk_gid" arangodb
                run usermod  -u "$disk_uid" -g "$disk_gid" arangodb
                # Files elsewhere still carry the old numeric owner.
                for p in /var/log/arangodb3 /var/lib/arangodb3-apps /run/arangodb3; do
                    [[ -e "$p" ]] && run chown -R arangodb:arangodb "$p"
                done
                changed "arangodb account moved to ${disk_uid}:${disk_gid}"
                say "account now ${disk_uid}:${disk_gid}"
                ;;
            chown)
                say "recursively chowning $DATADIR - this can take a while"
                run chown -R arangodb:arangodb "$DATADIR"
                changed "$DATADIR chowned to ${acct_uid}:${acct_gid}"
                ;;
        esac
    fi
elif [[ -d "$DATADIR" ]]; then
    note "$DATADIR exists but there is no arangodb account yet"
else
    say "no existing data directory - nothing to reconcile"
fi

# ---------------------------------------------------------------------
step "4. Service identity (users, groups, runtime directory)"
# ---------------------------------------------------------------------
# hades-sysusers.conf declares membership in weaver-admin. That group is
# owned by WeaverTools, not by HADES, and on a freshly installed machine
# it does not exist yet. systemd-sysusers fails on a membership line for
# a missing group, so the line is filtered out when the group is absent
# and picked up later once WeaverTools has been installed.
HAVE_WEAVER_ADMIN=1
if ! getent group weaver-admin >/dev/null 2>&1; then
    HAVE_WEAVER_ADMIN=0
    note "group weaver-admin does not exist. It belongs to WeaverTools, so
        HADES does not create it. Membership and the units'
        SupplementaryGroups entry are being skipped. After WeaverTools is
        installed, re-run this script to pick them up."
fi

SYSUSERS_SRC="$REPO_ROOT/services/systemd/hades-sysusers.conf"
if [[ $HAVE_WEAVER_ADMIN -eq 1 ]]; then
    run install -m 644 "$SYSUSERS_SRC" /etc/sysusers.d/hades.conf
else
    if [[ $DRY_RUN -eq 1 ]]; then
        say "would install /etc/sysusers.d/hades.conf without the weaver-admin membership"
    else
        grep -v '^m[[:space:]]\+hades[[:space:]]\+weaver-admin' "$SYSUSERS_SRC" \
            > /etc/sysusers.d/hades.conf
        chmod 644 /etc/sysusers.d/hades.conf
    fi
fi
run install -m 644 "$REPO_ROOT/services/systemd/hades-tmpfiles.conf" /etc/tmpfiles.d/hades.conf
run systemd-sysusers
run systemd-tmpfiles --create
changed "service identity applied"

if [[ $DRY_RUN -eq 0 ]]; then
    getent group hades >/dev/null || fail "group hades still missing after systemd-sysusers"
    say "hades: $(id hades)"
    say "/run/hades: $(stat -c '%U:%G %a' /run/hades 2>/dev/null || echo missing)"
fi

# ---------------------------------------------------------------------
step "5. Binary"
# ---------------------------------------------------------------------
# Two copies by design: the unit execs /usr/local/bin/hades, interactive
# shells usually resolve ~/.local/bin/hades first. Updating only one
# leaves the other silently running an older build.
run install -m 755 "$BINARY" /usr/local/bin/hades
changed "/usr/local/bin/hades"
say "/usr/local/bin/hades"

if [[ -n "${SUDO_USER:-}" ]]; then
    user_home=$(getent passwd "$SUDO_USER" | cut -d: -f6)
    if [[ -n "$user_home" ]]; then
        run install -d -o "$SUDO_USER" -g "$SUDO_USER" "$user_home/.local/bin"
        run install -m 755 -o "$SUDO_USER" -g "$SUDO_USER" "$BINARY" "$user_home/.local/bin/hades"
        changed "$user_home/.local/bin/hades"
        say "$user_home/.local/bin/hades"
    fi
else
    note "SUDO_USER not set - skipped the per-user copy in ~/.local/bin"
fi

# ---------------------------------------------------------------------
step "6. Configuration"
# ---------------------------------------------------------------------
run install -d -m 755 /etc/hades

# Non-secret configuration comes from the repository, so a reinstall
# converges on version control rather than resurrecting whatever had
# accumulated on the old root filesystem. Where the preserved copy
# differs, it is kept alongside as .preserved for the operator to
# reconcile deliberately.
install_tracked_conf() {
    local src="$1" dest="$2" mode="$3" owner="$4"
    local u="${owner%:*}" g="${owner#*:}"

    # Never clobber a differing file without leaving the old one behind.
    # The live copy can encode a real deployment choice that the
    # repository default does not know about - HF_HOME pointing at a
    # shared model cache is the standing example. Losing that silently
    # sends the next embedder start off to re-download several gigabytes.
    local existing=""
    if [[ -f "$dest" ]] && ! diff -q "$dest" "$src" >/dev/null 2>&1; then
        existing="$dest"
    fi

    local preserved=""
    if [[ -n "$PRESERVE_DIR" && -f "$PRESERVE_DIR$dest" ]] \
       && ! diff -q "$PRESERVE_DIR$dest" "$src" >/dev/null 2>&1; then
        preserved="$PRESERVE_DIR$dest"
    fi

    # Prefer the live file as the thing worth keeping: if both exist it is
    # the more recent statement of intent.
    local keep="${existing:-$preserved}"
    if [[ -n "$keep" ]]; then
        run install -m "$mode" -o "$u" -g "$g" "$keep" "$dest.preserved"
        note "$dest differs from the repository version. The repository
        version is now installed and the previous one is at
        $dest.preserved. Diff them, and fold anything machine-specific
        back into the repository rather than editing in place."
        changed "$dest.preserved (previous version kept)"
    fi

    run install -m "$mode" -o "$u" -g "$g" "$src" "$dest"
    if [[ -n "$keep" ]]; then
        say "$dest (from repo, previous kept as .preserved)"
    else
        say "$dest (from repo)"
    fi
}

install_tracked_conf "$REPO_ROOT/config/hades.yaml" /etc/hades/hades.yaml 640 root:hades
install_tracked_conf "$REPO_ROOT/services/systemd/embedder.conf"  /etc/hades/embedder.conf  644 root:root
install_tracked_conf "$REPO_ROOT/services/systemd/extractor.conf" /etc/hades/extractor.conf 644 root:root
install_tracked_conf "$REPO_ROOT/services/systemd/trainer.conf"   /etc/hades/trainer.conf   644 root:root

# Secrets are never in the repository. They come from the archive, or
# they are stubbed for the operator to fill in.
restore_secret() {
    local name="$1" mode="$2" owner="$3"
    local dest="/etc/hades/$name"
    local preserved="$PRESERVE_DIR/etc/hades/$name"

    if [[ -n "$PRESERVE_DIR" && -f "$preserved" ]]; then
        run install -m "$mode" -o "${owner%:*}" -g "${owner#*:}" "$preserved" "$dest"
        say "$dest (restored from archive)"
        changed "$dest restored"
        return 0
    fi
    if [[ -f "$dest" ]]; then
        say "$dest (already present - left untouched)"
        return 0
    fi
    return 1
}

if ! restore_secret daemon.conf 640 root:hades; then
    if [[ $DRY_RUN -eq 1 ]]; then
        say "would write a template /etc/hades/daemon.conf"
    else
        cat > /etc/hades/daemon.conf <<'CONF'
# Sourced by hades-daemon.service via EnvironmentFile=.
#
# ARANGO_PASSWORD is the password of the dedicated `hades` ArangoDB user,
# not root's. Create that user first - see scripts/install/setup-arangodb-user.sh
# or the README step "Create the ArangoDB user".
ARANGO_PASSWORD=CHANGE_ME

# The daemon opens a connection at startup, so it needs a database to
# open. _system is always present and neutral. Dispatched commands
# override the target per request.
HADES_DATABASE=_system

ARANGO_RO_SOCKET=/run/arangodb3/arangodb.sock
ARANGO_RW_SOCKET=/run/arangodb3/arangodb.sock
HADES_EMBEDDER_SOCKET=http://localhost:8087/v1
CONF
        chown root:hades /etc/hades/daemon.conf
        chmod 640 /etc/hades/daemon.conf
    fi
    note "/etc/hades/daemon.conf was written as a TEMPLATE. It has
        ARANGO_PASSWORD=CHANGE_ME and the daemon will not authenticate
        until you set the real password."
    changed "/etc/hades/daemon.conf (template)"
fi

restore_secret mcp-tokens 640 root:hades \
    || say "/etc/hades/mcp-tokens absent (only needed for the LAN MCP endpoint)"

# ---------------------------------------------------------------------
step "7. systemd units"
# ---------------------------------------------------------------------
# The arangod drop-in makes the Unix socket group-connectable and stops
# the packaged unit waiting for a PID file that never arrives. Without
# it, ArangoDB comes up "running" and every HADES command fails to
# connect.
ARANGO_DROPIN=/etc/systemd/system/arangodb3.service.d/hades.conf
ARANGO_DROPIN_CHANGED=0
if ! diff -q "$REPO_ROOT/services/systemd/arangodb3-hades.conf" \
        "$ARANGO_DROPIN" >/dev/null 2>&1; then
    ARANGO_DROPIN_CHANGED=1
fi
run install -D -m 644 "$REPO_ROOT/services/systemd/arangodb3-hades.conf" \
    "$ARANGO_DROPIN"
if [[ $ARANGO_DROPIN_CHANGED -eq 1 ]]; then
    say "arangodb3.service.d/hades.conf (changed)"
    changed "$ARANGO_DROPIN"
else
    say "arangodb3.service.d/hades.conf (already current)"
fi

# Older machines carry three hand-written drop-ins that this file
# supersedes. Leaving them in place means two sources of truth.
for legacy in override.conf pidfile-fix.conf socket-permissions.conf; do
    legacy_path="/etc/systemd/system/arangodb3.service.d/$legacy"
    if [[ -f "$legacy_path" ]]; then
        run rm -f "$legacy_path"
        note "removed superseded drop-in $legacy (now covered by hades.conf)"
        changed "removed $legacy_path"
        ARANGO_DROPIN_CHANGED=1
    fi
done

# The daemon unit is machine-independent. The embedder and trainer units
# carry a path to services/, substituted at install time so they are not
# pinned to one workstation.
install_unit() {
    local name="$1"
    local src="$REPO_ROOT/services/systemd/$name"
    if [[ $DRY_RUN -eq 1 ]]; then
        say "would install $name (SERVICES_DIR=$SERVICES_DIR)"
        return
    fi
    sed "s|@SERVICES_DIR@|$SERVICES_DIR|g" "$src" \
        > "/etc/systemd/system/$name"
    chmod 644 "/etc/systemd/system/$name"
    say "$name"
}

install_unit hades-daemon.service
if [[ $WITH_SERVICES -eq 1 ]]; then
    install_unit hades-embedder.service
    install_unit hades-extractor.service
    install_unit hades-trainer.service
else
    say "embedder, extractor and trainer units skipped (--with-services to install)"
fi

# A unit whose SupplementaryGroups names a group that does not exist
# fails at exec time, with an error that does not obviously point at the
# group. Drop weaver-admin from the list until WeaverTools creates it.
for unit in hades-daemon.service hades-trainer.service; do
    dropin_dir="/etc/systemd/system/$unit.d"
    dropin="$dropin_dir/no-weaver-admin.conf"
    if [[ $HAVE_WEAVER_ADMIN -eq 0 ]]; then
        [[ "$unit" == "hades-trainer.service" && $WITH_SERVICES -eq 0 ]] && continue
        if [[ $DRY_RUN -eq 1 ]]; then
            say "would write $dropin"
        else
            mkdir -p "$dropin_dir"
            cat > "$dropin" <<'CONF'
# Generated by hades-install.sh.
#
# The shipped unit declares SupplementaryGroups=weaver-admin arangodb.
# systemd refuses to exec a unit naming a group that does not exist, and
# weaver-admin is created by WeaverTools, not by HADES. This drop-in
# clears the list and re-adds only arangodb, which is the membership
# HADES actually requires for the ArangoDB socket.
#
# Delete this file and restart the unit once weaver-admin exists.
[Service]
SupplementaryGroups=
SupplementaryGroups=arangodb
CONF
            chmod 644 "$dropin"
        fi
        say "$unit: weaver-admin dropped via $dropin"
        changed "$dropin"
    elif [[ -f "$dropin" ]]; then
        run rm -f "$dropin"
        say "$unit: weaver-admin available again, removed $dropin"
        changed "removed $dropin"
    fi
done

run systemctl daemon-reload

# ---------------------------------------------------------------------
step "8. Start"
# ---------------------------------------------------------------------
if [[ $NO_START -eq 1 ]]; then
    say "skipped (--no-start)"
else
    if systemctl is-enabled --quiet arangodb3.service 2>/dev/null; then
        # A drop-in takes effect on restart, not on daemon-reload. Only
        # bounce the database when something it depends on actually
        # changed - a converge run on a healthy machine should not
        # interrupt service.
        if [[ $ARANGO_DROPIN_CHANGED -eq 1 ]]; then
            run systemctl restart arangodb3.service
            say "arangodb3 restarted (drop-in changed)"
        elif ! systemctl is-active --quiet arangodb3.service; then
            run systemctl start arangodb3.service
            say "arangodb3 started"
        else
            say "arangodb3 left running (drop-in unchanged)"
        fi
    else
        note "arangodb3.service is not enabled - enable it before the daemon
        will be able to connect: systemctl enable --now arangodb3.service"
    fi

    # `enable --now` is a no-op against an already-running unit, so an
    # updated unit file or a new binary would not take effect. Restart
    # explicitly when it is already up.
    if systemctl is-active --quiet hades-daemon.service; then
        run systemctl enable hades-daemon.service
        run systemctl restart hades-daemon.service
        say "hades-daemon restarted (picks up the new unit and binary)"
    else
        run systemctl enable --now hades-daemon.service
        say "hades-daemon enabled and started"
    fi

    if [[ $WITH_SERVICES -eq 1 ]]; then
        note "embedder, extractor and trainer units are installed but not
        started. They need a populated Python venv at $SERVICES_DIR/.venv
        and, for the embedder and trainer, a GPU. Start them by hand once
        the venv is rebuilt."
    fi
fi

# ---------------------------------------------------------------------
step "9. Verify"
# ---------------------------------------------------------------------
VERIFY_FAILED=0
check() {
    local label="$1"; shift
    if "$@" >/dev/null 2>&1; then
        printf '  ok    %s\n' "$label"
    else
        printf '  FAIL  %s\n' "$label"
        VERIFY_FAILED=1
    fi
}

if [[ $DRY_RUN -eq 1 ]]; then
    say "skipped in dry-run mode"
else
    check "group hades exists"            getent group hades
    check "/run/hades exists"             test -d /run/hades
    check "hades binary runs"             /usr/local/bin/hades --version
    check "arangodb socket present"       test -S /run/arangodb3/arangodb.sock

    if [[ $NO_START -eq 0 ]]; then
        # The daemon restarts on failure, so give it a moment to settle
        # into a state worth reporting.
        sleep 2
        check "hades-daemon active"       systemctl is-active --quiet hades-daemon.service
        check "daemon socket present"     test -S /run/hades/hades.sock
    fi

    # End-to-end: this is the check that proves the surviving database is
    # readable by the restored identity. `db stats` opens its own
    # connection rather than proxying through the daemon, so it needs the
    # password in the environment.
    if [[ -r /etc/hades/daemon.conf ]]; then
        db_pw=$(grep -E '^ARANGO_PASSWORD=' /etc/hades/daemon.conf | cut -d= -f2- || true)
        if [[ -z "$db_pw" || "$db_pw" == "CHANGE_ME" ]]; then
            printf '  skip  db stats (daemon.conf has no real password yet)\n'
        else
            if ARANGO_PASSWORD="$db_pw" /usr/local/bin/hades --database _system db stats >/dev/null 2>&1; then
                printf '  ok    db stats against _system\n'
            else
                printf '  FAIL  db stats against _system\n'
                VERIFY_FAILED=1
            fi
        fi
    fi
fi

# ---------------------------------------------------------------------
step "Summary"
# ---------------------------------------------------------------------
if [[ ${#CHANGES[@]} -eq 0 ]]; then
    say "no changes"
else
    say "changed:"
    printf '    - %s\n' "${CHANGES[@]}"
fi

if [[ ${#NOTES[@]} -gt 0 ]]; then
    echo
    say "follow-ups:"
    printf '    - %s\n' "${NOTES[@]}"
fi

echo
if [[ $VERIFY_FAILED -eq 1 ]]; then
    echo "Verification reported failures. Check: journalctl -u hades-daemon -n 50"
    exit 1
fi
[[ $DRY_RUN -eq 1 ]] && echo "Dry run complete - nothing was changed." || echo "Install complete."
