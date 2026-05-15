#!/usr/bin/env bash
#
# Bootstrap the `hades` ArangoDB user.
#
# HADES is DBA tooling and connects to ArangoDB as a named user (`hades`)
# rather than `root`. ACL-level restrictions (read-only on production
# databases, etc.) are managed by the operator in arangosh — this script
# only creates the user and gives it admin-tool-shaped default grants
# (rw on _system, rw default on new databases). The operator then tightens
# specific databases via arangosh as needed.
#
# Idempotent: re-running is safe. If the user already exists, grants are
# left untouched so operator-managed ACLs survive re-runs.
#
# Usage:
#   ARANGO_ROOT_PASSWORD=<root-pw> HADES_PASSWORD=<new-pw> \
#     scripts/install/setup-arangodb-user.sh
#
# Status:
#   Sketched, not yet tested on a clean machine. Test target is a VPS —
#   do not rely on this script for first-time setup on the development
#   workstation; do user creation manually there. See README "Manual
#   ArangoDB user setup" for the equivalent curl calls.

set -euo pipefail

: "${ARANGO_ROOT_PASSWORD:?set the current ArangoDB root password}"
: "${HADES_PASSWORD:?set the password to assign to the new hades user}"

SOCK="${ARANGO_SOCKET:-/run/arangodb3/arangodb.sock}"

if [[ ! -S "$SOCK" ]]; then
  echo "ArangoDB socket not found at $SOCK" >&2
  echo "Override with ARANGO_SOCKET=/path/to/arangodb.sock if non-default." >&2
  exit 1
fi

api() {
  curl -fsS --unix-socket "$SOCK" \
    -u "root:${ARANGO_ROOT_PASSWORD}" \
    -H 'Content-Type: application/json' "$@"
}

# 1. Create the user. 201 = created, 409 = already exists (both fine).
http=$(api -o /tmp/hades-bootstrap.out -w '%{http_code}' \
  -X POST http://localhost/_api/user \
  -d "{\"user\":\"hades\",\"passwd\":\"${HADES_PASSWORD}\",\"active\":true}" \
  || true)

case "$http" in
  201)
    echo "Created ArangoDB user 'hades'. Setting initial grants…"

    # rw on _system so HADES can create/drop databases.
    api -X PUT http://localhost/_api/user/hades/database/_system \
      -d '{"grant":"rw"}' >/dev/null

    # Default grant for newly-created databases. Wildcard '*' is URL-encoded.
    # Operator tightens specific databases (e.g. read-only production data
    # → 'ro') in arangosh after this runs.
    api -X PUT "http://localhost/_api/user/hades/database/%2A" \
      -d '{"grant":"rw"}' >/dev/null

    echo "Initial grants applied:"
    echo "  _system: rw"
    echo "  * (default for new databases): rw"
    echo
    echo "Restrict specific databases via arangosh, e.g.:"
    echo "  require('@arangodb/users').grantDatabase('hades', '<production-db-name>', 'ro')"
    ;;
  409)
    echo "ArangoDB user 'hades' already exists. Leaving grants untouched."
    echo "If you need to reset the password, do it in arangosh:"
    echo "  require('@arangodb/users').replace('hades', '<new-password>')"
    ;;
  *)
    echo "Unexpected HTTP $http from ArangoDB:" >&2
    cat /tmp/hades-bootstrap.out >&2 || true
    exit 1
    ;;
esac
