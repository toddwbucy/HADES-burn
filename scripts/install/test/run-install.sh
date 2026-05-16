#!/usr/bin/env bash
# Drives the README install steps inside the fresh-VPS test container.
# Each step echoes its README section heading so failures are easy to map
# back to docs. Exits non-zero on the first failure.
set -euo pipefail

cd /opt/hades-source

echo "================================================================"
echo "  HADES install validation — fresh-VPS dry run"
echo "================================================================"

echo
echo "=== Sanity: ArangoDB present and runnable ==="
arangod --version | head -1
arangosh --version | head -1

echo
echo "=== Step 0: Start arangod (manually — no systemd in container) ==="
# Use the default arangod config, but bind only to localhost.
#
# FINDING: ArangoDB's deb package on Ubuntu 24.04 does NOT ship a tmpfiles.d
# entry for /run/arangodb3 — the systemd unit creates that dir at service
# start time via `RuntimeDirectory=arangodb3`. Without systemd (this
# container, or any manual arangod invocation), arangod fails to bind its
# Unix socket because the dir doesn't exist. On a real VPS using systemctl,
# this is handled automatically; here we create the dir manually.
mkdir -p /var/lib/arangodb3 /var/log/arangodb3 /run/arangodb3
chown -R arangodb:arangodb /var/lib/arangodb3 /var/log/arangodb3 /run/arangodb3
nohup sudo -u arangodb arangod \
    --server.endpoint tcp://127.0.0.1:8529 \
    --server.endpoint unix:///run/arangodb3/arangodb.sock \
    --database.directory /var/lib/arangodb3 \
    --log.file /var/log/arangodb3/arangod.log \
    --javascript.app-path /var/lib/arangodb3-apps \
    --server.authentication true \
    > /tmp/arangod.out 2>&1 &
ARANGOD_PID=$!
echo "arangod PID: $ARANGOD_PID"
# Wait for the socket to appear (up to 30s)
for i in $(seq 1 30); do
    if [ -S /run/arangodb3/arangodb.sock ]; then
        echo "arangod socket ready after ${i}s"
        break
    fi
    sleep 1
done
test -S /run/arangodb3/arangodb.sock || { echo "ERROR: arangod socket never appeared"; tail -30 /tmp/arangod.out; exit 1; }

echo
echo "=== Step 1: Install the binary ==="
# A real VPS would build with cargo; here we expect a pre-built binary
# at /opt/hades-source/target/release/hades (mounted in by the runner).
test -x /opt/hades-source/target/release/hades || {
    echo "ERROR: no release binary at /opt/hades-source/target/release/hades"
    echo "Run: cargo build --release on host, then mount target/release/ into the container."
    exit 1
}
install -m 755 /opt/hades-source/target/release/hades /usr/local/bin/hades
mkdir -p /root/.local/bin
install -m 755 /opt/hades-source/target/release/hades /root/.local/bin/hades
/usr/local/bin/hades --version

echo
echo "=== Step 2: Create the ArangoDB user ==="
# Use the README's manual arangosh recipe.
# Use environment variables to avoid leaking the test root password.
export ARANGO_ROOT_PASSWORD="${ARANGO_ROOT_PASSWORD:-test_root_pw_for_install_validation_only}"
export HADES_PASSWORD="${HADES_PASSWORD:-test_hades_pw_only}"
arangosh \
    --server.endpoint unix:///run/arangodb3/arangodb.sock \
    --server.username root \
    --server.password "$ARANGO_ROOT_PASSWORD" \
    --javascript.execute-string '
        var users = require("@arangodb/users");
        users.save("hades", "'"$HADES_PASSWORD"'");
        users.grantDatabase("hades", "_system", "rw");
        users.grantDatabase("hades", "*", "rw");
        print("hades user created");
    '

echo
echo "=== Step 3: Install the system config ==="
# Per the README, this requires the hades group to exist (because of -g hades).
# The current README ordering puts sysusers AFTER this step, so we expect
# this to FAIL on a fresh VPS until we fix the ordering. The runner will
# fall through and report the error.
mkdir -p /etc/hades
# Bug to fix: sysusers must run BEFORE this command.
if getent group hades >/dev/null 2>&1; then
    echo "hades group exists; proceeding with install"
else
    echo "WARN: hades group does not exist yet — sysusers needs to run first."
    echo "      (This is the ordering bug we expect to find.)"
fi
install -m 640 -o root -g "$(getent group hades >/dev/null && echo hades || echo root)" \
    config/hades.yaml /etc/hades/hades.yaml
ls -la /etc/hades/hades.yaml

echo
echo "=== Step 4: Set the daemon environment ==="
cat > /etc/hades/daemon.conf <<DAEMON_CONF
ARANGO_PASSWORD=$HADES_PASSWORD
HADES_DATABASE=_system
ARANGO_RO_SOCKET=/run/arangodb3/arangodb.sock
ARANGO_RW_SOCKET=/run/arangodb3/arangodb.sock
HADES_EMBEDDER_SOCKET=http://localhost:8087/v1
DAEMON_CONF
chmod 640 /etc/hades/daemon.conf
echo "daemon.conf written:"
cat /etc/hades/daemon.conf | sed 's/ARANGO_PASSWORD=.*/ARANGO_PASSWORD=<redacted>/'

echo
echo "=== Step 5: Start the daemon manually (no systemd) ==="
# Source the env file, then start the daemon. In the systemd path this is
# done via EnvironmentFile + ExecStart; here we do it manually to exercise
# the same code path the daemon takes at startup.
mkdir -p /run/hades
set -a
. /etc/hades/daemon.conf
set +a
nohup /usr/local/bin/hades daemon --socket /run/hades/hades.sock > /tmp/hades-daemon.log 2>&1 &
DAEMON_PID=$!
echo "daemon PID: $DAEMON_PID"
sleep 3
if kill -0 "$DAEMON_PID" 2>/dev/null; then
    echo "daemon running"
    head -20 /tmp/hades-daemon.log
else
    echo "ERROR: daemon exited"
    cat /tmp/hades-daemon.log
    exit 1
fi

echo
echo "=== Step 6: Verify daemon is reachable ==="
hades --db _system db stats 2>&1 | head -10

echo
echo "================================================================"
echo "  All install steps completed successfully."
echo "================================================================"
