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
echo "=== Step 1: Verify the build artifact ==="
# A real VPS would `cargo build --release` here. The container deliberately
# has no Rust toolchain (it tests "what a fresh-VPS install gives you"), so
# we mount the host's prebuilt binary at /opt/hades-source/target/release.
test -x /opt/hades-source/target/release/hades || {
    echo "ERROR: no release binary at /opt/hades-source/target/release/hades"
    echo "On the host: cargo build --release, then mount target/release into the container."
    exit 1
}
echo "binary present: $(ls -l /opt/hades-source/target/release/hades)"

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
echo "=== Step 3: Install systemd users + tmpfiles ==="
# Creates the Unix `hades` user/group and the /run/hades runtime directory.
# Must run BEFORE step 5 (install config) because the config file is owned
# by group `hades`. This is the step the README previously had out of order.
install -m 644 services/systemd/hades-sysusers.conf /etc/sysusers.d/hades.conf
install -m 644 services/systemd/hades-tmpfiles.conf /etc/tmpfiles.d/hades.conf
systemd-sysusers
systemd-tmpfiles --create
getent group hades >/dev/null || {
    echo "ERROR: systemd-sysusers ran but the 'hades' group still doesn't exist."
    echo "Check /etc/sysusers.d/hades.conf and re-run \`systemd-sysusers\`."
    exit 1
}
echo "hades group: $(getent group hades)"

echo
echo "=== Step 4: Install the binary ==="
install -m 755 /opt/hades-source/target/release/hades /usr/local/bin/hades
mkdir -p /root/.local/bin
install -m 755 /opt/hades-source/target/release/hades /root/.local/bin/hades
/usr/local/bin/hades --version

echo
echo "=== Step 5: Install the system config ==="
# Requires the hades group to exist. Step 3 (sysusers) creates it; if that
# step didn't run or didn't succeed, fail fast with a clear message rather
# than silently falling back to a different group.
mkdir -p /etc/hades
if ! getent group hades >/dev/null 2>&1; then
    echo "ERROR: cannot install /etc/hades/hades.yaml — group 'hades' is missing."
    echo "Run step 3 (systemd-sysusers /etc/sysusers.d/hades.conf) before this step."
    exit 1
fi
install -m 640 -o root -g hades config/hades.yaml /etc/hades/hades.yaml
ls -la /etc/hades/hades.yaml

echo
echo "=== Step 6: Set the daemon environment ==="
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
echo "=== Step 7: Start the daemon (manual fallback — no systemd in container) ==="
# In the README this is `systemctl enable --now hades-daemon.service`, which
# uses EnvironmentFile=/etc/hades/daemon.conf + ExecStart=/usr/local/bin/hades.
# We can't run systemd in a vanilla container, so we source the env file and
# launch the daemon manually. This exercises the same code path the systemd
# unit would; what it doesn't exercise is the unit file itself (Phase 2 of #96).
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
echo "=== Step 8: Verify daemon is reachable ==="
hades --db _system db stats 2>&1 | head -10

echo
echo "================================================================"
echo "  All install steps completed successfully."
echo "================================================================"
