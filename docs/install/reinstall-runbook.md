# OS reinstall runbook

How to move a HADES installation onto a freshly installed operating
system when the data lives on ZFS pools that are exported and re-imported
rather than restored from backup.

This is not the first-time install guide. For that, see the Install
section of the README, or run `scripts/install/hades-install.sh` on a
clean machine. This document covers the case where the operating system
is replaced underneath a database that survives.

## The shape of the problem

The instinct is to treat this as "install fresh, then restore the data".
That is the wrong model here, and following it wastes a day.

The ArangoDB data directory is a ZFS dataset. Nothing about the wipe
touches it. Every database, the `hades` ArangoDB user, and every
per-database ACL grant comes back exactly as it was, because all of that
is stored inside the data directory rather than in the operating system.

What the wipe destroys is the host identity that makes the surviving data
usable:

* The secrets in `/etc/hades`, including the password the daemon uses to
  authenticate as the `hades` ArangoDB user. Lose that and you have a
  database you cannot log into.
* The systemd units, including a drop-in for `arangodb3.service` without
  which the Unix socket comes back inaccessible.
* The contents of `/etc/passwd` and `/etc/group`. ZFS stores file owners
  as numbers. A rebuilt machine that assigns a different UID to the
  `arangodb` account has a service that cannot read its own data.

So the work is restoring identity, not restoring data. Plan accordingly.

## Inventory

Confirm this against your own machine before starting, with
`zfs list -o name,mountpoint` and `df --output=source,fstype <path>`.
The layout below is what `hades-preserve.sh` recorded on the reference
workstation.

### Survives the wipe

| Dataset | Mountpoint | What is in it |
|---|---|---|
| `dbpool/arangodb` | `/var/lib/arangodb3` | Every database, all users, all ACL grants |
| `dbpool/olympus` | `/home/todd/olympus` | This repository, `services/.venv` |
| `fastpool/var-cache` | `/var/cache` | `/var/cache/hades/huggingface` model weights |
| `fastpool/weaver-huggingface` | `/opt/weaver/huggingface` | Shared HF cache, current `HF_HOME` |
| `bulk-store/backups` | `/bulk-store/backups` | Where the preserve archive should go |

### Does not survive

Everything on the root filesystem. For HADES that means:

| Path | Why it matters |
|---|---|
| `/etc/hades/daemon.conf` | Holds `ARANGO_PASSWORD` for the `hades` user |
| `/etc/hades/mcp-tokens` | LAN MCP endpoint bearer tokens |
| `/etc/hades/hades.yaml` | Recoverable from the repository |
| `/etc/hades/{embedder,extractor,trainer}.conf` | Recoverable from the repository, but check for local edits |
| `/etc/systemd/system/hades-*.service` | Recoverable from the repository |
| `/etc/systemd/system/arangodb3.service.d/` | Now tracked as `services/systemd/arangodb3-hades.conf` |
| `/etc/passwd`, `/etc/group` | The UID and GID numbers the ZFS files are owned by |
| `/usr/local/bin/hades`, `~/.local/bin/hades` | Rebuilt from source |

Note that `/home/todd` itself is on the root filesystem. Only
`/home/todd/olympus` and `/home/todd/.cache` are datasets. Anything else
under the home directory, including shell configuration, SSH keys and
`~/.claude`, is outside the scope of this runbook and needs its own
backup.

## Before the wipe

### 1. Capture host state

```bash
sudo scripts/install/hades-preserve.sh --dry-run    # inspect first
sudo ARANGO_ROOT_PASSWORD='<root-pw>' \
     scripts/install/hades-preserve.sh
```

`ARANGO_ROOT_PASSWORD` is optional. Supplying it adds a dump of ArangoDB
users and grants, which turns post-reinstall verification into a diff
rather than a memory test.

The archive lands in `/bulk-store/backups/hades-preserve/` with a
`latest.tar.gz` symlink. It contains credentials and is written mode
0600. The script prints the filesystem type of the output directory as
its last action. Read that line. If it does not say `zfs`, you are about
to preserve your secrets onto the disk you are about to erase.

### 2. Commit outstanding work

Anything uncommitted in a repository on a surviving pool is fine. Anything
uncommitted anywhere else is about to be deleted. Push what matters.

### 3. Record the pool layout

```bash
zpool status > /bulk-store/backups/zpool-status.txt
zfs list -o name,used,mountpoint > /bulk-store/backups/zfs-layout.txt
```

The preserve archive already contains this in `manifest.txt`. Keeping a
copy outside the archive means you can read it from a rescue shell
without extracting a tarball.

### 4. Export the pools

```bash
sudo systemctl stop hades-daemon hades-embedder hades-extractor hades-trainer
sudo systemctl stop arangodb3
sudo zpool export bulk-store
sudo zpool export dbpool
sudo zpool export fastpool
```

Exporting cleanly is what lets the new system import without `-f`. If the
installer or a stray process holds a dataset open, find it with
`fuser -vm <mountpoint>` rather than reaching for the force flag.

## During the OS install

Two things to get right:

**Do not let the installer touch the pool disks.** Partition and format
only the root device. An installer that helpfully offers to use all
available storage will destroy the pools.

**Do not create the `arangodb` account yet.** It has to be created with a
specific UID, before the ArangoDB package is installed. See below.

## After the OS install

### 5. Import the pools

```bash
sudo apt-get install -y zfsutils-linux
sudo zpool import bulk-store
sudo zpool import dbpool
sudo zpool import fastpool
zfs list -o name,mountpoint
```

Compare the mountpoints against what you recorded. Mountpoints are stored
as dataset properties, so they should come back on their own. If a
dataset mounts somewhere unexpected, fix it with
`zfs set mountpoint=<path> <dataset>` before continuing.

`dbpool/olympus` mounts at `/home/todd/olympus`, so the parent directory
has to exist and belong to the right user first. Create the human account
with its original UID before importing, or the repository comes back
owned by a stranger.

### 6. Re-create service accounts with their original numeric IDs

This is the step that makes the difference between a twenty minute
reinstall and a long afternoon. Read the numbers out of the preserve
archive:

```bash
ARCHIVE=/bulk-store/backups/hades-preserve/latest.tar.gz
tar xzOf "$ARCHIVE" ./identity.env          # read it
source <(tar xzOf "$ARCHIVE" ./identity.env)  # then load it into the shell
echo "arangodb should be ${ARANGODB_UID}:${ARANGODB_GID}, datadir is ${DATADIR_UID}:${DATADIR_GID}"
```

Those two numbers should agree. If they do not, trust `DATADIR_UID`: it
is what the surviving files are actually owned by.

Create the `arangodb` user and group with those exact IDs **before**
installing the ArangoDB package. The Debian package allocates a system
UID dynamically when the account does not already exist, and whatever it
picks will not match the 78 GB of data sitting on `dbpool/arangodb`:

```bash
sudo groupadd -g "$ARANGODB_GID" -r arangodb
sudo useradd -u "$ARANGODB_UID" -g arangodb -r \
     -d /usr/share/arangodb3 -s /bin/false arangodb
```

If the package is already installed and the account already exists with
the wrong number, `hades-install.sh` will detect the mismatch and offer
two remedies. Recovering is possible either way, but pre-seeding is
faster and touches nothing.

The `hades` account needs no special handling. `hades-sysusers.conf` pins
it to UID and GID 985, so it is deterministic across installs, and
`hades-install.sh` creates it.

### 7. Install prerequisites

ArangoDB, per the README Prerequisites section. Then the build
toolchain:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. ~/.cargo/env
sudo apt-get install -y protobuf-compiler build-essential
```

Optional, and only needed for the ingest paths that use them:
`rust-analyzer` and `gopls` for higher-fidelity Rust and Go code ingest,
and `libclang` for the C, C++ and CUDA analyzers. Check what the binary
can see afterwards with `hades tools status --workspace <path>`.

GPU drivers and the CUDA runtime are a separate concern with their own
ordering constraints. The embedder and trainer services need them. The
daemon and the CLI do not, so do not block the HADES install on the GPU
stack.

### 8. Build and install

The repository came back with the pool, so there is nothing to clone:

```bash
cd /home/todd/olympus/HADES-Burn
cargo build --release

sudo scripts/install/hades-install.sh \
     --from-preserve /bulk-store/backups/hades-preserve/latest.tar.gz \
     --dry-run
```

Read the dry run output, then run it for real without `--dry-run`.

If the ArangoDB account UID could not be pre-seeded in step 6, add
`--arango-uid-fix adopt` to move the account onto the data directory's
numbers. Use `--arango-uid-fix chown` only if something else on the
system already depends on the new UID, because it rewrites metadata for
every file in the store.

The installer restores secrets from the archive, installs non-secret
configuration from the repository, and writes any preserved copy that
differs as a `.preserved` file next to it. Diff those before deleting
them. On the reference workstation the live `embedder.conf` pointed
`HF_HOME` at the shared Weaver cache while the repository default points
at `/var/cache/hades/huggingface`. That is a real deployment choice, not
drift to be discarded.

### 9. Rebuild the Python services venv

`services/.venv` survives on the pool, but it is not portable. Its
interpreter is a symlink to `/usr/bin/python3`, so a new OS with a
different Python minor version leaves a venv that imports nothing. Do not
try to repair it:

```bash
cd /home/todd/olympus/HADES-Burn/services
rm -rf .venv
python3 -m venv .venv
.venv/bin/pip install -e .
make proto-gen
```

Then install the service units, which are not installed by default:

```bash
sudo ../scripts/install/hades-install.sh --with-services --no-start
make install-embedder
make install-trainer
```

The model weights themselves are on a surviving pool, so the first
embedder start does not re-download several gigabytes from HuggingFace.
Confirm `HF_HOME` in `/etc/hades/embedder.conf` points at whichever cache
actually survived.

## Verification

`hades-install.sh` runs these at the end and prints a pass or fail line
for each. Run them by hand if you need to check state later.

```bash
# Identity restored
getent group hades && id hades
stat -c '%u:%g' /var/lib/arangodb3 && id -u arangodb

# Transport up
test -S /run/arangodb3/arangodb.sock && ls -l /run/arangodb3/arangodb.sock
systemctl is-active arangodb3 hades-daemon
test -S /run/hades/hades.sock

# The real test: surviving data, readable through restored identity
export ARANGO_PASSWORD='<hades user password>'
hades --database _system db stats
```

The socket should be mode `770`. If it is not, the arangod drop-in did
not apply. A drop-in takes effect on restart, not on `daemon-reload`.

Then confirm the databases and their grants actually came back, rather
than assuming they did because the pool imported:

```bash
hades --database _system db list
```

Compare against `arango-users.json` in the preserve archive if you
recorded it. A production database that has come back with `rw` where it
previously had `ro` is a silent hazard, because the ACL layer is the only
write gate HADES has.

## Known failure modes

**The daemon fails to start with a group-related error.** systemd refuses
to exec a unit whose `SupplementaryGroups` names a group that does not
exist. The shipped units name `weaver-admin`, which belongs to
WeaverTools and will not exist on a fresh machine. `hades-install.sh`
detects this and writes a drop-in that clears the entry. Once WeaverTools
is installed, re-run the installer and the drop-in is removed
automatically.

**Every command reports a connection failure although arangod is
running.** Check the socket mode. The packaged unit leaves it
inaccessible to the `arangodb` group, and the drop-in that fixes it is
easy to forget on a manual install.

**arangod will not start after a pool import.** Almost always the UID
mismatch from step 6. Compare `stat -c %u /var/lib/arangodb3` against
`id -u arangodb`.

**The daemon starts but nothing authenticates.** Check whether
`/etc/hades/daemon.conf` is the template. If `hades-install.sh` ran
without `--from-preserve`, it writes `ARANGO_PASSWORD=CHANGE_ME` and says
so in its follow-ups.

**The embedder starts and then fails on first model load.** `HF_HOME`
points outside `ReadWritePaths`, so `ProtectSystem=strict` makes the
cache read-only and the lock file cannot be created. Both settings are in
`hades-embedder.service` and `/etc/hades/embedder.conf` and have to agree.

## After a successful run

Fold what you learned back into this document and into
`hades-install.sh`. The reinstall is the only real test either of them
gets, and the next one will be far enough away that nothing will be
remembered. In particular, record any step that needed manual
intervention: that is a gap in the installer, not a fact about the
machine.
