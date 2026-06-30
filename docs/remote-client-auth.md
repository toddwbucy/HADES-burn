# HADES Remote Client Authentication and Transport

**Status:** Draft, awaiting ratification
**Version:** 0.3
**Date:** 2026-06-30
**Relates to:** [daemon-protocol.md](daemon-protocol.md) (the wire protocol this carries over the network), [model-operation-vocabulary.md](model-operation-vocabulary.md) (the closed command surface)

**v0.3 change:** database creation, deletion, and all `_system` administration are
local-only. No provisioning broker, and no network path to `_system` Administrate
for any principal. A network principal may hold full database-level administration
of databases it owns, granted at provisioning time, and nothing on any other.

## Why this exists

Weaver agents run on a separate machine (a laptop) and need access to the single
HADES server over a network. The naive option, running a second HADES instance on
the laptop, was rejected: two databases to keep in sync and two sources of truth.
There is one HADES server. Remote callers reach it over an authenticated channel.

Three access patterns drive the design:

1. **Agent compute on the laptop, agent memory on the HADES node, one hop
   between.** The agent's memory is a database on the HADES server. This path is
   read-write to that one database, not the read-only profile an early draft
   assumed.
2. **The human operator reads and evaluates across databases** remotely, with no
   writes and no ability to administer the server over the wire.
3. **The human operator creates and fully administers project databases**, but
   only databases that principal owns, and with all `_system`-level administration
   (creating or dropping databases, managing users and grants) staying local.

This is **per-principal** access, not per-machine. Each Weaver agent runs as its
own Linux user (the Weaver sandbox model). Agent A holding access on a laptop
does not grant agent C on that same laptop any access. And each enrolled key is
pinned to a fixed scope (one database for a Weaver agent, or the set of databases
the human builder owns) with an ArangoDB grant scoped to match.

This is NOT an MCP server. It reuses the existing daemon protocol verbatim and
carries it over SSH.

## Non-goals

- Not a second HADES instance and not a database copy. One server, one graph set.
- Not box authentication. Two agents on the same laptop are distinct principals.
- Not an MCP server. The daemon protocol is the interface.
- Not a path to server administration. No network identity holds `_system`-level
  rights, can create or drop a database, or can manage ArangoDB users or grants
  over the wire. Those are local-only (see "Admin and provisioning are local"). A
  principal may hold full database-level administration of databases it owns, and
  nothing on any other.
- Not a self-service database broker. There is no network-reachable path to
  `_system` Administrate. Databases are created by a local provisioning step
  (Weaver for agent memory databases, local SSH for the human's project
  databases), never by a command over the wire.
- HADES does not provision Weaver agent users, does not place their keys, and does
  not create their databases. HADES owns the server side of access (enrollment,
  transport, the daemon, the scoped ArangoDB users) and serves scoped access to
  databases that already exist. See "Ownership boundary".

## The actors

| Principal | What it is | Memory path | Where it runs |
|-----------|------------|-------------|---------------|
| HADES server | The single daemon plus ArangoDB plus the embedder | n/a | The workstation/server |
| Weaver agent | An AI coding agent, one dedicated Linux UID each | read-write to one pinned database | The client machine (laptop) |
| Operator (evaluator) | The human, reading and evaluating across databases | read-only across databases | The client machine |
| Operator (builder) | The human, creating and administering own project databases | full DBA on owned databases, none on others | The client machine |
| Operator (admin) | The human administering the server itself | full, local only | On the server, after SSH and elevation |

## Architecture

The daemon protocol is transport-agnostic at the framing layer (length-prefixed
JSON). Today it runs over a local Unix socket. For remote access it runs over an
SSH channel. The daemon gains a stdio mode that speaks the same frames over
stdin and stdout, and SSH carries that stdio to and from the client.

```
client machine (laptop)                         HADES server
+-----------------------------+                 +---------------------------------+
| weaver agent (uid 1001)     |                 | sshd                            |
|   hades-client binary       |   ssh, key A    |   command= per key:             |
|     uses key A from agent A  | --------------> |   hades daemon-stdio            |
|     store, via ssh-agent     |  (JSON frames   |     --session agent             |
|                              |   over stdio)   |     --client-id agent-a         |
+-----------------------------+                 |     --database agent_a_mem      |
                                                |        |                        |
                                                |        v  connects to Arango as |
                                                |   hades_agent_a (scoped user)   |
                                                |        |                        |
                                                |        v  local unix socket     |
                                                |   ArangoDB  +  embedder         |
                                                +---------------------------------+
```

The client runs the equivalent of `ssh hades-server hades daemon-stdio`. The
daemon's existing `handle_connection` logic is unchanged. The transport in front
of it and the ArangoDB user behind it are what differ per principal.

Note on terminology: the per-key restriction is the `command="..."` option inside
`authorized_keys`, which can vary per key. It is not the global `ForceCommand`
directive in `sshd_config`, which cannot carry a per-key `--client-id` or
`--database`. Implementers must use the per-key `authorized_keys` form.

## Principals and grants

Flexibility lives in the per-key ArangoDB user, not in a fixed set of daemon
tiers. Different principals have different needs, expressed as different ArangoDB
grants. The daemon does not enumerate every capability combination. It defers
fine-grained authorization to ArangoDB, which is the authoritative gate (Critical
Rule #1).

| Principal | Key location | Daemon session | ArangoDB user | ArangoDB grant |
|-----------|--------------|----------------|---------------|----------------|
| Agent A | per-UID key on laptop | `agent` | `hades_agent_a` | Access on `agent_a_mem`, collection RW on its memory collections, RO on protected ones, none elsewhere |
| Agent B | per-UID key on laptop | `agent` | `hades_agent_b` | same shape, pinned to `agent_b_mem` |
| Operator (evaluator) | operator key | `agent` | `hades_ro_all` | Access plus collection RO across databases, Administrate on none |
| Operator (builder) | operator builder key | `agent` | `hades_build_todd` | database-level Administrate on owned databases (the `todd__*` set), none on every other database |
| Operator (admin) | local Unix socket | `admin` | the `hades` user | full, local only |

No network principal holds **server-level** (`_system`) Administrate, and none
holds any right on a database it does not own. Database-level Administrate is
granted only on owned databases. That is the load-bearing invariant of the whole
design (see "The no-server-Administrate invariant").

## The enrollment unit

Enrollment is `(principal, session, database, grant-spec)`. Every
privilege-bearing field is set by the operator on the server. None of them is a
client flag.

Operator inputs from the client side (Weaver):

- `public_key`: the agent's SSH public key. The private half never leaves the
  client.
- `label`: a client identity tag, for example `agent-a`. Not a Unix account.

Operator decisions (the client does not choose these):

- `session`: the daemon session ceiling.
- `scope`: the database this key may reach. For a single-DB tenant it is one
  database name. For a builder it is the naming prefix that identifies the owned
  set (for example `todd__`).
- `grant-spec`: the ArangoDB grant for this key's scoped user. For a tenant, for
  example `{db: agent_a_mem, access: ro, collections: {memory: rw, audit: ro}}`.
  For a builder, database-level Administrate on the owned databases and `none`
  elsewhere.

Output, written on the server:

- One `authorized_keys` line under the shared service account, with `command=`
  pinning `--session`, `--client-id`, and the scope (`--database` or
  `--namespace`), plus `restrict`.
- One ArangoDB user, scoped per the grant-spec, that the daemon process for this
  key authenticates as.

```
command="hades daemon-stdio --session agent --client-id agent-a --database agent_a_mem",restrict ssh-ed25519 AAAA...agentApub  agent-a
command="hades daemon-stdio --session agent --client-id todd-build --namespace todd__",restrict ssh-ed25519 AAAA...toddpub  todd-build
```

The scope (`--database` or `--namespace`) joins `--session` and `--client-id` in
the set the server pins and the client cannot override. The daemon must reject or
ignore any client-supplied scope. This is the command-injection concern of G3
extended to the database scope. Note that a builder's scope still grants nothing
on databases outside its prefix, because the scoped ArangoDB user has no grant on
them regardless of what the client sends.

Because the private key stays on the client, the shipped package contains no
secret. It holds the binary, the pinned server host key, and the host and port.
It can travel over any channel.

## The capability model: ownership inside two hard floors

Over the wire a principal can do whatever its scoped ArangoDB user permits, inside
two floors that hold for every principal without exception.

The two hard floors, refused on the network transport and withheld from every
network ArangoDB user:

- **No `_system` operations.** No creating or dropping databases, no managing
  ArangoDB users or grants. These need `_system` Administrate, which no network
  user holds. They happen only through local provisioning.
- **No raw AQL.** AQL can write and is expensive. The closed operation vocabulary
  is the only command surface over the wire.

Inside those floors, the scoped ArangoDB user decides the rest:

| Principal shape | ArangoDB grant | What it can do over the wire |
|-----------------|----------------|------------------------------|
| Single-DB tenant (Weaver agent) | DB-level Access on one DB, collection RW/RO | reads, and document create/update/delete in granted collections of that one DB |
| Namespace owner (human builder) | DB-level Administrate on owned DBs, none elsewhere | all of the above, plus DDL inside owned DBs: create or drop collections, create indexes, graph create or drop or materialize, `db.purge` |
| Evaluator (human) | DB-level Access plus collection RO across DBs | read and evaluate, no writes |

The current daemon classifies DML (`db.insert`, `db.update`, `db.delete`) and DDL
(`db.create_collection`, graph create or drop, `db.schema.init`, `db.purge`)
together as Admin tier. This design separates three classes:

- **DML and DDL become network-reachable, gated by the scoped ArangoDB user.** A
  tenant's Access-level user cannot run DDL. A builder's Administrate-level user
  can, but only on its owned databases.
- **`_system` operations** (create or drop database, user or grant management) are
  refused on the network transport and impossible for any network user.
- **Raw AQL** is refused on the network transport.

Two gates agree on the floors: the transport refuses the `_system` and raw-AQL
command classes, and no network ArangoDB user could execute them. Production and
reserved databases are untouchable because no network user is granted any access
to them, so the scoped user cannot see them at all. There is no need for net-new
`memory.*` commands. Weaver layers memory semantics on top of generic scoped CRUD
on the client side, which keeps the ownership boundary intact.

### The no-server-Administrate invariant

No network ArangoDB identity holds **server-level** (`_system`) Administrate, and
none holds any right on a database it does not own. Database-level Administrate is
granted only on owned databases. Mapped to ArangoDB's access levels:

- **Server level (`_system`) Administrate** is never granted to a network user. It
  is the credential for creating or dropping databases and for managing users and
  grants. It stays local. This is the absolute floor that keeps production safe
  even against a daemon compromise, because there is no network path to it at all.
- **Database-level Administrate** on an owned database lets the owner do DDL inside
  it (collections, indexes, graphs, purge) with no reach outside it. Granted per
  owned database, `none` elsewhere.
- **Database-level Access** plus **collection Read/Write or Read-Only** gives a
  tenant document CRUD without DDL. Read-Only protects a container from document
  deletion. `none` hides it.

So "create and drop collections in my own database, but cannot create or drop the
database itself, cannot touch a database I do not own, and cannot administer the
server" maps onto `grantDatabase(Administrate)` on owned databases,
`grantDatabase(none)` elsewhere, and no `_system` grant. This is core ArangoDB
functionality. Confirm the grant levels behave this way on the deployed ArangoDB
edition before relying on them.

Production and reserved databases (`_system`, `bident_burn`, every research and
production database) are protected by the same mechanism: no network user is
granted any access to them, so they are invisible and untouchable over the wire.
No runtime denylist is needed in the daemon. The grant is the gate.

Every "cannot reach the server or another tenant" claim in this document depends
on this invariant. It is a precondition, not something daemon code provides.

## Two-layer security model

The cryptographic gate is the boundary. The launch guard is hygiene. They have
different strength and must not be confused.

### Layer 1: the cryptographic gate (the boundary)

- Each principal has its own SSH keypair. The private key lives only in that UID's
  store (`~/.ssh` at 0600 owned by the UID, or that UID's ssh-agent). The kernel
  prevents any other UID from reading it.
- The public key sits in the server's `authorized_keys` under a single shared,
  locked-down service account. Principals are told apart by key, not by separate
  Unix accounts on the server.
- The `command=` option forces one command and strips all other SSH capability
  via `restrict` (no shell, no forwarding, no pty, no X11, no agent forwarding).
- The daemon process for that key authenticates to ArangoDB as that key's scoped
  user. ArangoDB enforces what the key may touch.

Enforced by the kernel, by sshd, by crypto, and by ArangoDB. This is the
boundary.

### Layer 2: the launch guard (hygiene, not a boundary)

The client binary may self-check before doing anything: `geteuid()` matches the
intended UID, the owner of `/proc/self/exe` matches, `/etc/machine-id` matches.
These catch misconfiguration and raise the bar against casual misuse. They are
not a security boundary. Whoever controls the machine can patch out a self-check,
and because the binary carries no secret, bypassing the check yields nothing.

An OS-level companion is real enforcement of who may launch the file:
`chown agent-a hades-client && chmod 0500`. Only that UID can `exec` it. This
governs who may start the binary, not who the binary authenticates as.

## The per-agent invariant

> The private key is only ever in a per-UID location: an ssh-agent, or `~/.ssh`
> at 0600 owned by that agent UID. Never embedded in the binary. Never in a
> shared or world-readable path.

If this holds, per-agent isolation falls out automatically. The binary carries no
secret, so possessing it grants nothing. The key is readable only by its owning
UID, so one agent cannot use another agent's identity. The kernel does the
enforcing.

## Per-database isolation via the scoped ArangoDB user

This is the core tenancy mechanism, not an optional backstop.

The `daemon-stdio` process for a given key authenticates to ArangoDB as that
key's scoped user and **holds no other ArangoDB credential in memory**. Process-
level credential isolation is what makes this gate independent of the daemon's
own correctness. Agent A's process cannot reach agent B's memory or any research
database even if the daemon's session check has a bug or the client smuggles a
database override, because the ArangoDB user it runs as cannot see those
databases at all.

The scoped user mirrors the session's intended capability at the ArangoDB layer.
The daemon session and the ArangoDB grant encode the same capability
independently, so a bug in one is caught by the other. A shared, over-privileged
ArangoDB connection would defeat this: a daemon compromise would then reach
everything that connection can do. The per-key scoped user, held by a per-key
process, is the mitigation.

## Mutual authentication and instance binding

| Direction | Mechanism | Result |
|-----------|-----------|--------|
| Server authenticates the principal | public key in server `authorized_keys` | only enrolled principals connect, identified individually |
| Client authenticates the server | server host key pinned in the client bundle | the binary connects only to the instance it was built for |

The pinned server host key is the load-bearing thing baked into the per-client
package. A binary built for instance X cannot be repurposed against instance Y,
and a roaming laptop on a hostile network cannot be steered to an impostor. The
client must enforce the pin strictly: `StrictHostKeyChecking=yes` against only the
bundled `known_hosts`, with no TOFU and no `accept-new`. A client that prompts on
first connect has no instance binding.

## Tier gate reconciliation (cross-document, required at ratification)

Locally, the session tier was inspectability "as much as enforcement"
(README.md and daemon-protocol.md). Over the network that posture does not hold.
README.md today states:

> Security against malicious clients is not the goal here. That is enforced at the
> ArangoDB layer via ACL grants on the dedicated `hades` ArangoDB user.

Two claims are folded together there, and they pull apart over the network:

- Claim (a), "tier is documentation as much as enforcement", must be scoped to the
  local Unix-socket transport. Over the network the `command=` pins the session,
  but the authenticated client still sends arbitrary commands over the channel,
  and the daemon's rejection of an out-of-tier command is a security boundary.
  The tier-gating path is security-critical code. The entire agent-session command
  surface, not only the tier check, must be tested against hostile input,
  including the case "an agent session sends every admin, DDL, and raw-AQL
  command". A panic or logic bug in any reachable handler is now a remote concern.
- Claim (b), "the ArangoDB ACL is the real boundary", is the correct principle and
  is kept. It is honored only when each remote key runs as a scoped ArangoDB user.
  Sharing the full `hades` user would make claim (b) false for any database that
  user can write.

**Action required before ratification:** update README.md and daemon-protocol.md
so they scope claim (a) to the local transport and tie claim (b) to the scoped
per-key ArangoDB users. This document must not land while those two still assert
the old posture without qualification.

## Admin and provisioning are local

Two things never cross the wire: server administration, and database provisioning.
Both happen on the server.

- The ArangoDB endpoint and web interface are bound to `127.0.0.1` only, never
  `0.0.0.0`. They are never advertised on the network.
- Raw AQL, any `_system` operation, and any database create or drop are permitted
  only from a local ArangoDB user on the server.
- The human operator administers the server by first connecting over SSH, then
  elevating to the local admin context. There is no remote `admin` session over
  the wire.
- **Database provisioning is local and out of band.** Weaver creates agent memory
  databases at agent-birth through its own local provisioning flow (a
  WeaverTools-owned identity runs the create and grant on the server). This
  document does not specify that mechanism, since it is WeaverTools' domain. The
  human creates project databases by local SSH. In both cases the `_system`
  operation executes on the server, and the network principal only ever receives a
  scoped ArangoDB user on databases that already exist. See "Ownership boundary".

Localhost-only binding and out-of-band provisioning together are what keep
`_system` Administrate off the network entirely.

## Write consistency across the hop

SSH gives the transport half for free: confidentiality and per-packet integrity,
so nobody on the hop reads or tampers with memory in flight. That is done. The
part SSH does not give, and the part "maintain internal integrity" really
requires, is write consistency over an unreliable link.

A roaming laptop drops the channel. If it drops mid-write, the agent's memory
must not be left half-written. Two properties are required:

- **Atomic writes.** A memory operation completes fully or not at all. Prefer
  single-document operations, which ArangoDB applies atomically. Use stream
  transactions only when a write must span documents. A stream transaction whose
  connection drops mid-flight aborts rather than half-applying, which is the
  desired failure mode, and which means the agent must retry the whole operation.
- **Idempotent writes.** Re-applying an operation after a reconnect is a no-op.
  This requires a client-generated operation id carried on each write, so the
  server can deduplicate a replay. The existing "reconnect once on broken pipe"
  behavior is safe only when writes are idempotent.

This is the real internal-integrity risk in this topology. It is a torn write
across a flaky hop, not eavesdropping. See R-9.

## Escalation paths for stronger binding

The baseline (on-disk per-UID key plus full-disk encryption) is adequate when the
client machine and its root are trusted. Two escalations cover stronger needs
without any server-side change, because the server still only sees a public key.

- **Per-user against a malicious root:** a FIDO or `sk-ed25519` hardware key. The
  private key never exists on disk and every use needs a physical touch.
- **Per-machine, non-extractable:** a TPM-sealed key, for example via
  `ssh-tpm-agent`. The key is sealed to this machine's TPM and cannot be copied
  off it, even by root. This is the recommended answer for Weaver agents that
  cannot each hold a hardware token.

## Ownership boundary (HADES vs WeaverTools)

| Concern | Owner |
|---------|-------|
| The daemon, the `daemon-stdio` transport, the wire protocol | HADES |
| The server `authorized_keys` and `command=` template | HADES (operator) |
| The scoped per-key ArangoDB users and their grants | HADES (operator) |
| Session and scope assignment per enrolled key | HADES (operator) |
| Serving scoped network access to databases that already exist | HADES |
| Creating agent memory databases (the local `_system` create and grant) | WeaverTools |
| Creating the human's project databases (local SSH) | Operator (local admin) |
| What agent memory writes mean (memory semantics) | WeaverTools |
| Provisioning Weaver agent Linux users | WeaverTools |
| Generating agent keypairs and placing private keys in agent stores | WeaverTools |
| Submitting `{public_key, label}` for enrollment | WeaverTools |

HADES is the access path and the DBA gate. It serves scoped access to databases
that already exist. It does not create databases and does not hold a
`_system`-capable credential reachable over the wire. Weaver owns provisioning of
agent users and agent memory databases, and the meaning layered on top. HADES does
not modify Weaver users, Weaver ACLs, or the per-agent embedders.

## Network underlay (optional)

Confidentiality and mutual authentication come from SSH itself. A VPN, for example
WireGuard, can sit under the SSH channel as a second independent gate and to give
a roaming laptop a stable address. The VPN is optional and orthogonal. The SSH
layer is the authentication of record either way.

## Connection lifecycle

The daemon was built for connect-once, issue-many. A remote principal keeps one
SSH channel up and reuses it across many commands. The client reconnects on a
broken channel. SSH keepalives (`ServerAliveInterval`) detect a dropped link for a
roaming laptop. On stdin EOF the `daemon-stdio` process must exit cleanly so
dropped channels do not accumulate orphaned processes holding ArangoDB
connections.

## Threat model and residual risk

A gate is only worth claiming if you can name what defeats it and what catches the
failure.

### Trust assumptions

The model holds only while these hold. Each is a precondition, not a guarantee the
design provides.

- The HADES server host is trusted. Root on the server is outside the model by
  construction.
- The private key is in a per-UID location only. If it is not, per-agent
  isolation breaks with it.
- sshd `StrictModes` is on, so the shared service account's home and
  `authorized_keys` are not group- or world-writable. That one file is the trust
  anchor for all identity, session, and database assignment.
- The client enforces the pinned host key strictly, with no TOFU and no
  `accept-new`.
- No network ArangoDB user holds server-level (`_system`) Administrate, and none
  holds any right on a database it does not own. See the no-server-Administrate
  invariant.
- No daemon process reachable over the wire holds a `_system`-capable ArangoDB
  credential.

### The gates

Defense in depth. Each gate is independent, and a higher gate backstops a
consequence the one below could otherwise leave uncovered.

| # | Gate | Enforced by | Stops | Defeated by | Backstop |
|---|------|-------------|-------|-------------|----------|
| G1 | SSH key authentication | sshd plus crypto | Anyone without an enrolled private key | Theft of the private key | G2 |
| G2a | Per-UID key isolation | OS file permissions | Another unprivileged user on the client reading the key | Malicious client root | G2b |
| G2b | Key non-extractability | FIDO or TPM hardware | Copying the key off the machine at all, even by root | Physical compromise of the token or TPM | Physical custody |
| G3 | Forced command plus restrict | sshd and `authorized_keys` | Shell, forwarding, pty, pivot, and client-chosen session or database | A forced-command line that reads `$SSH_ORIGINAL_COMMAND` or client env | Review and test that the forced command ignores all client input |
| G4 | Transport refusal of `_system` and raw AQL | daemon code | Any principal invoking create or drop database, user or grant management, or raw AQL over the wire | A bug in the refusal path | G5 |
| G5 | Scoped ArangoDB user | ArangoDB | Writes, DDL, or data access outside the key's grant, and any reach into a non-owned or reserved database, regardless of daemon behavior | The daemon process holding an over-privileged or `_system` credential | Process-level credential isolation (one scoped user per key, never `_system`) |
| G6 | Host-key pin | client `known_hosts` | MITM or redirection to an impostor server | Loss of the pinned host key with no rotation path | SSH host certificate under a pinned CA |

### Residual risks

- **R-1 Malicious root on the client.** An on-disk per-UID key is readable by
  client root. Mitigation: FIDO or TPM keys (G2b). Accepted for trusted-client
  deployments, required for untrusted ones.
- **R-2 Stolen key, live use.** A copied key authenticates as the principal until
  revoked. The binary self-checks do not help, since there is no secret to
  withhold. Mitigation: hardware or TPM keys (R-1), anomaly detection on
  `--client-id`, and a bounded session lifetime (R-4).
- **R-3 Over-privileged ArangoDB connection.** If a remote `daemon-stdio` process
  connects as the full `hades` user, a G4 bypass or a daemon RCE reaches
  everything that user can do. Mitigation: one scoped ArangoDB user per key, held
  by a per-key process that has no other credential (G5). This is now the core
  tenancy mechanism, not an option. Resolved in design, pending implementation.
- **R-4 Revocation does not reach live sessions.** Removing the `authorized_keys`
  line stops new connections only. An existing connect-once channel survives until
  it is torn down. SSH certificate expiry does not help either, since a live
  session is not re-validated mid-channel. Mitigation: an operator kill of the
  active session or process on revoke, or a per-request revocation-list check in
  the daemon, plus a daemon-enforced maximum session duration. The session-
  lifetime bound must be enforced actively by the daemon, not assumed from the
  key or certificate. Separately, revoking a builder key must not strand or
  auto-drop the databases that key owns. The ownership record and the databases
  outlive the key. Define a lifecycle (retain, transfer, or archive owned
  databases on revoke), and never auto-drop on revoke.
- **R-5 Host-key rotation invalidates every client bundle.** A raw pinned host key
  means a server reinstall or key compromise requires re-shipping to all clients.
  Mitigation: pin a CA via `@cert-authority` and issue a host certificate, so the
  host key rotates under a stable pin (G6 backstop). Open, decide with the
  certificate question below.
- **R-6 Shared service account, one server UID.** All agent daemon processes run
  as the same UID and can `ptrace` one another by default. The per-agent boundary
  is client-side. Mitigation: `kernel.yama.ptrace_scope` at 2 or higher, and
  systemd hardening (`NoNewPrivileges`, `ProtectSystem`, private tmp) on the
  daemon. The ArangoDB scoped user (G5) limits what a cross-process read could
  reach even if this fails.
- **R-7 Resource exhaustion.** Agent vector search and traversals are expensive,
  and a channel is connect-once issue-many. Mitigation: per-key concurrency caps,
  sshd `MaxSessions` and `MaxStartups`, and per-query cost limits on top of the
  existing 16 MiB and `MAX_LIMIT=1000` floors. Database creation is a local,
  human- or Weaver-gated step, not a self-service over-the-wire vector, so it is
  not an exhaustion path from a network client. The local provisioning flow should
  still enforce per-owner quotas (maximum databases, maximum total size) so a
  builder namespace cannot exhaust disk.
- **R-8 Orphaned daemon processes.** A roaming client that reconnects often spawns
  a new `daemon-stdio` per link. Mitigation: clean exit on stdin EOF so dropped
  channels do not accumulate processes holding ArangoDB connections.
- **R-9 Torn write across the hop.** A dropped channel mid-write can leave agent
  memory half-applied, corrupting the substrate the agent relies on. SSH protects
  the bytes in flight but not write consistency. Mitigation: atomic and idempotent
  memory operations (see "Write consistency across the hop"), single-document
  where possible, stream transactions only when a write must span documents, and a
  client-generated operation id so a replay after reconnect deduplicates.

### Invariants that must hold for the claims to be true

If any of these is false, the corresponding claim is overstated.

1. The private key is only ever in a per-UID location (per-agent isolation).
2. The shipped package contains no secret (safe to deliver over any channel).
3. The forced command ignores all client-supplied input, including
   `$SSH_ORIGINAL_COMMAND` and environment (no session, scope, or argument
   injection).
4. The client enforces the pinned host key with no interactive fallback (instance
   binding).
5. No network ArangoDB user holds server-level (`_system`) Administrate or any
   right on a non-owned database, and no wire-reachable daemon process holds a
   `_system`-capable credential (nothing reaches the server or another tenant over
   the wire).
6. Each remote `daemon-stdio` process holds only its key's scoped ArangoDB
   credential (per-database isolation independent of daemon correctness).
7. Memory writes are atomic and idempotent (no torn state across the hop).
8. Every command is logged daemon-side with its `--client-id` and key fingerprint,
   giving command-level attribution. ArangoDB attributes at the database and user
   level through the per-key scoped user. The two layers together attribute fully.
   The earlier claim that principals share one ArangoDB identity per database does
   not hold in the scoped-user model and is dropped (attribution).

## Open questions

- SSH certificates versus raw keys. Raw keys in one `authorized_keys` file make
  that file the trust anchor for all identity, make revocation a file edit, and
  make host-key rotation re-ship every bundle. SSH certificates fix all three:
  host certs under a pinned CA allow host-key rotation without re-shipping, and
  user certs with short lifetimes plus a key revocation list turn revocation into
  expiry. Cost is CA key management. Under roughly five principals, raw keys are
  fine. Above that, certificates. Decide before the first enrollment.
- One shared client binary per HADES instance, or one binary per agent with a
  baked-in UID launch guard. The shared binary is simpler and loses no security,
  since the secret is the per-agent key.
- Whether to ship the TPM-sealed-key path in the first version or document it as a
  follow-on.
- The provisioning and rotation cadence for the per-key ArangoDB users, and the
  enrollment ledger format.
- Where the ownership record lives. The scoped ArangoDB user's effective grants
  already encode which databases a principal owns, so a separate ledger may be
  redundant. Decide whether `orient` and a `list-my-databases` view derive owned
  databases from the grants directly, or from a ledger maintained at provisioning
  time. The ledger, if kept, must outlive the key for the revocation lifecycle
  (R-4).
- Raw AQL on owned databases. A builder arguably has "full admin" of an owned
  database, which could include raw AQL. Recommendation: keep raw AQL off the wire
  even for owned databases, since it is the worst footgun (writes plus cost). If
  allowed later, only against owned databases and with hard time and cost caps.

## What this delivers

- One HADES server, no second instance, no sync.
- Per-principal access, enforced by the Linux user and the SSH key, not the
  machine.
- Per-database tenancy, enforced by a scoped ArangoDB user per key, independent of
  daemon correctness.
- Nothing reaches the server or another tenant over the wire. No network identity
  can administer the server, create or drop a database, manage users or grants,
  run raw AQL, or touch a database it does not own.
- Full database-level administration of owned databases over the wire for the
  human builder: DDL inside its own databases, and nothing outside them.
- Document create and delete within a pinned database for agents that need it,
  bounded by collection-level grants.
- Read and evaluate across databases for the evaluator, with no write and no
  admin.
- Admin and database provisioning reachable only locally. HADES provisions
  nothing over the wire: databases are created by Weaver (agent memory) or by
  local SSH (the human's project databases), and HADES serves scoped access to
  what exists. ArangoDB is bound to localhost.
- Mutual authentication, with the client verifying the server and the server
  verifying the principal.
- A binary bound to one HADES instance by the pinned server host key.
- Server-issued, revocable identities, with the private key never leaving the
  client and the shipped package carrying no secret.
- Atomic, idempotent memory writes that survive a dropped hop.
