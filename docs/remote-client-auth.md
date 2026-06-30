# HADES Remote Client Authentication and Transport

**Status:** Draft, awaiting ratification
**Version:** 0.2
**Date:** 2026-06-30
**Relates to:** [daemon-protocol.md](daemon-protocol.md) (the wire protocol this carries over the network), [model-operation-vocabulary.md](model-operation-vocabulary.md) (the closed command surface)

## Why this exists

Weaver agents run on a separate machine (a laptop) and need access to the single
HADES server over a network. The naive option, running a second HADES instance on
the laptop, was rejected: two databases to keep in sync and two sources of truth.
There is one HADES server. Remote callers reach it over an authenticated channel.

Two access patterns drive the design:

1. **Agent compute on the laptop, agent memory on the HADES node, one hop
   between.** The agent's memory is a database on the HADES server. This path is
   read-write to that one database, not the read-only profile an early draft
   assumed.
2. **The human operator also uses HADES remotely** to read and evaluate across
   databases, with no ability to administer the server over the wire.

This is **per-principal** access, not per-machine. Each Weaver agent runs as its
own Linux user (the Weaver sandbox model). Agent A holding access on a laptop
does not grant agent C on that same laptop any access. And each enrolled key is
pinned to **one specific database** with an ArangoDB grant scoped to match.

This is NOT an MCP server. It reuses the existing daemon protocol verbatim and
carries it over SSH.

## Non-goals

- Not a second HADES instance and not a database copy. One server, one graph set.
- Not box authentication. Two agents on the same laptop are distinct principals.
- Not an MCP server. The daemon protocol is the interface.
- Not a path to server administration. No network identity can administer the
  box, drop a database, or change schema. Admin is local-only (see "Admin is
  local-only").
- HADES does not provision Weaver agent users and does not place their keys.
  HADES owns the server side (enrollment, transport, the daemon, the scoped
  ArangoDB users) and adapts to whatever Weaver provisions. See "Ownership
  boundary".

## The actors

| Principal | What it is | Memory path | Where it runs |
|-----------|------------|-------------|---------------|
| HADES server | The single daemon plus ArangoDB plus the embedder | n/a | The workstation/server |
| Weaver agent | An AI coding agent, one dedicated Linux UID each | read-write to one pinned database | The client machine (laptop) |
| Operator (remote) | The human, reading and evaluating across databases | read-only across databases | The client machine |
| Operator (admin) | The human administering the server | full, local only | On the server, after SSH and elevation |

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
| Operator (remote) | operator key | `agent` | `hades_ro_all` | Access plus collection RO across databases, Administrate on none |
| Operator (admin) | local Unix socket | `admin` | the `hades` user | full, local only |

No network principal in this table holds ArangoDB **Administrate** on any
database. That is the load-bearing invariant of the whole design (see "The
no-Administrate invariant").

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
- `database`: the one database this key may reach.
- `grant-spec`: the ArangoDB grant for this key's scoped user, for example
  `{db: agent_a_mem, access: ro, collections: {memory: rw, audit: ro}}`.

Output, written on the server:

- One `authorized_keys` line under the shared service account, with `command=`
  pinning `--session`, `--client-id`, and `--database`, plus `restrict`.
- One ArangoDB user, scoped per the grant-spec, that the daemon process for this
  key authenticates as.

```
command="hades daemon-stdio --session agent --client-id agent-a --database agent_a_mem",restrict ssh-ed25519 AAAA...agentApub  agent-a
```

`--database` joins `--session` and `--client-id` in the set the server pins and
the client cannot override. The daemon must reject or ignore any client-supplied
database. This is the command-injection concern of G3 extended to the database.

Because the private key stays on the client, the shipped package contains no
secret. It holds the binary, the pinned server host key, and the host and port.
It can travel over any channel.

## The capability model: documents over the network, schema only local

The network can populate and modify documents within existing containers. It
cannot change the schema. This single rule sorts every command.

| Network-allowed (gated by the key's ArangoDB user) | Network-forbidden (local admin only) |
|-----------------------------------------------------|--------------------------------------|
| reads: `db.get`, `db.list`, `db.count`, `db.recent`, graph traverse, neighbors, shortest-path, `graph-embed` queries | raw AQL (`db.aql`): AQL can write and is expensive, never on the wire |
| DML: `db.insert`, `db.update`, `db.delete`, succeeding only where the key's ArangoDB user grants collection Read/Write | schema and DDL: create or drop collection, create index, graph create or drop or materialize, `db.schema.init`, create or drop database, `db.purge` |

The current daemon classifies DML (`db.insert`, `db.update`, `db.delete`) as
Admin tier, next to the destructive DDL commands. This design splits them. DML
becomes network-reachable, gated by the ArangoDB user rather than by tier. The
DDL and raw-AQL set is refused on the network transport outright, on top of the
ArangoDB grant withholding the privilege. Two independent gates then agree: the
transport will not carry the destructive command, and no network identity could
execute it.

This is why general node create and delete is safe over the network here. The
catastrophic floor is held by ArangoDB (no Administrate) and by transport
DDL-refusal, not by restricting the command vocabulary. There is no need for
net-new `memory.*` commands. Weaver layers memory semantics on top of generic
scoped CRUD on the client side, which keeps the ownership boundary intact.

### The no-Administrate invariant

No network ArangoDB identity ever holds database-level **Administrate** or any
server-level admin right. Mapped to ArangoDB's two access levels:

- Database level **Access** (not Administrate) lets the user operate inside a
  database without creating or dropping collections or the database itself.
- Collection level **Read/Write** lets the user create and delete documents in
  named collections. Collection level **Read-Only** protects a container from
  document deletion. Collection level **none** hides it.

So "create and delete nodes, but cannot drop the database, and cannot delete
nodes in a protected container" maps onto `grantDatabase(Access)` plus
`grantCollection(rw | ro | none)` per collection. This is core ArangoDB
functionality. Confirm `grantCollection` behaves this way on the deployed
ArangoDB edition before relying on it.

Every "cannot destroy anything over the network" claim in this document depends
on this invariant holding. It is a precondition, not something daemon code
provides.

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

## Admin is local-only

Administration is not a network session. It is reached only by logging into the
server and elevating there.

- The ArangoDB endpoint and web interface are bound to `127.0.0.1` only, never
  `0.0.0.0`. They are never advertised on the network.
- Raw AQL and any DDL or admin action are permitted only from a local ArangoDB
  user on the server.
- The operator works remotely by first connecting over SSH to the server, then
  elevating to the local admin context. There is no remote admin path and no
  remote `admin` session over the wire.

Localhost-only binding is what makes "admin is local" enforceable rather than
aspirational.

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
| Session and database assignment per enrolled key | HADES (operator) |
| What agent memory writes mean (memory semantics) | WeaverTools |
| Provisioning Weaver agent Linux users | WeaverTools |
| Generating agent keypairs and placing private keys in agent stores | WeaverTools |
| Submitting `{public_key, label}` for enrollment | WeaverTools |

HADES is the access path and the DBA gate. It serves scoped node CRUD on a pinned
database. Weaver owns the meaning layered on top. HADES does not modify Weaver
users, Weaver ACLs, or the per-agent embedders.

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
- No network ArangoDB user holds Administrate. See the no-Administrate invariant.

### The gates

Defense in depth. Each gate is independent, and a higher gate backstops a
consequence the one below could otherwise leave uncovered.

| # | Gate | Enforced by | Stops | Defeated by | Backstop |
|---|------|-------------|-------|-------------|----------|
| G1 | SSH key authentication | sshd plus crypto | Anyone without an enrolled private key | Theft of the private key | G2 |
| G2a | Per-UID key isolation | OS file permissions | Another unprivileged user on the client reading the key | Malicious client root | G2b |
| G2b | Key non-extractability | FIDO or TPM hardware | Copying the key off the machine at all, even by root | Physical compromise of the token or TPM | Physical custody |
| G3 | Forced command plus restrict | sshd and `authorized_keys` | Shell, forwarding, pty, pivot, and client-chosen session or database | A forced-command line that reads `$SSH_ORIGINAL_COMMAND` or client env | Review and test that the forced command ignores all client input |
| G4 | Daemon session and DDL refusal | daemon code | An agent session invoking admin, DDL, or raw AQL | A bug in the gating path | G5 |
| G5 | Scoped ArangoDB user | ArangoDB | Writes or data access outside the key's grant, regardless of daemon behavior | The daemon process holding an over-privileged credential | Process-level credential isolation (one scoped user per key) |
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
  key or certificate.
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
  existing 16 MiB and `MAX_LIMIT=1000` floors.
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
   `$SSH_ORIGINAL_COMMAND` and environment (no session, database, or argument
   injection).
4. The client enforces the pinned host key with no interactive fallback (instance
   binding).
5. No network ArangoDB user holds Administrate (nothing destructive over the
   wire).
6. Each remote `daemon-stdio` process holds only its key's scoped ArangoDB
   credential (per-database isolation independent of daemon correctness).
7. Memory writes are atomic and idempotent (no torn state across the hop).
8. Every command is logged with its `--client-id` and key fingerprint, since all
   principals share one ArangoDB-side service identity per database and the DB
   logs cannot otherwise attribute (attribution).

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

## What this delivers

- One HADES server, no second instance, no sync.
- Per-principal access, enforced by the Linux user and the SSH key, not the
  machine.
- Per-database tenancy, enforced by a scoped ArangoDB user per key, independent of
  daemon correctness.
- Nothing destructive over the wire. No network identity can drop a database,
  drop a collection, change schema, or run raw AQL.
- Document create and delete within a pinned database for agents that need it,
  bounded by collection-level grants.
- Read and evaluate across databases for the remote operator, with no write and no
  admin.
- Admin reachable only by local SSH and elevation, with ArangoDB bound to
  localhost.
- Mutual authentication, with the client verifying the server and the server
  verifying the principal.
- A binary bound to one HADES instance by the pinned server host key.
- Server-issued, revocable identities, with the private key never leaving the
  client and the shipped package carrying no secret.
- Atomic, idempotent memory writes that survive a dropped hop.
