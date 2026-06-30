# HADES Remote Client Authentication and Transport

**Status:** Draft, awaiting ratification
**Version:** 0.1
**Date:** 2026-06-30
**Relates to:** [daemon-protocol.md](daemon-protocol.md) (the wire protocol this carries over the network)

## Why this exists

Weaver agents run on a separate machine (a laptop) and need read-mostly access
to the HADES knowledge graph that lives on the HADES server. The naive option,
running a second HADES instance on the laptop, was rejected: it would mean two
databases to keep in sync and two sources of truth. There is one HADES server.
Remote callers reach it over an authenticated network channel.

This is **per-agent** access, not per-machine access. Each Weaver agent runs as
its own Linux user (the Weaver sandbox model). Agent A holding HADES access on a
laptop does NOT grant agent C on that same laptop any access. The boundary is the
Linux user, enforced by the kernel, not the machine.

This is NOT an MCP server. It reuses the existing daemon protocol verbatim and
carries it over SSH.

## Non-goals

- Not a second HADES instance and not a database copy. One server, one graph.
- Not box authentication. Two agents on the same laptop are distinct principals.
- Not an MCP server. The daemon protocol is the interface.
- HADES does not provision Weaver agent users and does not place their keys.
  HADES owns the server side (enrollment, transport, the daemon) and adapts to
  whatever Weaver provisions. See "Ownership boundary" below.

## The actors

| Principal | What it is | Where it runs |
|-----------|------------|---------------|
| HADES server | The single HADES daemon plus ArangoDB plus the embedder | The workstation/server |
| Weaver agent | An AI coding agent, running as a dedicated Linux UID | The client machine (laptop) |
| Operator | The human admin who enrolls agents and assigns tiers | The server |

## Architecture

The daemon protocol is transport-agnostic at the framing layer (length-prefixed
JSON). Today it runs over a local Unix socket. For remote access it runs over an
SSH channel. The daemon gains a stdio mode that speaks the same frames over
stdin and stdout, and SSH carries that stdio to and from the client.

```
client machine (laptop)                         HADES server
+-----------------------------+                 +-------------------------------+
| weaver agent (uid 1001)     |                 | sshd                          |
|   hades-client binary       |   ssh, key A    |   ForceCommand per key:       |
|     uses key A from agent A  | --------------> |   hades daemon-stdio          |
|     store, via ssh-agent     |  (JSON frames   |     --session agent           |
|                              |   over stdio)   |     --client-id agent-a       |
+-----------------------------+                 |        |                      |
                                                |        v  local unix socket   |
                                                |   ArangoDB  +  embedder        |
                                                +-------------------------------+
```

The client runs the equivalent of `ssh hades-server hades daemon-stdio`. The
daemon's existing `handle_connection` logic is unchanged. Only the transport in
front of it differs.

## Two-layer security model

The design separates the cryptographic gate (the real boundary) from the launch
guard (operational hygiene). These are different things with different strength.

### Layer 1: the cryptographic gate (the boundary)

This is what decides who can do anything.

- Each Weaver agent has its own SSH keypair. The private key lives only in that
  agent UID's store (`~/.ssh` at mode 0600, owned by that UID, or that UID's
  ssh-agent). The kernel prevents any other UID from reading it.
- The agent's public key sits in the server's `authorized_keys` under a single
  shared, locked-down service account. Agents are told apart by their key, not
  by separate Unix accounts on the server.
- The server entry forces a single command and strips all other SSH capability:

  ```
  command="hades daemon-stdio --session agent --client-id agent-a",restrict ssh-ed25519 AAAA...agentApub  agent-a
  ```

  `restrict` removes shell, port forwarding, pty, X11, and agent forwarding. That
  key can ONLY run the HADES daemon at the assigned tier. If the key leaks, the
  holder gets the HADES surface at that tier and nothing else. No shell on the
  server, no filesystem, no pivot.

This layer is enforced by the kernel (file permissions), by sshd
(`authorized_keys` plus `ForceCommand`), and by crypto (the SSH handshake). It is
the boundary.

### Layer 2: the launch guard (hygiene, not a boundary)

The client binary may self-check before it does anything:

- `geteuid()` matches the UID the binary was issued for.
- The owner of `/proc/self/exe` matches that UID.
- `/etc/machine-id` matches the machine the binary was issued for.

These catch misconfiguration early and raise the bar against casual misuse. They
are NOT a security boundary. Whoever controls the machine can patch out a
self-check, and because the binary carries no secret, bypassing the check yields
nothing. The guard is worth having for clear failure modes. It does not enforce
the per-agent property. Layer 1 does.

An OS-level companion to the guard is real enforcement of who can launch the
file: `chown agent-a hades-client && chmod 0500`. Only that UID can `exec` it.
This governs who may start the binary, not who the binary authenticates as. The
two together give clean per-UID launch plus the crypto gate.

## The per-agent invariant

One rule makes the difference between per-agent and per-box:

> The private key is only ever in a per-user location: an ssh-agent, or `~/.ssh`
> at mode 0600 owned by that agent UID. Never embedded in the binary. Never in a
> shared or world-readable path.

If this holds, per-agent isolation falls out automatically. The binary carries no
secret, so possessing the binary grants nothing. The key is readable only by its
owning UID, so one agent cannot use another agent's identity. The kernel does the
enforcing.

## Mutual authentication and instance binding

Authentication runs both directions, by construction of SSH.

| Direction | Mechanism | Result |
|-----------|-----------|--------|
| Server authenticates the agent | agent public key in server `authorized_keys` | only enrolled agents connect, identified individually |
| Client authenticates the server | server host key pinned in the client binary or its bundled `known_hosts` | the binary connects ONLY to the instance it was built for |

The pinned server host key is the load-bearing thing to bake into the per-client
package. It means a binary built for instance X cannot be repurposed against
instance Y, and a roaming laptop on a hostile network cannot be steered to an
impostor server. Neither side trusts the other on faith.

## Enrollment contract

The operator enrolls an agent from two inputs and one decision.

Inputs provided by the client side (Weaver):

- `public_key`: the agent's SSH public key. The private half never leaves the
  client.
- `label`: a client identity label, for example `agent-a` or `todd-laptop-a`.
  This is a tag for the enrollment ledger and the `authorized_keys` comment. It
  is not a Unix account.

Decision made by the operator (not the client):

- `tier`: the access tier this key is granted. The client does not choose its own
  privileges. Default for Weaver agents is `agent`.

Output, written on the server:

- One `authorized_keys` line under the shared service account, with `ForceCommand`
  encoding `--session <tier>` and `--client-id <label>`, plus `restrict`.

Because the private key stays on the client, the package shipped to the client
contains no secret. It holds the binary, the pinned server host key, and the host
and port. It can travel over any channel. The earlier problem of delivering a
private key over a secure wire does not arise in this model.

## Tier mapping

The daemon already classifies every command into a tier and gates by session
type (see daemon-protocol.md). Weaver agents map to the `agent` session, which
reaches Agent-tier commands only: queries and Persephone task operations, no
admin DDL, no raw AQL, no schema mutation. Admin work stays local to the server.
The tier is fixed at enrollment in the `ForceCommand`, so a remote caller cannot
request a higher tier than its key was granted.

## Escalation paths for stronger binding

The baseline (on-disk per-UID key plus full-disk encryption) is adequate when the
client machine and its root are trusted. Two escalations cover stronger needs
without any server-side change, because the server still only sees a public key.

- **Per-user against a malicious root:** a FIDO or `sk-ed25519` hardware key. The
  private key never exists on disk and every use needs a physical touch. This
  upgrades per-user to per-possession-of-the-token.
- **Per-machine, non-extractable:** a TPM-sealed key, for example via
  `ssh-tpm-agent`. The key is sealed to this machine's TPM and cannot be copied
  off it, even by root. This is the cryptographic form of "only runs on this
  machine," and it is the recommended answer for Weaver agents that cannot each
  hold a hardware token.

## Ownership boundary (HADES vs WeaverTools)

| Concern | Owner |
|---------|-------|
| The daemon, the `daemon-stdio` transport, the wire protocol | HADES |
| The server `authorized_keys` and `ForceCommand` template | HADES (operator) |
| Tier assignment per enrolled key | HADES (operator) |
| Provisioning Weaver agent Linux users | WeaverTools |
| Generating agent keypairs and placing private keys in agent stores | WeaverTools |
| Submitting `{public_key, label}` for enrollment | WeaverTools |

HADES adapts to whatever Weaver provisions. HADES does not modify Weaver users,
Weaver ACLs, or the per-agent embedders.

## Network underlay (optional)

mTLS-grade confidentiality and mutual auth come from SSH itself. A VPN, for
example WireGuard, can sit under the SSH channel as a second independent gate and
to give a roaming laptop a stable address. The VPN is optional and orthogonal.
The SSH layer is the authentication of record either way.

## Connection lifecycle

The daemon was built for connect-once, issue-many. A remote agent keeps one SSH
channel up and reuses it across many commands. The client reconnects on a broken
channel. SSH keepalives (`ServerAliveInterval`) detect a dropped link for a
roaming laptop. The existing daemon client already reconnects once on a broken
pipe, which carries over to the SSH transport.

## Open questions

- One shared client binary per HADES instance, or one binary per agent with a
  baked-in UID launch guard. The shared binary is simpler and loses no security,
  since the secret is the per-agent key. The per-agent binary adds the launch
  guard. Decide per deployment.
- Whether to ship a TPM-sealed-key path in the first version or document it as a
  follow-on.
- Revocation workflow detail: removing a line from `authorized_keys` is the
  mechanism. The ledger format and the rotation cadence are to be specified.

## What this delivers

- One HADES server, no second instance, no sync.
- Per-agent access, enforced by the Linux user and the SSH key, not the machine.
- Mutual authentication. The client verifies the server, the server verifies the
  agent.
- A binary bound to one HADES instance by the pinned server host key.
- Server-issued, revocable, rotatable identities, with the private key never
  leaving the client.
- A shipped package that contains no secret.
