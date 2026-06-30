# The Bastion Playbook — running the method in a Claude Code session

> **The operational companion to the foundation family.** Layer 1
> ([`bastion-of-context.md`](bastion-of-context.md)) is the *why*, Layer 2
> ([`the-bastion.md`](the-bastion.md)) is the *what*, Layer 3
> ([`../graph-methodology.md`](../graph-methodology.md)) is the *how on ArangoDB*.
> This is the *how in a session*: the sequence of moves to stand up — or retrofit
> — a bastion on a project, with a Claude Code agent doing the labor.
>
> Mechanics it calls (`create-database`, `schema apply`, `codebase ingest`,
> queries, embeddings) belong to the **`hades`** skill; the dissection method
> belongs to the **`hades-graph-methodology`** skill. This playbook sequences
> them and names the human gate at each step.

---

## How to open the session

Point the agent at this file and the foundation family, then say which path
you're on:

> *"We're standing up a bastion for `<project>`. Read `docs/foundation/`. I have
> a founding document / I need to write one. This is greenfield (new) / brownfield
> (existing code). Follow the playbook — draft, and I'll ratify."*

That last clause is the whole protocol. Read the next section before anything
else.

---

## The standing division of labor (read this first)

This is principle 8 made into a session rule, and it is what keeps the sessions
honest:

| The agent (Claude) does | The human (you) does |
|---|---|
| Dissects documents, drafts **candidate** axioms / principles / smells / decisions | **Ratifies.** Canon enters only on your word. |
| Authors Layer 2/3 architecture and mechanics | Corrects architecture; **authors Layer 1** (the *why*) |
| Runs ingests, queries, builds the suspect set, surfaces drift | **Adjudicates** each disagreement — amend the canon or reject the build |
| Lays the trace (anchors), proposes the de-ratification | Decides what is retired |

**The agent never writes to Layer 1 unprompted.** When practice surfaces a new
*why*, the agent drafts a candidate principle and stops; you ratify or rewrite it.
(New *how* that serves existing principles, the agent may author as Layer 2 and
you correct — that distinction is the gate doing its job.) When in doubt, the
agent proposes; it does not assert.

---

## Step 0 — The scope gate

Before building anything, apply the externality test (Layer 2 §2): does this
infrastructure **outlive its makers**, and does the lost *why* land on people who
had no say? If yes, the bastion is not optional. If it's throwaway code whose lost
*why* costs only you, a bastion is a preference — don't pay for one. Say so and
stop here if it doesn't clear the bar.

Then pick the path: **greenfield** (you have or will author the foundation first)
or **brownfield** (the code exists and the foundation is missing, partial, or
drifted). Most "new projects" are greenfield at the root and go brownfield within
weeks — start greenfield, keep the brownfield section bookmarked.

---

## Greenfield — foundation first, code proves forward

Run these in order. Each phase names its **artifact**, its **gate**, and the
**move**.

### P1 — Author the foundation (the IS / IS-NOT gate)

- **Move.** Hand the agent your founding document (a paper, a hypothesis doc, a
  PRD set) — or write one with it. The agent dissects it into **candidate** `IS`
  and `IS_NOT` principles, each with a **verbatim source quote** and proposed
  provenance.
- **Gate (human).** You ratify each principle. No ratified quote → no entry. This
  is the irreducible human act; the rest of the build hangs off it.
- **Artifact.** The two axiom containers — the opinionated foundation everything
  else proves against. Near-parity between `IS` and `IS_NOT` membership is the
  health signature.

> If you have no founding document, that *is* P1: the dissection forces you to
> state the identity. Write the foundation by arguing with the agent about what
> the project **is not**.

### P2 — Derive the smells (good and bad)

- **Move.** The agent reads the axioms (and any charter's *prescribe* / *proscribe*
  clauses) and proposes **smells**: bad smells (forbidden pattern → a match is a
  violation) and good smells (required pattern → its absence is a violation).
- **Gate (human).** Confirm each smell traces to a real principle. A smell with no
  axiom behind it is invented, not derived — cut it.
- **Artifact.** The enforceable rule layer (Layer C).

### P3 — Provision and schema

- **Move.** Provision a **HADES-owned** database (never a production research DB)
  and author the schema — collections, edge definitions, the named graph,
  `relation_order`, `feature_dim`, `model_type`. Drive this through the `hades`
  skill (`db create-database`, `schema apply`).
- **Gate.** `relation_order` includes the semantic/structural relations and
  **excludes** process/PM relations (the human-UI boundary).
- **Artifact.** An empty, well-typed graph ready for ingest.

### P4 — Ingest the foundation, then the code

- **Move.** Document-first: ingest the dissected foundation, then **ingest the
  code** (`hades codebase ingest`).
- **Caution (learned).** The **code ingest is a distinct layer that must actually
  run.** A build that dissects docs and axioms but skips code produces a concept
  graph with no migration deltas — it *looks* complete and is not.
- **Artifact.** The concept graph plus the code graph, not yet bridged.

### P5 — Bridge and run the suspect set

- **Move.** Bridge code to the foundation (`implements` spec, `cites` equation,
  `complies` / `violates` smell). Then run the **absence queries**.
- **Artifact.** The **suspect set** = the union of boundary violations, unsatisfied
  good smells, orphan code (no spec), and unembodied concepts (no code). This is
  your work-list. Non-connection is the signal.

### P6 — Lay the trace (close the loop both ways)

- **Move.** As each load-bearing unit is placed, write its **docstring anchor** —
  and **generate the anchor from the graph**, not by hand, so the code-side anchor
  and the graph-side bridge edge cannot diverge. The anchor names the stable
  **graph-node id**, the source spec/PRD, and the relation.
- **Gate.** Only load-bearing code gets an anchor; glue and re-exports inherit
  their module's. Orphans correctly get none.
- **Artifact.** A loop closed both ways — the graph points at the code, the code
  points back at the graph; a later verifier flags any anchor whose node moved.

---

## Brownfield — existing code, foundation reconstructed

The code is there and the map is missing or wrong. You can't author the gate up
front; you **manufacture** it by reconciliation. Same foundation, opposite order.

1. **Validate the apparatus on one unit first.** Do **not** run the whole repo.
   Pick the most **sharply-bounded** module — the one with the clearest organ
   identity, where foreign code stands out by contrast, and that changes least.
   **Pre-register** success: the flagged orphans are all real (no phantoms) *and*
   a manual read finds none the loop missed. Check both. If it flags phantoms or
   misses orphans, fix the loop **there**, on the cheap unit, not three units deep.
2. **Reconcile requirement-by-requirement.** Trace each `PRD → Spec → code` and
   sort it on two axes — *implemented?* × *should it be?*:
   - **A** (yes / yes) → **ratify**. The real implemented set; it becomes the gate.
   - **B** (yes / no) → **suspect**: built but unwanted or drifted → de-ratify or fix.
   - **C** (no / no) → **de-ratify the requirement**: close it, keep the *why*.
   - **D** (no / yes) → **gap**: the genuine backlog.
3. **Treat every orphan as a question, not a verdict.** Code with no spec is four
   ways: (1) real-but-undocumented → write the spec; (2) wrong organ → move it;
   (3) vestigial → delete or justify; (4) **tacit invariant the system leaned on →
   articulate it** (the most valuable, and the one "no spec means delete"
   destroys). Surface the constraint (1, 4) or remove the organisation (2, 3).
4. **The audit produces the gate.** Cell A and the articulated invariants are the
   ratified nodes a greenfield project would have authored up front. Once they
   exist, the code has something to bridge to — you are now greenfield going
   forward.

---

## The standing loop (after bootstrap)

The bastion is not a one-time build; it's a discipline the project runs on.

- **Prove before merge.** Each change bridges to the canon. The suspect set is a
  standing query, not an audit — orphans and violations surface continuously.
- **Adjudicate friction.** When the build meets reality and they disagree, that's
  a **decision for you**: amend the canon or reject the build. Neither
  waterfall-freeze nor agile-dissolve. The agent records the friction; you rule.
- **De-ratify honestly.** When experiment or accumulated friction falsifies a
  ratified axiom, run the rite: demote (don't delete), keep the retirement record,
  **cascade** — everything that proved against it drops to *pending
  re-ratification*. A canon that can't be demoted is a self-fulfilling prophecy.
- **Track wall vs scaffolding.** Mark each mechanism permanent (wall) or
  transitional (scaffolding); give scaffolding a retirement condition. A prop left
  standing after its wall ships is a **bypass** — an attack surface, not just
  clutter.
- **Harvest the deltas.** The divergence register read as `(deviation → correction)`
  pairs is a dataset — the material that trains whatever will later enforce the
  convention without a prop. The gate produces the data that retires its own
  scaffolding.
- **Keep documents reconciled.** Reconciling a doc is three acts: add the new,
  **remove the superseded**, refresh the date/status. De-stale **surgically** —
  never a blanket find-and-replace (it turns "X and Y folded into Z" into "Z and Z
  into Z"); keep the untouched original as the safety net.
- **Keep the provenance ritual.** Two resolutions, stitched by citations: public
  and coarse (issue -> PR -> squash commit stamping `(#N)`), local and fine
  (PRD -> Spec -> code, as repo files). Ingest only immutable facts (git
  history, merged PRs); read mutable forge state live via `gh`, never mirror
  it. No second work-tracking store, ever (see `the-method-hades.md`, "The
  provenance ritual").

---

## Where the tooling isn't there yet

Some of the method leads its implementation in HADES (Layer 2 §12 names the open
gaps: the de-ratification rite, the friction record, structural-conformance edges,
the wall/scaffolding node-role, the delta-ledger dual-read). **The method works
before the automation does** — where an edge type or a query isn't built, you and
the agent do that step by hand, and the by-hand pass is what proves the shape
before it's coded. Say which gap you're standing in and the agent will run it
manually rather than pretend the tool exists.

---

## See also

- [`bastion-of-context.md`](bastion-of-context.md) — Layer 1, the *why*.
- [`the-bastion.md`](the-bastion.md) — Layer 2, the *what* (the twelve principles,
  the two modes, the construction layer, conformance).
- [`../graph-methodology.md`](../graph-methodology.md) — Layer 3, the *how on
  ArangoDB*.
- `hades` skill — command mechanics. `hades-graph-methodology` skill — the
  dissection method.
