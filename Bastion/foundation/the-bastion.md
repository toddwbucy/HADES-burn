# The Bastion — A Backend-Agnostic Method for Context Governance

> **Layer 2 of the foundation family** — the architecture, the *what*. This
> document is the founding specification the implementation is measured against.
> It serves the principles in [`bastion-of-context.md`](bastion-of-context.md)
> (Layer 1, the *why*) and is realized on ArangoDB by
> [`../graph-methodology.md`](../graph-methodology.md) (Layer 3, the *how on this
> backend*).
>
> This document was written **backwards** — recovered from a system (HADES) that
> crystallized in situ, then lifted clear of the substrate it grew on. That it
> describes the same structure two sibling projects arrived at by the opposite
> route (building forward from a founding artifact) is the evidence in §10, not a
> coincidence.

---

## 1. Thesis

A **bastion** is a project's canon — its identity, its ratified decisions, its
conventions, the *why* behind all three — stood up as a **context graph** and
wired into the code and release pipeline as a **governance layer**.

It is not a knowledge base you read. It is a gate you **prove against**. New
work — a concept, a file, a decision, a release — earns its place by tracing,
through ratified connections, back to an opinionated foundation. What cannot
trace is not "untagged"; it is **suspect**, and the bastion makes that absence
the primary query. The product is the negative space.

The backend is incidental. ArangoDB happens to host the reference
implementation, but nothing in the method needs it (§8, the backend contract).
What is load-bearing is *how you use a graph substrate to govern a project*, not
which substrate. HADES is one species; the bastion is the genus.

---

## 2. Scope — where the method is not optional

The bastion is for **infrastructure that must outlive its makers**, where the
cost of the lost *why* is externalized onto people who had no say in the
maintenance. The dividing line is the **externality**: where the failure lands
on others and the lifespan exceeds the tenure, the bastion is not optional;
everywhere else it is a preference.

This scope clause binds everything below. The method is **demonstrated, not
proven universal** — three instances (§10), one structure, this boundary. The
honest path is to capture forward from today (cheap, faithful — the *why* caught
as it is made) and backfill the past on contact, while the people who hold it are
still here to ask — the greenfield and brownfield modes of §5.

---

## 3. Invariants — the twelve principles as architectural commitments

Each commitment is what the structure must *do* to serve a principle from Layer
1. The principle is the warrant; the commitment is the obligation on the build.

| # | Principle (Layer 1) | Architectural commitment |
|---|---------------------|--------------------------|
| C1 | Query for the miss | Non-connection is a **first-class, standing query**, not an audit. Orphans and unembodied claims are selected by the *absence* of an edge, cheaply, on demand. |
| C2 | Membership earned; artifact is the unit | Entry is **by proof, not insertion**. The ratified unit is the **artifact, judged whole**; its extracted details **inherit** that verdict (§4, ratification rule). |
| C3 | Graph reflects given context, not reality | The graph is a model of **ratified context**, never a claim about the world. A structural boundary separates *what we were told* from *what is true*; experiment lives **outside**, and results re-enter only as ratified context (the re-entry path is an open gap — §12). |
| C4 | Constraint is generative | An **opinionated foundation comes first**. The identity gate is authored before content; content proves against it. No foundation → a heap. |
| C5 | Canon lives between two failures | Revision is **adjudicated, never automatic**. When the build meets reality, the friction is captured as an artifact and forced to a **human verdict — amend canon or reject build**. Neither waterfall-freeze nor agile-dissolve. |
| C6 | Replace the priesthood's labor, not its authority | The coordination a clerical class once held in their heads becomes a **queryable artifact**. Authority stays human and **concentrates**; labor moves to the graph and the agents. |
| C7 | The de-ratification rite is the firewall | There is an **exit gate, not only an entry gate**. A ratified axiom is **demotable** by an explicit rite that cascades to its dependents (§4, de-ratification). Sacred enough to ground everything, revisable enough to be retired. |
| C8 | Humans author, agents labor, graph holds canon | Three roles, **structurally separated**: humans author and adjudicate; agents build and surface drift; the graph holds the canon at query speed. The human-authority boundary is distinct from the agent-labor surface. |
| C9 | The canon must be live | The canon is a **pre-action gate**, proven against *before* submission, not a document cited only to scold. Every convention a breach surfaces is **captured once** and becomes enforceable, so it never surprises anyone again. |
| C10 | Trust migrates to the artifact | Trust attaches to **artifacts** — auditable, transferable — not to a contributor's intentions. The canon's scope must reach the **dead ground**: the build and release pipeline, or the gate has a blind angle. |
| C11 | Wall and scaffolding | Node role is **first-class**: every mechanism is **permanent (wall)** or **transitional (scaffolding)**. Scaffolding carries a **retirement condition** and may not be load-bearing in the end state — a prop left standing is a **bypass** (the security face of C10). See §6. |
| C12 | The negative space is generative | The divergence register is **dual-read**: a compliance ledger *and* a dataset. Each `(deviation → correction)` is a **training example** for whatever will later enforce the convention without a prop. The gate produces the data that retires its own scaffolding — the bastion as **flywheel, not filter**. See §6. |

---

## 4. The architecture — five layers, stated backend-neutral

The method is invariant across the document it is pointed at and the substrate it
runs on. Content changes per source; structure does not.

**Layer A — Foundation, dissected.** *Document-first, code-last.* The foundation
document is dissected into typed **concept nodes** — definitions, abstractions,
axioms, lineage — wired with internal relational edges (within a source) and
cross-source edges (between sources). This is a relational map of *what the
document says*, before any judgement of identity. (That ordering is the
**greenfield** mode; brownfield inverts it — code first, foundation
reconstructed — see §5.)

**Layer B — The identity gate (IS / IS-NOT).** The opinionated foundation,
encoded as **exactly two** containers: `IS` (what the project *is*) and `IS_NOT`
(what it is defined *against*). Every concept carries **both poles** — a `basis`
edge to `IS` and a `validated-against` edge to `IS_NOT`. Two edges per concept;
**near-parity in their counts is the health signature** that membership was
earned on both poles. Every principle inside a container is grounded in a
**verbatim source quote** with structured provenance and **human ratification** —
no ratified quote, no entry.

**Layer C — Smells derived from axioms (and the charters that bridge to them).**
Enforceable rules that turn principles into **code-level constraints**. Each smell
traces to the specific `IS`/`IS_NOT` principle it enforces or guards, and carries
the concrete patterns at stake. Smells come in **two polarities**: a **bad smell**
(a *proscribe* statement) names a forbidden pattern whose **match** is a
violation; a **good smell** (a *prescribe* statement) names a required pattern
whose **absence** is a violation. The good smell makes the miss symmetric — not
only "forbidden pattern present" but "required pattern missing" — so the layer
checks the **content** of a requirement, not just its structural placement. A
smell is `derived` (read straight from an axiom, or projected from a charter's
prescribe/proscribe clause) or `operational` (promoted from a recurring real bug),
and it carries the axiom it bridges to, the node it came from, and a scope. This
is C9 made mechanical: a convention a breach surfaces is captured once, as a
smell, and enforced thereafter.

**Layer D — Artifacts bridged to the foundation.** Code, decisions, specs, and
releases are ingested and **bridged** to the concept graph: a file `implements` a
spec, `cites` an equation, `complies`/`violates` a smell; a decision `traces` to
a requirement. The artifact now inherits the same identity test the concepts
were held to.

**Layer E — The suspect set.** The payoff, as **queries over absence** (C1):
*orphan artifact* (no bridge edge — embodies nothing in the foundation),
*unembodied concept* (no structural-embodiment edge — a claim with no
implementation), *broken provenance* (a detail whose chain to a ratified artifact
is severed). The divergence register stops being a hand-maintained list and
becomes a standing query. **Non-connection is the signal.**

### The ratification rule — the unit of proof is the artifact (C2)

A node earns its place one of two ways, decided by one question: **does it exist
in the world on its own?**

- **Artifacts prove.** A node that corresponds to a real-world artifact — a file,
  an issue, a spec, a decision record — must prove against the axioms to be
  **ratified**, and it is judged *as a whole*.
- **Details inherit.** A node born *from* an already-ratified artifact — its
  extracted claims, its configuration — **inherits** that artifact's verdict. It
  is not gated again on its own.

The criterion is **existence, not hierarchy**: anything with its own footprint in
the world can change — and therefore *drift* — independently of its parent, so it
must stand on its own. Anything that exists only as a description inside a
ratified artifact cannot drift alone, so it rides the parent's verdict. **Lineage
and ratification are different edges**: one records where a node came from, the
other records that it proved. Never conflate them.

### The de-ratification rite — the firewall (C7)

*This is the exit gate. The entry gate above is thorough; without a symmetric
exit, a ratified axiom that reality later falsifies stays load-bearing forever —
a prophecy that fulfills itself. The rite is what keeps the bastion a
pre-registration and not a closed loop.*

The rite has four obligatory moves:

1. **Trigger.** De-ratification begins only from a **falsification event**: an
   experiment result re-entering from outside the graph (C3), or an adjudicated
   **friction record** (C5) where the human verdict was *the canon is wrong, not
   the build*. An axiom is never retired casually or automatically.

2. **Demote, do not delete.** The axiom moves from `ratified` to **`retired`**,
   carrying a **retirement record**: what falsified it, who adjudicated, when, and
   what (if anything) replaces it. The node stays in the graph. The *why of the
   retraction* is itself canon, and itself inheritable — a future maintainer must
   be able to query why an axiom was demoted, not just that it was.

3. **Cascade.** Every artifact that proved against the retired axiom is
   **re-opened**: its `basis` edges to the retired axiom are flagged and those
   artifacts drop to **`pending re-ratification`**. They do **not** silently keep
   their old verdict. This cascade is the firewall proper — a demoted axiom does
   not leave its dependents standing on retracted ground.

4. **Re-settle.** Each re-opened artifact is re-proven against the amended
   foundation, or itself retired. The suspect set (Layer E) surfaces the
   re-opened set as a standing query until it is drained.

Entry is hard (verbatim quote + human ratification) so the foundation is sacred;
exit exists (this rite) so the foundation is revisable; the cascade ensures
revision is honest rather than cosmetic. **That triad is principle 7.**

---

## 5. Two operating modes — greenfield and brownfield

The layers (§4) are invariant; the order in which you *populate* them is not.
There are two methodological approaches, and which one you are in turns on a
single fact: **did the foundation exist before the code, or must it be recovered
from code that already drifted?** Most real projects are the second.

### Greenfield — foundation first, code proves forward

The foundation is authored, or pre-exists, before the artifacts it governs. Layer
A's *document-first, code-last* is literal: the gate is built, then each artifact
proves forward against it. This approach **stresses the entry gate** — C2
(ratification) and C4 (foundation-first) carry the load. The exit gate is nearly
idle: when the canon is external and fixed — a published paper you do not get to
revise — de-ratification has nothing to fire on.

*Reference instance: **NL-Hecate***. The author's papers are the fixed
foundation; code earns its place by proving forward against equations it cannot
amend.

### Brownfield — code first, foundation reconstructed

The code already exists and the foundation is missing, partial, or drifted — the
common case, and the one §2 calls *backfill the past on contact*. You cannot
author the gate up front, because the ratified nodes the code must bridge to **do
not exist yet**. So you *manufacture* them with a **reconciliation audit**: Layer
E run by hand, before the graph can run it automatically. This approach
**stresses the exit gate** — C5 (adjudication) and C7 (de-ratification) carry the
load, because every requirement and every built module has to be tried against
the question *is this still canon, or has it drifted?*

*Reference instances: **WeaverTools*** — PRDs never reconciled to the
`weaver-database` code — and **HADES itself**, the reflexive case: built
backwards, foundation written last, and the tool that will eventually *automate
the very audit WeaverTools now runs by hand*.

### The reconciliation audit — the brownfield bootstrap

Trace each foundation requirement `PRD → Spec → code` and sort it on two axes —
*is it implemented?* × *should it be?*:

| | should be | should **not** be |
|---|---|---|
| **implemented** | **A — keep.** Built and wanted → **ratify** (C2). This is the graph's real *implemented* set; it becomes the gate. | **B — divergent.** Built but unwanted, or drifted from its spec → **suspect set** (Layer E); a **de-ratification** candidate (C7). |
| **not implemented** | **D — gap.** The genuine backlog → **unembodied concept** (Layer E). | **C — abandoned.** Proposed, then dropped → **de-ratify the requirement** (C7): close it, keep the *why*. |

The four cells are not a new framework — they are the method's existing
vocabulary in audit form (A → C2, B and C → C7, D → Layer E). Cell **B** is the
one an audit-for-gaps alone would miss: code that exists with no live requirement
behind it is exactly the orphan the immune-system graph is built to surface — but
an orphan is a question, not a verdict: *no bridge edge* resolves four ways (§7),
only one of which is deletion. And
the audit's evidence discipline mirrors the gate's: **every cell-assignment cites
the exact code module** that does or does not embody the requirement, the same
way a gate principle cites its verbatim quote. A PRD self-labelled "drafted"
proves nothing about what shipped; only the code does.

### Why both modes are load-bearing

The two approaches stress opposite gates, so **each is the debugger for the gate
the other cannot reach**. The de-ratification rite (§4) is structurally
*invisible* in greenfield — you never retire a fixed external canon — and
*central* in brownfield, where cells B and C are de-ratification carried out by
hand. A method documented only from greenfield reference work would have no
occasion to discover the exit gate at all; the brownfield instances are what
force it into view. The reconciliation audit is also where Layer E meets C5: the
graph surfaces the *absence*, but sorting each absence into kill (B), build (D),
or close (C) is the **irreducible human adjudication**. The graph finds the
suspects; the matrix is the courtroom.

---

## 6. The construction layer — wall, scaffolding, and the delta ledger

A bastion governs a system *being built*, so the canon must hold not only what is
ratified true but what is ratified **temporary**. This adds a **node-role** axis
orthogonal to the verdict states (`ratified` / `pending` / `retired`): every
mechanism is **wall** or **scaffolding**.

- **Wall** — permanent structure: ratified conventions, the released contract,
  behaviour internalised where it belongs (in code, in a trained model, in the
  harness). Self-standing; nothing props it.
- **Scaffolding** — transitional: the system prompt that teaches a convention, a
  heuristic stand-in for a detector not yet trained, an explicit tool surface
  that exists only until the behaviour is automatic. Scaffolding carries a
  **retirement condition** (`retire_when: <its wall replacement ships>`) and a
  record of **what it feeds**.

**The delta ledger.** The suspect set (Layer E) read *forward* is a compliance
finding — *this actual deviates from the ideal*. Read as a *pair*, each
`(deviation → correction)` edge is a **labelled example** of the convention. The
two are the same edges, twice: the immune system's divergence register **is** the
training corpus for whatever will later enforce the convention without a prop — a
trained model, a lint rule, a codegen template; the medium is immaterial. This is
the third reading of "both sides of the keyboard" (§9): non-connection does not
only constrain the model and spare the humans re-derivation, it is the
**feedstock** that builds the automation. The bastion is a **flywheel** — the
gate generates the data that retires the gate's own scaffolding.

**The retirement invariant.** No scaffolding node may be load-bearing in the end
state. This is a standing query — the **build map** — over three conditions:

- *wall not yet built* → the scaffolding **must stay** (tearing it down drops the
  structure).
- *wall built, deltas harvested* → the scaffolding is **safe to retire** (and by
  C7, retired with its record, not deleted).
- *wall built but scaffolding still reachable* → the **vulnerability**: a bypass
  left leaning against a convention that was supposed to be load-bearing
  elsewhere. The canon was internalised, but a public ladder still stands — the
  security face of C10 (the canon's scope must reach the dead ground) and of C11.
  Scaffolding is exactly the surface an adversary drives directly.

The construction layer is where the two modes meet the dataset: the brownfield
reconciliation audit (§5) sorts the B / C / D cells, and those cells **are** the
deltas the ledger harvests. The audit is not only how a bastion is bootstrapped
onto existing code — it is the **first turn of the flywheel**.

---

## 7. Conformance — code against documentation

The brownfield reorganisation (§5) does not end when the code is in the right
crates. It leaves a **standing discipline**: keeping code and documentation in
agreement *after* the reconciliation lands. Two products, one method — the
reorganisation is the one-time job, conformance is the invariant it establishes.

**Descriptive mirror, prescriptive commitment.** Documentation is not one tree.
The **architecture mirror** — the spec layer that mirrors the code's own
structure — is *descriptive*: it tracks the code and is expected to change as the
code changes. The **requirement layer** (PRD / decision record) is
*prescriptive*: it is the commitment. Conflating them re-commits structure that
was deliberately left open — every refactor that moves a function becomes a
requirement edit, and documentation maintenance is coupled to code churn. This
split is the mechanism behind P5: the descriptive mirror absorbs churn (no
waterfall-freeze), the prescriptive layer holds the commitments (no
agile-dissolve). **The mirror describes; the PRD commits.**

**Conformance is rules, not a manifest.** A manifest yields *missing-file* and
*extra-file*. The graph encodes the architecture as **edges** — dependency,
boundary, contract — so ingesting code yields the *precise* violation: *this unit
imports across a boundary it may not*, *this code claims a contract it does not
satisfy*, *this code sits in the wrong organ*. This extends the
compliance-before-merge gate from **semantic** provenance (does this trace to a
ratified meaning?) to **structural and content** conformance.

**Content conformance, via the smell layer.** Structural rules ask *does this obey
the boundaries?* The smell layer (Layer C) asks *does this satisfy the
requirement?* Ingested code is matched against the good and bad smells, and a
`complies` / `violates` edge records each — a bad-smell match names the forbidden
pattern and the axiom it breaks; an unsatisfied good smell names a requirement the
code was meant to meet and does not. The build's **suspect set is the union** of
these content violations, the boundary breaks, the orphan code (no spec), and the
unembodied concepts (no code) — and that union **is the migration work-list**.

**The strongest gate is executable.** A crate boundary is a dependency edge the
*compiler* checks. Promoting an organ to its own crate turns an asserted axiom —
"only the harness touches the database" — into a **compiler-enforced invariant**:
the build refuses code that breaks the boundary. *The cellular model becomes the
build graph.* This is C10 (the canon must reach the dead ground) and P1 (the
miss) taken to their limit — the canon stops being something code is *checked
against* and becomes something the substrate *refuses to violate*. Make the canon
executable wherever the substrate allows; the compiler is the cheapest gate that
never sleeps.

**Two uses of the graph, kept separate.** In brownfield the graph is used twice,
and conflating them breaks the reorganisation. As **driver**, it matches messy
code against the target structure to find what is misplaced — and this is
*rules*-driven, because the code being moved does not yet carry anchors (their
absence is the disease, not the instrument). As **steady-state trace**, it is the
docstring anchors written *as each piece is placed* — each load-bearing unit
naming the stable **graph-node identifier** it embodies (plus the source spec/PRD
and the relation), the way formula-code names its equation. Critically the anchor
is **generated from the graph**, not hand-authored: the graph is the source of
truth, so the code-side anchor and the graph-side bridge edge **cannot silently
diverge**. The reorganisation is the **act of laying the trace**, not a thing the
trace guides. This closes the loop **both ways** — the graph points at the code
(edges), the code points back at the graph (anchors) — so after it lands, rules
catch wrong-place and wrong-dependency, and a verifier flags any anchor whose node
has moved or vanished. Only load-bearing code carries an anchor; glue and
re-exports inherit their module's.

**The orphan is a question, not a verdict.** Code with no spec is the
highest-signal find precisely because it is *ambiguous*, four ways, each wanting a
different action: (1) it implements a real requirement nobody wrote down — *write
the spec*; (2) it belongs in another organ — *move it* (most of a leaked
boundary); (3) it implements nothing intended — *delete or justify it*; (4) it
encodes a **tacit invariant** the system leaned on unstated — *articulate it*,
the most valuable case, because it surfaces architecture the system depended on
without ever stating. "No spec means delete" destroys case four. Underneath is
P4: code with no spec is **organisation with no stated constraint**, and the only
honest resolutions are to *surface the constraint* (1, 4) or *remove the
organisation* (2, 3). The opposite orphan — spec with no code — is the unembodied
concept (Layer E); the two directions together are the **both-poles parity check**
(Layer B) applied to code.

**Coverage is per-node, never aggregate.** The target is not "every file
annotated." It is *every load-bearing spec feature has implementing code that
anchors back to it*, checked per feature with the both-poles query — because an
aggregate annotated-file percentage **hides exactly the orphans the method exists
to find**.

**Validate the apparatus before the work leans on it.** The conformance loop is
an instrument, and an instrument is calibrated before it is trusted. Run the
reorganisation one unit at a time and make the **first unit do double duty**: it
extracts the unit (the visible job) *and* it proves the loop produces trustworthy
orphan signals (the real job). **Pre-register** the success criteria — the flagged
orphans are all real (no phantoms) *and* a manual read finds none the loop missed
(complete) — and check both after the run. Choose the most sharply-bounded unit
first, where foreign code stands out by contrast and the target moves least; if
the loop flags phantoms or misses real orphans, it is fixed there, cheaply, not
three units deep. This is the foundation's own spirit — *the apparatus tests the
bet against reality* — turned on the apparatus itself. One failure mode the first
run exposes and the rest inherit: a build that dissects the documentation and the
axioms but **skips the code ingest** produces a concept graph with no migration
deltas — it *looks* complete and is not. The code ingest is a distinct layer that
must actually run, not an implied step.

**Reconciliation hygiene.** Keeping the documentation in agreement *with itself*
during a reorganisation is its own discipline. Reconciling a document is **three
acts, not one**: the new content has to be present, the superseded content has to
be *gone*, and the currency metadata (date, status) has to be refreshed. A pass
that only *adds* leaves old and new mixed and reads as stale even when the content
is correct — all three together are what "reconciled" means. (Removing the
superseded content is the document-level form of C7: demote, don't merely
accrete.) De-staling is **surgical, never blanket**: a document holds two kinds of
reference to a retired name — a *live* reference that should be renamed, and a
*narrative-history* reference that records what the old name was and must be kept
(the retirement record's *why*). A blanket find-and-replace cannot tell them
apart — one such replace once turned "X and Y folded into Z" into "Z and Z into
Z." Keep the untouched original as the safety net, and run the systematic de-stale
at rebuild time, not as a per-document `sed`.

> The concrete mechanics — the structural-conformance edge types
> (`may-depend-on`, `implements-contract`, `belongs-to-organ`), the
> `file → module-spec → crate-spec → PRD → axiom` granularity, and the
> docstring-anchor format — are backend- and language-specific and live in Layer 3
> ([`../graph-methodology.md`](../graph-methodology.md)), not here.

---

## 8. The backend contract — what makes "any backend" true

"Backend-agnostic" is an aspiration until the substrate's obligations are written
down. They are. A substrate can host a bastion **iff** it provides all five:

1. **Typed nodes.** Concept and artifact nodes with a type and structured
   properties, addressable by a stable identity.
2. **Typed, directed edges.** Relations carry a type, so identity edges
   (`basis` / `validated-against`), lineage edges, and bridge edges are
   distinguishable from one another.
3. **Bounded traversal.** Multi-hop traversal from a node along typed edges, so a
   provenance chain (`artifact → … → axiom`) can be walked and verified.
4. **Absence queries.** Selection of nodes by the **absence** of an edge of a
   given type. *This is the non-negotiable one.* A substrate that can answer
   "what connects" but not "what fails to connect" cannot express the suspect set,
   and the suspect set is the product. A bastion is a database you query for the
   miss; a backend that cannot query the miss is not a bastion backend.
5. **Provenance-bearing records.** Nodes and edges can carry the metadata the gate
   reads: verbatim source, ratification stamp, authority basis, verdict state
   (`ratified` / `pending` / `retired`).

**Accelerants, not requirements:** vector similarity (for semantic bridging) and
learned structural embeddings (for drift detection) make a bastion *better*, not
*possible*. A bastion can run without either.

Note what the contract does **not** name: a query language, a storage engine,
ArangoDB, AQL, "collection." Any substrate that satisfies 1–5 — a native graph
database, an RDF triplestore, a relational schema with recursive CTEs, even a
typed in-memory structure for a small enough project — can host a bastion.
ArangoDB is one satisfier. HADES is its command-line interface. The choice is an
implementation detail to be **tested against this contract**, not argued.

---

## 9. Both sides of the keyboard — engineering *and* management

The graph is usually described as **context engineering**: it constrains what a
model may generate, rejecting concepts outside the ontology. The same machinery
is **context management** for the humans and agents making decisions — and over a
project's life that is the heavier half.

- **Decisions are artifacts.** A choice — backend, dependency, version, license —
  has its own footprint and can drift, so by the ratification rule it proves on
  its own. It gets a **decision record**, traces to the same axioms, and earns
  ratification — or the recorded verdict **deferred / pending**, which is context
  preserved, not work lost.
- **Non-connection surfaces holes in decisions, not only code.** When a candidate
  cannot trace a *compliant* connection to the ratified requirements — no backend
  that satisfies §8 on the available medium, no license that permits the intended
  use at the intended scale — that **absence is the architectural hole**, surfaced
  before it is built.
- **The method is self-applying.** Using it to build a thing surfaces the holes in
  that thing's architecture, and surfacing them produces the artifacts — decision
  records, specs, the divergence register — that keep those holes closed.

The payoff is symmetric. Context engineering keeps the model inside the ontology;
context management keeps the humans and agents from re-deriving what was already
decided. Both read the same artifacts; both trust non-connection as the signal.

---

## 10. Three instances — the argument from acquisition-path and mode invariance

The method is **demonstrated, not proven universal** (§2). The demonstration is
that the same governance structure appears whether you arrive at it forward from a
founding artifact or backward from a running system. Three instances, three
different routes, one structure:

| Instance | Founding artifact | Acquisition path | What it governs |
|----------|-------------------|------------------|-----------------|
| **NL-Hecate** | The author's papers (Behrouz et al.) — pre-existing | **Forward** from a foundation that already existed | An ab-initio implementation of a research program; code proves against the papers' equations and axioms |
| **WeaverTools** | The HAH hypothesis document — authored deliberately | **Forward** from a foundation authored on purpose | A harness for local coding models; the build proves against the hypothesis |
| **HADES** | *This founding document* — written last | **Backward** from a system that crystallized in situ | The bastion method itself; the most reflexive case — HADES is the tool that *builds* bastions, so its foundation **is** the method |

That a foundation reached three ways yields one structure is evidence the pattern
is **acquisition-path-invariant** — real, not an artifact of how it was built. It
is emphatically **not** a claim of universality: all three fall inside the §2
scope (infrastructure that outlives its makers). The third row is why this
document had to be the genus and not the species: HADES's subject matter is the
method, so a founding document that bottomed out at "an ArangoDB CLI" would have
described the species and lost the genus.

These three also sort onto the two **operating modes** of §5 — and the sort is
*not* the acquisition path. **NL-Hecate** is greenfield: an external, fixed
foundation, code proving forward, the entry gate doing the work. **HADES** and
**WeaverTools** are brownfield: foundation reconciled after drift, the exit gate
doing the work. WeaverTools is the instructive one — its foundation *was*
authored forward (the HAH document), and the `weaver-database` code drifted into
brownfield anyway. **Forward authorship does not immunize against drift; only
continuous proving does** — which is precisely why the reconciliation is a
standing query, not a one-time audit. That a forward-acquired project and a
backward-acquired one both land in brownfield, and the *same* Layer-E machinery
reconciles both, widens the invariance claim from acquisition path to operating
mode.

---

## 11. Reference implementation

HADES is the reference implementation of this method on ArangoDB. Everything
substrate-specific — AQL edge collections, named graphs, `relation_order`,
structural-embedding training (RGCN / inductive GraphSAGE), the embedder service,
the CLI surface — lives **below this line**, in Layer 3, and never leaks up into
the method:

- [`../graph-methodology.md`](../graph-methodology.md) — the five layers and the
  ratification rule as realized on ArangoDB, with the canonical `NL` instance.
- [`../declarative-schema.md`](../../docs/declarative-schema.md) — the `schema apply`
  format (`relation_order`, `feature_dim`, `model_type`).
- [`../codebase-graph-ontology.md`](../../docs/codebase-graph-ontology.md) — the
  `codebase_*` collections and structural edges (Layer D ingest).
- The HADES CLI skill (`~/.claude/skills/hades/`) — the command mechanics.

Quarantining the backend here is what keeps §8's promise honest: Layer 2 must
never silently re-acquire a backend dependency. The directory structure enforces
the discipline the prose claims.

---

## 12. This architecture's own suspect set

Per the method, a founding document ships with its own divergence register —
otherwise it is the self-fulfilling prophecy principle 7 warns against. Running
the suspect-set query (Layer E) over this document's own commitments, six are
**specified here but not yet embodied** in the HADES implementation:

- **C3 — the re-entry path.** The boundary between *ratified context* and *the
  world* is stated, but there is no artifact type yet for an **experiment result
  re-entering** as ratified context. Open.
- **C5 — the friction record.** Adjudication is named as the irreducible human
  act, but the **friction event** (build meets reality) is not yet a modeled
  artifact forcing the amend-or-reject verdict. Open.
- **C7 — the de-ratification rite.** Specified in §4 for the first time. The
  *entry* gate is fully built; the **exit** rite (demote, cascade, re-settle) is
  not yet implemented. The spec leads the implementation here, deliberately. Open
  in code.
- **C10 — the dead ground.** Trust-in-artifact is built, but the canon's scope
  does **not yet reach the build and release pipeline**. The CI/CD provenance is
  outside the graph; the gate has the blind angle the principle warns of. §7
  specifies the remedy — structural-conformance edges and the compiler-enforced
  boundary — but HADES has only semantic bridge edges so far. Open.
- **C11 — wall vs scaffolding.** HADES has no node-role axis: nothing in the graph
  marks a mechanism as **permanent** or **transitional**, and no scaffolding node
  carries a **retirement condition**. The retirement invariant cannot yet be
  queried. Open.
- **C12 — the delta ledger.** The divergence register is read only as a compliance
  finding; the `(deviation → correction)` **dual-read** that would make it a
  dataset — the flywheel — is not modeled. Open.

These are not apologies. They are the method's signature move turned on itself:
the honest holes are what make this document a pre-registration rather than a
prophecy. Each is a candidate for the next architecture session, and each, when
closed, closes by producing the artifact that keeps it closed.

---

## See also

- [`bastion-of-context.md`](bastion-of-context.md) — Layer 1, the philosophical
  foundation (the *why*) this architecture serves.
- [`../graph-methodology.md`](../graph-methodology.md) — Layer 3, the ArangoDB/HADES
  reference implementation (the *how on this backend*).
- [`bastion-playbook.md`](bastion-playbook.md) — the operational playbook (the *how
  in a Claude Code session*) that sequences this architecture into a build.
