# Research Goals

This document describes the research questions HADES-Burn was built to investigate, the role the project plays as experimental apparatus, and its position within the broader Nested Learning research program.

## The Problem

Retrieval-augmented generation (RAG) was originally formulated as a single-shot augmentation: one retrieval per query, feeding a fixed-length context window at inference time. Contemporary agent systems violate that premise at several points simultaneously. An agent working on a non-trivial task issues tens to hundreds of retrievals across a single session; maintains state across many such sessions; operates against corpora (source code, research literature, technical documentation) with structural properties — symbol hierarchies, citation graphs, typed cross-references — that similarity-based retrieval is not designed to surface; and runs, increasingly, on local hardware where the latency budget between inference turns is measured in single-digit milliseconds rather than hundreds.

Three gaps follow from this shift.

First, **retrieval latency becomes a first-class constraint** rather than a deployment detail. A system that issues a hundred retrievals during a task amplifies every millisecond of transport overhead by two orders of magnitude.

Second, **context management is the missing layer between retrieval and generation** — not *what* to retrieve, but what the agent accumulates, reuses, and forgets across long horizons. Standard RAG pipelines treat each retrieval as stateless. Agent workflows are the opposite: the value of a retrieval often depends on what has already been retrieved, committed, or discarded earlier in the session.

Third, **structural properties of the corpus are discarded** when retrieval reduces every document to a position in an embedding space. For corpora where the structure is the content — imports, citations, equation-to-code links, typed cross-references — similarity alone is informationally lossy. Retrieval that can traverse the structure is qualitatively different from retrieval that cannot.

HADES-Burn was built to investigate these three gaps in combination.

## Research Questions

The following are the questions HADES-Burn is positioned to study. They are stated in a form that admits measurement or refutation.

**RQ1. What is the achievable latency floor for graph-grounded retrieval when every transport is a Unix domain socket?**
For local-inference agent workflows, the end-to-end latency of a single retrieval is bounded below by the slowest transport in the chain. HADES-Burn's architecture — Unix sockets for client-to-daemon, daemon-to-database, daemon-to-embedder, and daemon-to-extractor, with no HTTP or TCP hop in the query hot path — is one specific point in the design space. The project enables comparison against HTTP-based RAG baselines under matched workloads and establishes a reference floor for the Unix-socket-based configuration.

**RQ2. What context-management primitives are required for agents to accumulate experience coherently across long horizons?**
An agent operating on a persistent graph over many sessions needs primitives beyond "retrieve top-k." Candidates include provenance-tracked insertion, typed-edge materialization, ontology-grounded chunking that respects semantic boundaries (symbol boundaries in code, equation boundaries in papers), and a closed operation vocabulary that constrains what the agent can request from the graph. HADES-Burn implements a specific set of choices; the research question is which of those primitives load-bear in practice, measured against observable agent behavior — task completion, redundant retrievals, context collapse incidents.

**RQ3. Does ontology-grounded retrieval measurably outperform similarity-only retrieval on research-paper and code corpora?**
Corpora with strong structural properties — citation graphs, symbol-reference graphs, equation-to-implementation links — provide retrieval signal that pure embedding similarity ignores. HADES-Burn's schema stores this structure as typed edges (`cites`, `imports`, `implements`, `references_equation`, and others) materialized from the ingest pipeline and queryable via graph traversal. This enables hybrid retrieval (vector + graph) to be compared head-to-head with vector-only retrieval on identical corpora. Integration of the OG-RAG methodology (arXiv:2412.15235) as an additive ontology-grounded hypergraph layer is analyzed in [docs/og-rag-report.md](docs/og-rag-report.md).

**RQ4. Is end-to-end local retrieval-augmented agent deployment viable on commodity research hardware?**
Most production RAG systems assume a cloud deployment model: hosted embedding APIs, hosted vector databases, hosted inference. For research groups, on-premises teams, and privacy-constrained workflows, there is a live question of whether a fully local stack — local inference, local embedder, local graph store, all communicating over IPC rather than network — is a practical configuration or only a demonstration. HADES-Burn is used daily as such a configuration; the project's operational history is part of the evidence for this question.

## HADES-Burn as Experimental Apparatus

HADES-Burn plays a dual role. It is both a system under study in its own right and the context-management substrate for the Nested Learning research program — serving as the graph back-end for paper ingestion, equation extraction, cross-paper analysis, and the provenance graph that links papers to implementations. This dual role is deliberate and produces measurement points that would be difficult to obtain from a system used only in a benchmark setting.

- **Latency distributions under realistic load.** Every development session issues real retrievals at irregular intervals under the pressure of actual tasks. The daemon collects timing data representative of the workload the system was designed for, not a synthetic benchmark.
- **Agent-memory coherence over long horizons.** The same persistent graph is queried across weeks of development. Divergence, staleness, and redundancy effects become observable over natural timescales rather than simulated ones.
- **Retrieval quality on working corpora.** The project's own source code and the NL paper set are both ingested. Retrieval quality is judged by whether the agent completes useful work, not by offline NDCG@k on a held-out set.
- **AI-assisted development workflow data.** The project has been developed using Claude Code and local model agents as primary implementation tools, with the author in the role of research director. The commit and refactor history is itself data on what retrieval and context primitives make those workflows productive.

## Related Work and Publications

HADES-Burn is positioned within several lines of existing work.

- **The Nested Learning research program** — a multi-paper effort from the Mirrokni research group at Google Research on nested-optimization memory systems and continuous memory. Relevant arXiv identifiers include 2512.24695 (capstone), 2501.00663 (Titans), 2504.13173 (MIRAS), 2505.23735 (Atlas), 2511.07343 (TNT), 2504.05646 (Lattice), and 2512.23852 (Trellis). HADES-Burn is the context substrate for the Rust implementation of this program (NL-Hecate, a sibling project under the same author).
- **Ontology-grounded retrieval.** OG-RAG (arXiv:2412.15235) introduces an ontology-grounded hypergraph retrieval method whose integration as an additive layer over HADES-Burn's existing collections is analyzed in [docs/og-rag-report.md](docs/og-rag-report.md).
- **Graph-based retrieval-augmented generation.** GraphRAG-style work from Microsoft Research and related projects informs the typed-edge materialization choices and the design of structural-plus-semantic hybrid retrieval.
- **Code-aware retrieval.** Symbol-graph approaches from the code-search literature, together with the semantic model exposed by rust-analyzer, inform the AST-level ingestion pipeline.

## Long-Term Vision

HADES-Burn is intended to mature as open research infrastructure for the broader community working on low-latency retrieval, local-LLM deployment, and persistent agent memory. The project is developed in the open under the Apache License, Version 2.0. Collaboration from academic and industry researchers is welcome — particularly work on measured baselines for hybrid-retrieval quality, alternative ontology seeds for domain-specific deployments, and the agent-memory coherence measurement problem, which is currently under-addressed in the published literature.

The aim is not to produce a product. The aim is to produce a credible, reproducible reference configuration for a research question — *what does a fully local, graph-grounded, low-latency retrieval stack enable an agent to do reliably over long horizons* — that the community can then study, modify, and argue with.
