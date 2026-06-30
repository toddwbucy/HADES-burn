# Semantic Code Analysis: Tiers, Roadmap, and the Symbol-Identity Contract

**Status:** Roadmap note, awaiting ratification
**Date:** 2026-06-30
**Relates to:** [codebase-graph-ontology.md](codebase-graph-ontology.md), issue #148 (symbol-identity), the `code/` analyzers in `hades-core`

## Purpose

This note records two things. First, the tier of language tooling HADES depends on
and where each candidate language sits in it. Second, the cross-language
**symbol-identity contract** that the tooling landscape forces, so the issue-#148
identity decision is recorded as a general contract rather than a Rust-only fix.

## The line: semantic, not syntactic

HADES needs analyzers that **resolve** names, types, and references, not ones that
only parse syntax. A syntactic parser (tree-sitter, an ast-level CST) can tell you
that a token is a function call. It cannot reliably tell you which function is
called, what type a value has, or which type a method belongs to across files. The
resolved facts are exactly what turn a parse tree into a code graph worth querying:
call edges, type relations, trait or interface implementations, cross-file
references.

So the bar is "rust-analyzer level": a compiler-grade or language-server-grade
analyzer with a consumable API. Tree-sitter is below the bar for graph
construction. It is useful for cheap structural chunking, not for the bridge edges.

One nuance about the current code base. Python's `ast` module and rust-analyzer
are not the same tier. `ast` is syntactic. rust-analyzer is semantic. So HADES's
two existing language paths sit on different rungs today: the Rust path has
semantic enrichment, the Python path is `ast`-only. Bringing Python to true
parity means adding a semantic tool (Pyright or Jedi) on top of `ast`.

## The tiers

| Tier | Languages | Analyzer | Source shape |
|------|-----------|----------|--------------|
| Compiler-as-a-library | C#, TypeScript/JS, Go, C/C++ | Roslyn / the TypeScript compiler API / `go/ast` plus `go/types` plus gopls / libclang or clangd | Often a single tool gives syntax and semantics together |
| Semantic, mature, official | Rust, Java, Swift | rust-analyzer plus `syn` / javac Compiler Tree API or Eclipse JDT / SourceKit or IndexStoreDB | Rust uses two independent tools (the root of #148) |
| Semantic, good, less polished API | Scala, Kotlin, OCaml, Haskell | SemanticDB or Metals / Kotlin analysis API / merlin / GHC API | SemanticDB is a precomputed semantic index format |
| Dynamic, weaker by nature | Ruby, plain JS, Lua | Sorbet, ruby-lsp | Dynamism caps static resolution |

Several languages in the top tier are tooled as well as or better than Rust.
Roslyn and the TypeScript compiler are compilers exposed as libraries. Go's
`go/types` over `go/ast` is in the standard library and is one of the cleanest
semantic toolchains available.

## Single-source versus two-source, and why it drives identity

Many top-tier languages give syntax and semantics from **one** tool: Roslyn, the
TypeScript compiler, libclang, and `go/types` over `go/ast`. One source produces
one set of consistent keys. There is no second analyzer to reconcile.

Rust is the exception that produced issue #148. `syn` (fast structural parse) and
rust-analyzer (semantic enrichment) are independent projects. HADES merges their
output, so the two must agree on a symbol key, or enrichment duplicates a vertex
instead of overwriting it. Python, once a semantic tool is added on top of `ast`,
would reinherit the same two-source reconciliation.

The fragile part in both two-source cases is **reconstructing a qualified-name
string that both analyzers produce identically**. Every language has its own
qualification rules: overload signatures, extension or impl owners, declaration
merging, receiver syntax, template instantiations. Forcing byte-identical
reconstructed names per language is a combinatorial burden.

### The contract

> A symbol's identity is its **location span** (file plus byte or line range), not
> a reconstructed qualified name. The qualified name is derived display metadata,
> not the key.

A span-based identity is language-agnostic. It is identical whether a language
gives one analyzer or two, and it dissolves the two-source agreement constraint
for Rust and Python at once. It also sidesteps the cases that break name-based
keys outright: C++ overloading (same name, different signature), template
instantiations, and out-of-line member definitions whose lexical container is not
their logical owner. This is the contract issue #148 should implement, scoped as a
cross-language decision rather than a Rust patch.

## Roadmap priority

Current footing: Rust at semantic depth (`syn` plus rust-analyzer), Python at
syntactic depth (`ast`, semantic parity pending Pyright or Jedi).

Priority order for new language support, by value and cost in this workspace:

1. **C++ and CUDA (priority).** The workspace has CUDA kernels (NL_Hecate, the
   candle fork) that need semantic-level analysis. See the next section.
2. **Go.** Hermes is Go, and `go/types` over `go/ast` is the cheapest, cleanest
   semantic toolchain of any candidate. High value, low cost, single-source.
3. **Python to parity.** Add Pyright or Jedi over the existing `ast` path so the
   Python graph carries resolved types and references, not only structure.

## C++ and CUDA: the priority lane

The semantic tool is **libclang** (stable C API, Python bindings via
`clang.cindex`) or Clang LibTooling for the full AST, or clangd for incremental
indexing. This is the real Clang frontend, so it resolves types, overloads,
templates, and references at compiler grade.

CUDA is first-class in this lane. Clang has a CUDA frontend, so `.cu` and `.cuh`
files parse with the CUDA language mode and a target architecture. Kernel and
device qualifiers (`__global__`, `__device__`, `__host__`) surface as attributes,
and a kernel launch (`kernel<<<grid, block>>>(args)`) parses as a
`CUDAKernelCallExpr`. Function declarations, types, calls, and templates are all
available semantically.

Two costs to plan for:

- **Compilation database.** Clang needs the exact flags (include paths, defines,
  C++ standard, CUDA architecture) to resolve a translation unit. That means a
  `compile_commands.json`. CMake can emit one with `CMAKE_EXPORT_COMPILE_COMMANDS`.
  An nvcc-driven build may need a capture tool or a clang-as-CUDA-compiler step to
  produce the same database. This is the heavy and fiddly part of the lane, and it
  is per-project, not per-file.
- **Templates and overloading sharpen the identity question.** C++ has overloading
  (same name, different signature) and template instantiations. Name-based keys
  cannot distinguish them. This is the clearest case for the span-based identity
  contract above, and it is why C++ being a priority strengthens that decision
  rather than complicating it.

## Open questions and next steps

- Implement the span-based identity contract in the Rust path first (resolves
  #148), shaped so the Go and C++ paths inherit it rather than re-solving keying.
- Confirm the libclang traversal needed for CUDA kernels on the workspace's actual
  `.cu` and `.cuh` files, and how to source the `compile_commands.json` for the
  NL_Hecate and candle builds.
- Decide the Python semantic tool (Pyright versus Jedi) when the Python path is
  brought to parity.
- Confirm how each analyzer exposes a stable span so the identity contract has a
  concrete key per language.
