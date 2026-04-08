# OG-RAG: Ontology-Grounded Retrieval-Augmented Generation

## Report for HADES-Burn Integration Discussion

**Source paper**: Sharma, Kumar, Li (Microsoft Research, Dec 2024) — arXiv:2412.15235v1
**Local copy**: `2412.15235v1.pdf`
**Date**: 2026-03-31

---

## 1. The Problem OG-RAG Solves

Standard RAG retrieves text chunks by embedding similarity. This has three weaknesses:

1. **No domain grounding** — chunks are domain-agnostic blobs of text. The retrieval has no understanding of what entities, attributes, or relationships matter in the domain.
2. **Poor attribution** — when an LLM generates a response from text chunks, it's difficult for humans (or other models) to trace which specific facts led to which conclusions.
3. **Weak deductive reasoning** — text chunks don't preserve the relational structure needed for multi-step logical inference. A chunk might contain half of a causal chain, with the other half in a different chunk that wasn't retrieved.

GraphRAG (Edge et al., 2024) partially addresses this by extracting entities/relationships into a knowledge graph, but it relies on ad-hoc LLM extraction without a domain schema — the graph structure is whatever the LLM happens to produce, with no guarantee of consistency or completeness.

## 2. Core Idea

OG-RAG introduces a **domain-specific ontology** as a structural prior on the knowledge graph. Instead of letting the LLM extract arbitrary entities, the ontology defines:

- What **entity types** exist in the domain (e.g., Crop, Seed, Region, Pest)
- What **attributes** each entity type has (e.g., Crop → has name, has growing zone, has sowing time)
- What **relationships** are valid between entity types (e.g., CropGrowingZone → has region → Region)

Documents are then mapped onto this ontology to produce **factual blocks** — self-contained clusters of (subject, attribute, value) triples where every term is grounded in the ontology vocabulary.

## 3. Architecture (Three Phases)

### Phase 1: Ontology-Mapped Data (Preprocessing)

**Input**: Documents D + Domain Ontology O
**Output**: Ontology-mapped data I = D(O) — a set of factual blocks

Process:
1. Define (or learn) a domain ontology O as a set of entity-attribute-value triples: O ⊆ S × A × (S ∪ {φ}), where S = entity set, A = attribute set, φ = unspecified value
2. For each document chunk, use an LLM to map the content onto the ontology, producing JSON-LD with entities and their attribute values grounded in ontology terms
3. Where the ontology specifies a value for (entity, attribute), use it; where it doesn't (v = φ), extract the value from the document text

**Key point**: The ontology provides the *schema*; the documents provide the *data*. The LLM's job is to align document text to ontology terms, not to invent structure from scratch.

**Ontology construction**: The paper uses a semi-automated approach — a proprietary "ontology learning module" generates a baseline, then domain experts review and refine it. They note that many fields already have rich pre-existing ontologies available (e.g., IPTC SNaP for news).

### Phase 2: Hypergraph Construction (Preprocessing)

**Input**: Ontology-mapped data I
**Output**: Hypergraph H(I) = (N, E)

Process:
1. **Flatten** each factual block F into a set of flattened blocks F̄ (Algorithm 1). This recursively resolves nested entity references, producing key-value pairs where the key concatenates the entity chain (s ⊕ a) and the value is either another entity reference or a terminal value.

   Example: A factual block about Soybean cultivation gets flattened to:
   ```
   (Crop ⊕ name, Soybean)
   (Crop ⊕ used_for, Food)
   (Crop ⊕ has_growing_zone ⊕ CropGrowingZone ⊕ name, Madhya Pradesh)
   (Crop ⊕ has_growing_zone ⊕ CropGrowingZone ⊕ has_region, Indore, Bhopal)
   (Crop ⊕ has_growing_zone ⊕ CropGrowingZone ⊕ has_seed ⊕ ... ⊕ seeding_rate, 60-80kg/ha)
   ```

2. Each flattened key-value pair becomes a **hypernode** n ∈ N
3. Each original factual block (the set of related hypernodes) becomes a **hyperedge** e ∈ E

**Key insight**: A hyperedge connects *multiple* nodes simultaneously (not just pairs), capturing multi-dimensional relationships. The hyperedge "Soybean cultivation in Madhya Pradesh" bundles the crop name, region, seed variety, seeding rate, and growing conditions as a single retrievable unit.

**Complexity**: O(|I| · |O| · log|O|) worst case for maximum nesting; O(|I|) for flat structures. Storage is proportional to number of hyperedges.

### Phase 3: Hypergraph-Based Retrieval (Query Time)

**Input**: Query Q, Hypergraph H(I), Embedding function Z, top-k, max hyperedges L
**Output**: Minimal context C_H(Q) — a set of up to L hyperedges covering the query-relevant facts

Process (Algorithm 2):
1. Embed all hypernodes (both keys and values) using sentence embedding function Z
2. Find **N_S(Q)**: top-k hypernodes by similarity between their key (s ⊕ a) and the query Q
3. Find **N_V(Q)**: top-k hypernodes by similarity between their value (v) and the query Q
4. Union: N(Q) = N_S(Q) ∪ N_V(Q) — gives 2k candidate hypernodes
5. **Greedy set-cover**: repeatedly select the hyperedge that covers the most uncovered candidate hypernodes, until all candidates are covered or L hyperedges are selected
6. Return the selected hyperedges as structured context

**The set-cover is the key optimization.** Instead of returning all hyperedges that contain any relevant hypernode (which could be noisy and redundant), it finds the *minimum* set that covers all relevant facts. This is a matroid-constrained linear optimization, solvable greedily with provable approximation guarantees (Korte et al., 2011).

**No additional LLM calls at query time.** The retrieval is pure embedding similarity + combinatorial optimization. The LLM is only called once at the end to generate the response from the retrieved context.

## 4. Results

Evaluated on agriculture domain (Soybean, Wheat — 85 expert-curated documents) and news domain (149 long-form articles). Compared against vanilla RAG, RAPTOR, and GraphRAG.

### Query Answering (Table 1 & 2)
| Metric | OG-RAG vs RAG | OG-RAG vs RAPTOR | OG-RAG vs GraphRAG |
|--------|---------------|------------------|--------------------|
| Context Recall | +55% avg | +30% avg | +40% avg |
| Context Entity Recall | +110% avg | +50% avg | +70% avg |
| Answer Correctness | +40% avg | +10% avg | +15% avg |
| Answer Relevance | +16% avg | +5% avg | comparable |

### Efficiency (Table 3)
| Method | T_pre (sec) | T_query (sec) |
|--------|-------------|---------------|
| RAG | 11 | 2.5 |
| RAPTOR | 72 | 4.8 |
| GraphRAG | 157 | 5.7 |
| **OG-RAG** | **30** | **3.8** |

OG-RAG preprocessing is 2-5x faster than RAPTOR/GraphRAG. Query time adds only ~1.3 seconds over vanilla RAG while delivering dramatically better accuracy.

### Context Attribution (Table 4, human study)
- OG-RAG contexts verified **29% faster** by human evaluators
- OG-RAG contexts rated **30% higher** for factual support (3.46 vs 2.67 on 1-5 scale)

### Deductive Reasoning (Table 5)
When given a fixed set of deductive rules (e.g., "1 kg of herbicide produces 18-27 kg CO2e"), OG-RAG contexts enabled LLMs to correctly apply multi-step reasoning significantly better than all baselines. This is because the ontology-grounded structure preserves the entity relationships needed for logical chaining.

## 5. Strengths and Limitations

### Strengths
- **Domain grounding** eliminates ad-hoc entity extraction — the ontology constrains what gets extracted
- **Hyperedge bundling** keeps related facts together, avoiding the "split fact" problem of text chunking
- **Greedy set-cover** produces compact, non-redundant context
- **Attribution** is trivial — each fact in the response maps to a hyperedge, which maps to source documents
- **No LLM calls at query time** — retrieval is pure compute (embeddings + set-cover)
- **Preprocessing efficiency** — significantly faster than RAPTOR and GraphRAG

### Limitations
- **Ontology construction** — requires domain expertise or a separate learning step. The paper uses a proprietary module + expert review. This is the main bottleneck for new domains.
- **Broad retrieval scope** — the paper notes that OG-RAG sometimes retrieves too much context (Tables 2, 5 show occasional Answer Relevance dips), suggesting the set-cover can over-include
- **Static ontology** — the paper doesn't address ontology evolution as new documents introduce new concepts
- **Unstructured text only** — the paper evaluates on agriculture docs and news articles. No code, no structured data sources

## 6. Relevance to HADES-Burn

### What HADES already has that maps to OG-RAG

| OG-RAG Concept | HADES Equivalent | Status |
|----------------|------------------|--------|
| Ontology entities | `codebase_symbols` (kind, name, signature, visibility) | Exists |
| Ontology relationships | `codebase_edges` (defines, calls, implements) | Exists |
| Factual blocks | Symbol + its edges (a function with its call targets, file, signature) | Exists implicitly |
| Embedding infrastructure | Jina V4 + ArangoDB FAISS ANN | Exists |
| Document extraction | Persephone gRPC | Exists |

### What would need to be added

1. **Ontology schema collection** — Formal definition of entity types, attribute types, and valid relationships per domain. Small, mostly static. Defines the "shape" of knowledge for each domain (academic papers, codebases, mixed).

2. **Ontology mapping step** — Post-extraction processing that maps raw extracted content onto the ontology schema:
   - For **code**: Largely mechanical. AST extraction already produces typed symbols with attributes and relationships. The "mapping" is a structured transformation, not an LLM call.
   - For **papers**: Requires an LLM prompt (via Persephone) to map abstract/body text onto the academic ontology. The paper's Appendix B.1 shows the prompt template — it asks the LLM to produce JSON-LD mapping document content to ontology terms.

3. **Hypernode collection** — Flattened key-value pairs from the factual blocks. Each hypernode stores:
   - `key`: concatenated entity-attribute path (e.g., "Function ⊕ calls")
   - `value`: terminal value (e.g., "verify_token")
   - Source reference back to the originating document/symbol

4. **Hypernode embeddings** — Embeddings of both keys and values using existing Jina V4 infrastructure. Two vectors per hypernode (or one concatenated).

5. **Hyperedge (fact cluster) collection** — Each document references the set of hypernode keys it bundles. This is the unit of retrieval.

6. **OG-Retrieve query path** — New retrieval function alongside existing `similarity_search`:
   - Embed query
   - Top-k hypernode search (two passes: match on key, match on value)
   - Greedy set-cover to select minimum covering hyperedges
   - Return structured fact clusters

### Integration approach: Additive, not destructive

The OG-RAG layer can be implemented as **new collections and a post-processing pipeline step** without modifying any existing collections, schemas, or pipeline stages:

```
Current:  Extract → Chunk → Embed → Store
Proposed: Extract → Chunk → Embed → Store → [Ontology Map] → [Cluster]
                                               ↑ new step      ↑ new step
```

The ontology mapping reads from existing collections (symbols, edges, chunks) and writes to new collections (hypernodes, hypernode_embeddings, fact_clusters). The existing retrieval path stays untouched; OG-Retrieve is an additional query mode.

### The code domain advantage

The paper's biggest limitation — ontology construction requiring domain experts — is largely irrelevant for code. Programming languages *are* formal ontologies:

- Entity types = language constructs (function, struct, trait, module, import)
- Attributes = syntactic properties (name, visibility, signature, return type, line span)
- Relationships = structural connections (defines, calls, implements, imports)

The AST parser (`syn`, `rustpython-parser`, `rust-analyzer`) already extracts all of this. For code, the "ontology" is the language grammar, and the "ontology mapping" is the AST analysis — both of which HADES already does natively.

For academic papers, an ontology would need to be defined (Paper, Method, Dataset, Result, Citation, Author, Venue) and an LLM mapping step added. This is the real new work.

### Latency considerations for Hecate

OG-RAG's query-time performance (3-4 seconds in Python) is dominated by embedding computation and vector search. In HADES's architecture:

- Embedding: Jina V4 via Unix socket (already optimized)
- Vector search: ArangoDB FAISS ANN via Unix socket (already optimized)
- Set-cover: ~15 lines of greedy algorithm, pure CPU, microsecond-scale in Rust

The retrieval path for Hecate would be: query → embed (1 call) → 2 ANN searches → set-cover → return structured facts. This should be achievable in low double-digit milliseconds, making it viable as a tool inside a reasoning loop rather than a one-shot context injection.

## 7. Key Algorithms (Reference)

### Algorithm 1: Flatten a Factual Block

```
FLATTEN(F):
  F̄ = {}
  F̄_0 = {(s ⊕ a, v) : (s,a,v) ∈ F, v ∈ V, (s',a',s) ∉ F}  // no dependencies
  F̄ = F̄ ∪ {F̄_0}
  for (s, a, s') ∈ F \ F̄_0 do
    if s' ∈ S then                           // s' is another entity
      F̄_{s'} = F̄_0 ∪ {(s ⊕ a ⊕ s' ⊕ a', v') : (s',a',v') ∈ F}
      F̄ = F̄ ∪ FLATTEN(F̄_{s'})             // recurse
  return F̄
```

### Algorithm 2: OG-Retrieve

```
OG-RETRIEVE(Q, H(I), Z, k, L):
  N, E = nodes and edges of H(I)
  N_S(Q) = top-k argmax_{(s,a,v) ∈ N} ⟨Z(s ⊕ a), Z(Q)⟩    // match on key
  N_V(Q) = top-k argmax_{(s,a,v) ∈ N} ⟨Z(v), Z(Q)⟩          // match on value
  N(Q) = N_S(Q) ∪ N_V(Q)                                     // 2k candidates
  C_H(Q) = {}
  while |N(Q)| > 0 and |C_H(Q)| < L do
    C_H(Q) = C_H(Q) ∪ argmax_{e ∈ E} |{n ∈ N(Q) : n ∈ e}|  // greediest edge
    remove covered nodes from N(Q)
  return C_H(Q)
```

## 8. Open Questions for Discussion

1. **Ontology granularity for code**: Should the code ontology be language-specific (Rust ontology vs Python ontology) or unified with language-agnostic entity types?

2. **Cross-domain edges**: How should Paper → Code links work? (e.g., "Paper X describes Method Y which is implemented in Function Z") These bridge the academic and codebase ontologies.

3. **Ontology evolution**: When new code constructs or paper topics appear, how does the ontology grow? Manual curation, or LLM-assisted expansion?

4. **Hyperedge granularity**: For code, is one hyperedge per function the right level? Or should it be per-module? Per-call-chain? The paper doesn't deeply explore this tradeoff.

5. **Dual embedding strategy**: The paper embeds keys and values separately. For Jina V4 with its 2048-dim vectors, this means 2x the embedding storage for hypernodes. Is the dual search worth the cost, or could a single concatenated embedding suffice?

6. **Integration with existing chunking**: Should OG-RAG replace text chunking for retrieval, or complement it? The paper positions it as a replacement, but there may be queries where raw text chunks are more appropriate (e.g., "show me the code around line 42").

7. **Phase sequencing**: Where does this fit in the HADES-Burn phase plan? After the current Phase 3 code analysis work? As part of Phase 4 (prefetcher)? As a separate phase?

---

*Paper reference: Sharma, K., Kumar, P., & Li, Y. (2024). OG-RAG: Ontology-Grounded Retrieval-Augmented Generation for Large Language Models. arXiv:2412.15235v1.*
