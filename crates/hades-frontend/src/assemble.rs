//! The coupling layer: assemble a [`GraphSnapshot`] by shelling out to `hades`.
//!
//! This is the *entire* seam between the frontend and the backend. It depends on
//! three output contracts only:
//!   * `hades <db> db graph list`   -> enveloped JSON, `.data.graphs[]` with
//!     `edge_definitions[].{collection,from[],to[]}`.
//!   * `hades <db> db collections`  -> enveloped JSON, `.data.collections[]` with
//!     `{name,count}` (used for honest "X of Y" slice totals).
//!   * `hades <db> db export <col>` -> raw **jsonl**, one bare document per line
//!     (NOT enveloped — it streams). Nodes carry `_id`; edges add `_from`/`_to`.
//!
//! Nothing here links `hades-core`. The I/O (`Backend`, `run`) is kept thin and
//! separate from the pure transformation (`parse_*`, `discover_collections`,
//! `build_snapshot`) so the logic is unit-testable against fixtures with no live
//! `hades` — see the tests at the bottom. That separation is the price of loose
//! coupling paid back as testability.

use crate::contract::{
    label_for, ChunkText, Edge, GraphDelta, GraphSnapshot, Node, SnapshotMeta, INTERNAL_FIELDS,
};
use anyhow::{anyhow, Context, Result};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use tokio::process::Command;

/// A parsed ArangoDB document.
type Doc = serde_json::Map<String, serde_json::Value>;

/// Discover every database the hades user can access (`db databases` run against
/// `_system`). Used when `serve` is launched without an explicit `--db` list, so
/// the viewer offers exactly the databases ArangoDB grants — the ACL layer is the
/// authority, matching HADES's own security model.
pub async fn discover_databases(bin: &str) -> Result<Vec<String>> {
    let output = Command::new(bin)
        .args(["--db", "_system", "db", "databases"])
        .output()
        .await
        .with_context(|| format!("failed to spawn `{bin}`"))?;
    if !output.status.success() {
        return Err(anyhow!(
            "`{bin} --db _system db databases` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    parse_database_list(&output.stdout)
}

/// Parse `db databases` output (`.data.databases[]`).
pub fn parse_database_list(bytes: &[u8]) -> Result<Vec<String>> {
    let v: serde_json::Value =
        serde_json::from_slice(bytes).context("db databases: invalid JSON envelope")?;
    let arr = v
        .get("data")
        .and_then(|d| d.get("databases"))
        .and_then(|x| x.as_array())
        .ok_or_else(|| anyhow!("db databases: missing .data.databases"))?;
    Ok(arr.iter().filter_map(|s| s.as_str().map(String::from)).collect())
}

/// How to invoke the backend CLI. `bin` defaults to `hades` on PATH; `database`
/// is threaded as the global `--db` flag before every subcommand.
#[derive(Clone)]
pub struct Backend {
    pub bin: String,
    pub database: String,
    /// Optional per-collection export cap. `Some(n)` bounds each collection to
    /// `n` documents and marks the snapshot `partial`; `None` exports fully.
    pub limit: Option<u32>,
}

// ===================== I/O (thin) =====================

impl Backend {
    /// Run `hades --db <database> <args...>` and return captured stdout.
    async fn run(&self, args: &[&str]) -> Result<Vec<u8>> {
        let output = Command::new(&self.bin)
            .arg("--db")
            .arg(&self.database)
            .args(args)
            .output()
            .await
            .with_context(|| format!("failed to spawn `{}`", self.bin))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!(
                "`{} --db {} {}` failed ({}): {}",
                self.bin,
                self.database,
                args.join(" "),
                output.status,
                stderr.trim()
            ));
        }
        Ok(output.stdout)
    }

    /// List named graphs available in the database (via the Gharial-backed
    /// `db graph list`), resolved to their edge topology.
    pub async fn list_graphs(&self) -> Result<Vec<GraphInfo>> {
        parse_graph_list(&self.run(&["db", "graph", "list"]).await?)
    }

    /// Fetch document counts per collection (`db collections`).
    async fn collection_counts(&self) -> Result<BTreeMap<String, usize>> {
        parse_collection_counts(&self.run(&["db", "collections"]).await?)
    }

    /// Expand one node's immediate neighborhood (depth 1) via `db graph
    /// neighbors`. Returns a delta for the viewer to merge. ~10ms per call, so
    /// interactive click-to-expand needs no warm daemon at this scale.
    ///
    /// `node`/`graph_name`/`direction` arrive from an unauthenticated HTTP query
    /// on a LAN-exposed server, so they are hardened against argv flag-smuggling
    /// (see [`reject_flaglike`]): a value like `--database=NL` in the `node`
    /// positional would otherwise override the pinned `--db` global and escape
    /// the one-database containment. The untrusted positional is placed after a
    /// `--` argv separator and leading-dash values are rejected outright;
    /// `direction` is whitelisted.
    pub async fn neighbors(
        &self,
        graph_name: &str,
        node: &str,
        direction: &str,
        limit: u32,
    ) -> Result<GraphDelta> {
        reject_flaglike("graph", graph_name)?;
        reject_flaglike("node", node)?;
        if !matches!(direction, "any" | "inbound" | "outbound") {
            return Err(anyhow!(
                "invalid direction `{direction}` (want any|inbound|outbound)"
            ));
        }
        let limit_s = limit.to_string();
        let out = self
            .run(&[
                "db", "graph", "neighbors", "--graph", graph_name, "-d", direction, "-n", &limit_s,
                // `--` ends option parsing; the untrusted node is a bare positional
                // after it, so it can never be reinterpreted as a flag.
                "--", node,
            ])
            .await?;
        parse_neighbors(&out)
    }

    /// Fetch a single node by its full `_id` (`collection/key`) via `db get`.
    /// Lets a deep link resolve a node that isn't in the currently loaded slice.
    pub async fn fetch_node(&self, id: &str) -> Result<Option<Node>> {
        reject_flaglike("id", id)?;
        let (collection, key) = id
            .split_once('/')
            .ok_or_else(|| anyhow!("invalid node id `{id}` (want collection/key)"))?;
        reject_flaglike("collection", collection)?;
        reject_flaglike("key", key)?;
        let out = self.run(&["db", "get", "--", collection, key]).await?;
        parse_single_node(&out)
    }

    /// Fetch the source text associated with a file, ordered by chunk index.
    ///
    /// This is codebase-schema-aware: text lives in `codebase_chunks` linked by
    /// `file_key` (files/symbols carry no inline text). Uses a FIXED AQL query
    /// with a bind variable for `file_key`, so the untrusted value is escaped by
    /// ArangoDB, never string-interpolated — no AQL injection. Returns an empty
    /// vec for graphs without this schema.
    pub async fn file_content(&self, file_key: &str) -> Result<Vec<ChunkText>> {
        reject_flaglike("file_key", file_key)?;
        let bind = serde_json::json!({ "fk": file_key }).to_string();
        let out = self
            .run(&[
                "db",
                "aql",
                "FOR c IN codebase_chunks FILTER c.file_key == @fk \
                 SORT c.chunk_index ASC RETURN { chunk_index: c.chunk_index, text: c.text }",
                "-b",
                &bind,
            ])
            .await?;
        parse_chunk_texts(&out)
    }

    /// Stream a collection's documents (raw jsonl) into parsed objects.
    async fn export_collection(&self, collection: &str) -> Result<Vec<Doc>> {
        let mut args = vec!["db", "export", collection];
        let limit_str;
        if let Some(n) = self.limit {
            limit_str = n.to_string();
            args.push("-n");
            args.push(&limit_str);
        }
        let text = String::from_utf8(self.run(&args).await?).context("db export: non-UTF8 output")?;
        parse_jsonl(&text, collection)
    }

    /// Assemble a full snapshot for one named graph: discover its collections,
    /// export them, learn their true sizes, then build the payload (pure).
    pub async fn assemble(&self, graph_name: &str) -> Result<GraphSnapshot> {
        let graph = self
            .list_graphs()
            .await?
            .into_iter()
            .find(|g| g.name == graph_name)
            .ok_or_else(|| anyhow!("named graph `{graph_name}` not found in {}", self.database))?;

        let (edge_collections, node_collections) = discover_collections(&graph);

        let mut node_docs = BTreeMap::new();
        for col in &node_collections {
            node_docs.insert(col.clone(), self.export_collection(col).await?);
        }
        let mut edge_docs = BTreeMap::new();
        for col in &edge_collections {
            edge_docs.insert(col.clone(), self.export_collection(col).await?);
        }

        // True totals (pre-limit) so the viewer can honestly say "X of Y".
        let counts = self.collection_counts().await?;
        let node_total = sum_counts(&counts, &node_collections);
        let edge_total = sum_counts(&counts, &edge_collections);

        Ok(build_snapshot(
            &self.database,
            graph_name,
            node_collections,
            edge_collections,
            node_docs,
            edge_docs,
            node_total,
            edge_total,
            self.limit,
        ))
    }
}

// ===================== pure transformation =====================

/// A named graph and its resolved edge topology.
#[derive(Debug, Clone)]
pub struct GraphInfo {
    pub name: String,
    pub edge_collections: Vec<EdgeTopology>,
}

/// One edge collection and the vertex collections it connects.
#[derive(Debug, Clone)]
pub struct EdgeTopology {
    pub collection: String,
    pub from: Vec<String>,
    pub to: Vec<String>,
}

/// Parse `db graph list` output into named graphs with edge topology.
pub fn parse_graph_list(bytes: &[u8]) -> Result<Vec<GraphInfo>> {
    let v: serde_json::Value =
        serde_json::from_slice(bytes).context("db graph list: invalid JSON envelope")?;
    let graphs = v
        .get("data")
        .and_then(|d| d.get("graphs"))
        .and_then(|g| g.as_array())
        .ok_or_else(|| anyhow!("db graph list: missing .data.graphs"))?;
    Ok(graphs
        .iter()
        .filter_map(|g| {
            Some(GraphInfo {
                name: g.get("name")?.as_str()?.to_string(),
                edge_collections: g
                    .get("edge_definitions")
                    .and_then(|e| e.as_array())
                    .map(|defs| collect_topology(defs))
                    .unwrap_or_default(),
            })
        })
        .collect())
}

/// Parse `db collections` output into a name -> count map.
pub fn parse_collection_counts(bytes: &[u8]) -> Result<BTreeMap<String, usize>> {
    let v: serde_json::Value =
        serde_json::from_slice(bytes).context("db collections: invalid JSON envelope")?;
    let cols = v
        .get("data")
        .and_then(|d| d.get("collections"))
        .and_then(|c| c.as_array())
        .ok_or_else(|| anyhow!("db collections: missing .data.collections"))?;
    Ok(cols
        .iter()
        .filter_map(|c| {
            Some((
                c.get("name")?.as_str()?.to_string(),
                c.get("count")?.as_u64()? as usize,
            ))
        })
        .collect())
}

/// Parse `db graph neighbors` output (enveloped JSON, `.data.results[]` of
/// `{edge, vertex}`) into a merge-ready delta. The edge/vertex collections are
/// derived from their `_id` (`collection/key`).
pub fn parse_neighbors(bytes: &[u8]) -> Result<GraphDelta> {
    let v: serde_json::Value =
        serde_json::from_slice(bytes).context("db graph neighbors: invalid JSON envelope")?;
    let results = v
        .get("data")
        .and_then(|d| d.get("results"))
        .and_then(|r| r.as_array())
        .ok_or_else(|| anyhow!("db graph neighbors: missing .data.results"))?;

    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for item in results {
        if let Some(vtx) = item.get("vertex").and_then(|x| x.as_object())
            && let Some(node) = node_from_doc(vtx.clone())
                && seen.insert(node.id.clone()) {
                    nodes.push(node);
                }
        if let Some(edg) = item.get("edge").and_then(|x| x.as_object())
            && let Some(edge) = edge_from_doc(edg.clone()) {
                edges.push(edge);
            }
    }
    Ok(GraphDelta { nodes, edges })
}

/// Parse `db get` output (enveloped, `.data` is the document) into a [`Node`].
pub fn parse_single_node(bytes: &[u8]) -> Result<Option<Node>> {
    let v: serde_json::Value =
        serde_json::from_slice(bytes).context("db get: invalid JSON envelope")?;
    Ok(v.get("data")
        .and_then(|d| d.as_object())
        .cloned()
        .and_then(node_from_doc))
}

/// Parse `db aql` content output (enveloped, `.data.results` is the array) into
/// ordered chunk texts.
pub fn parse_chunk_texts(bytes: &[u8]) -> Result<Vec<ChunkText>> {
    let v: serde_json::Value =
        serde_json::from_slice(bytes).context("db aql: invalid JSON envelope")?;
    let rows = v
        .get("data")
        .and_then(|d| d.get("results"))
        .and_then(|r| r.as_array())
        .ok_or_else(|| anyhow!("db aql: missing .data.results"))?;
    Ok(rows
        .iter()
        .filter_map(|r| {
            Some(ChunkText {
                // chunk_index may arrive as a number or a numeric string.
                chunk_index: r
                    .get("chunk_index")
                    .and_then(|c| c.as_u64().or_else(|| c.as_str().and_then(|s| s.parse().ok())))
                    .unwrap_or(0) as u32,
                text: r.get("text")?.as_str()?.to_string(),
            })
        })
        .collect())
}

/// Parse raw jsonl (one document per line) into objects. `collection` is only
/// used to enrich error messages.
pub fn parse_jsonl(text: &str, collection: &str) -> Result<Vec<Doc>> {
    let mut docs = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let doc: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("db export {collection}: bad jsonl at line {}", i + 1))?;
        match doc {
            serde_json::Value::Object(m) => docs.push(m),
            _ => return Err(anyhow!("db export {collection}: line {} is not an object", i + 1)),
        }
    }
    Ok(docs)
}

/// Resolve `(edge_collections, node_collections)` a named graph spans. Edge
/// collections are named directly; vertex collections are the union of every
/// edge's from/to sets. Both are de-duplicated and ordered.
pub fn discover_collections(graph: &GraphInfo) -> (Vec<String>, Vec<String>) {
    let edge: Vec<String> = graph
        .edge_collections
        .iter()
        .map(|e| e.collection.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let node: Vec<String> = graph
        .edge_collections
        .iter()
        .flat_map(|e| e.from.iter().chain(e.to.iter()).cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    (edge, node)
}

/// Build a [`GraphSnapshot`] from already-fetched documents. Pure: no I/O, so it
/// is exercised directly by the unit tests.
#[allow(clippy::too_many_arguments)]
pub fn build_snapshot(
    database: &str,
    graph_name: &str,
    node_collections: Vec<String>,
    edge_collections: Vec<String>,
    node_docs: BTreeMap<String, Vec<Doc>>,
    edge_docs: BTreeMap<String, Vec<Doc>>,
    node_total: usize,
    edge_total: usize,
    limit: Option<u32>,
) -> GraphSnapshot {
    let mut attribute_index: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut nodes: Vec<Node> = Vec::new();
    let mut edges: Vec<Edge> = Vec::new();

    for col in &node_collections {
        let mut keys: BTreeSet<String> = BTreeSet::new();
        for mut doc in node_docs.get(col).cloned().unwrap_or_default() {
            let Some(id) = take_str(&doc, "_id") else {
                continue; // a vertex with no _id cannot be referenced by edges
            };
            let key = take_str(&doc, "_key").unwrap_or_default();
            let label = label_for(&doc, &key);
            strip_internal(&mut doc, &mut keys);
            nodes.push(Node { id, collection: col.clone(), label, degree: 0, attrs: doc });
        }
        attribute_index.insert(col.clone(), keys.into_iter().collect());
    }

    let present: BTreeSet<&str> = nodes.iter().map(|n| n.id.as_str()).collect();

    for col in &edge_collections {
        let mut keys: BTreeSet<String> = BTreeSet::new();
        for mut doc in edge_docs.get(col).cloned().unwrap_or_default() {
            let id = take_str(&doc, "_id").unwrap_or_default();
            let (Some(source), Some(target)) = (take_str(&doc, "_from"), take_str(&doc, "_to"))
            else {
                continue; // malformed edge
            };
            // Skip edges whose endpoints are outside the fetched node set, so the
            // graph handed to the viewer never dangles.
            if !present.contains(source.as_str()) || !present.contains(target.as_str()) {
                continue;
            }
            strip_internal(&mut doc, &mut keys);
            edges.push(Edge { id, source, target, collection: col.clone(), attrs: doc });
        }
        attribute_index.insert(col.clone(), keys.into_iter().collect());
    }

    // Undirected degree within this payload.
    let mut degree: HashMap<&str, u32> = HashMap::new();
    for e in &edges {
        *degree.entry(e.source.as_str()).or_default() += 1;
        *degree.entry(e.target.as_str()).or_default() += 1;
    }
    for n in &mut nodes {
        n.degree = degree.get(n.id.as_str()).copied().unwrap_or(0);
    }

    let node_count = nodes.len();
    let edge_count = edges.len();
    // A slice is partial when a cap was set AND fewer items came back than the
    // collections actually hold.
    let partial = limit.is_some() && (node_count < node_total || edge_count < edge_total);

    GraphSnapshot {
        meta: SnapshotMeta {
            database: database.to_string(),
            named_graph: graph_name.to_string(),
            source: "snapshot",
            partial,
            node_collections,
            edge_collections,
            attribute_index,
            node_count,
            edge_count,
            node_total,
            edge_total,
        },
        nodes,
        edges,
    }
}

// ===================== small helpers =====================

fn collect_topology(defs: &[serde_json::Value]) -> Vec<EdgeTopology> {
    defs.iter()
        .filter_map(|d| {
            Some(EdgeTopology {
                collection: d.get("collection")?.as_str()?.to_string(),
                from: str_array(d.get("from")),
                to: str_array(d.get("to")),
            })
        })
        .collect()
}

fn str_array(v: Option<&serde_json::Value>) -> Vec<String> {
    v.and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect())
        .unwrap_or_default()
}

fn sum_counts(counts: &BTreeMap<String, usize>, collections: &[String]) -> usize {
    collections.iter().map(|c| counts.get(c).copied().unwrap_or(0)).sum()
}

/// Reject a value that could be smuggled as a CLI flag. A leading `-` (even after
/// the `--` separator, defense in depth) or an empty value is refused before the
/// value ever reaches `Command`. Prevents an HTTP caller from injecting argv flags
/// such as `--database=other` that would escape the pinned-database containment.
fn reject_flaglike(field: &str, value: &str) -> Result<()> {
    if value.is_empty() {
        return Err(anyhow!("empty `{field}` value"));
    }
    if value.starts_with('-') {
        return Err(anyhow!("invalid `{field}` value `{value}`: leading dash not allowed"));
    }
    Ok(())
}

/// The collection portion of an ArangoDB `_id` (`collection/key`).
fn collection_of(id: &str) -> &str {
    id.split_once('/').map(|(c, _)| c).unwrap_or("")
}

/// Build a [`Node`] from a raw document, deriving its collection from `_id`.
/// Used by neighborhood expansion, where the collection isn't known up front.
fn node_from_doc(mut doc: Doc) -> Option<Node> {
    let id = take_str(&doc, "_id")?;
    let key = take_str(&doc, "_key").unwrap_or_default();
    let collection = collection_of(&id).to_string();
    let label = label_for(&doc, &key);
    strip_internal(&mut doc, &mut BTreeSet::new());
    Some(Node { id, collection, label, degree: 0, attrs: doc })
}

/// Build an [`Edge`] from a raw document, deriving its collection from `_id`.
fn edge_from_doc(mut doc: Doc) -> Option<Edge> {
    let id = take_str(&doc, "_id")?;
    let source = take_str(&doc, "_from")?;
    let target = take_str(&doc, "_to")?;
    let collection = collection_of(&id).to_string();
    strip_internal(&mut doc, &mut BTreeSet::new());
    Some(Edge { id, source, target, collection, attrs: doc })
}

/// Read a string field without removing it.
fn take_str(doc: &Doc, field: &str) -> Option<String> {
    doc.get(field).and_then(|v| v.as_str()).map(String::from)
}

/// Remove ArangoDB-internal fields from `attrs` and record the remaining keys.
fn strip_internal(doc: &mut Doc, keys: &mut BTreeSet<String>) {
    for f in INTERNAL_FIELDS {
        doc.remove(*f);
    }
    for k in doc.keys() {
        keys.insert(k.clone());
    }
}

// ===================== tests =====================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn docs(v: serde_json::Value) -> Vec<Doc> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_object().unwrap().clone())
            .collect()
    }

    #[test]
    fn parse_graph_list_extracts_topology() {
        let bytes = serde_json::to_vec(&json!({
            "data": { "graphs": [
                { "name": "g1", "edge_definitions": [
                    { "collection": "defines", "from": ["files"], "to": ["symbols"] }
                ]},
                { "name": "g2", "edge_definitions": [] }
            ]}
        })).unwrap();
        let graphs = parse_graph_list(&bytes).unwrap();
        assert_eq!(graphs.len(), 2);
        assert_eq!(graphs[0].name, "g1");
        assert_eq!(graphs[0].edge_collections[0].collection, "defines");
        assert_eq!(graphs[0].edge_collections[0].from, ["files"]);
    }

    #[test]
    fn discover_collections_unions_and_dedupes() {
        let g = GraphInfo {
            name: "g".into(),
            edge_collections: vec![
                EdgeTopology { collection: "calls".into(), from: vec!["symbols".into()], to: vec!["symbols".into()] },
                EdgeTopology { collection: "defines".into(), from: vec!["files".into()], to: vec!["symbols".into()] },
            ],
        };
        let (edges, nodes) = discover_collections(&g);
        assert_eq!(edges, ["calls", "defines"]);
        assert_eq!(nodes, ["files", "symbols"]); // deduped + sorted
    }

    #[test]
    fn parse_jsonl_skips_blank_and_reads_all_fields() {
        let text = "{\"_id\":\"a/1\",\"name\":\"x\"}\n\n{\"_id\":\"a/2\",\"name\":\"y\"}\n";
        let out = parse_jsonl(text, "a").unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[1]["name"], "y");
    }

    #[test]
    fn parse_jsonl_reports_bad_line() {
        let err = parse_jsonl("{\"ok\":1}\nnot json\n", "c").unwrap_err().to_string();
        assert!(err.contains("line 2"), "got: {err}");
    }

    #[test]
    fn build_snapshot_wires_labels_degrees_attrs_and_strips_internals() {
        let node_docs = BTreeMap::from([
            ("files".to_string(), docs(json!([
                { "_id": "files/a", "_key": "a", "_rev": "r1", "path": "a.rs", "status": "PROCESSED" }
            ]))),
            ("symbols".to_string(), docs(json!([
                { "_id": "symbols/s", "_key": "s", "name": "Foo", "kind": "struct" }
            ]))),
        ]);
        let edge_docs = BTreeMap::from([
            ("defines".to_string(), docs(json!([
                { "_id": "defines/e", "_from": "files/a", "_to": "symbols/s", "sym": "Foo" }
            ]))),
        ]);

        let snap = build_snapshot(
            "db", "g",
            vec!["files".into(), "symbols".into()],
            vec!["defines".into()],
            node_docs, edge_docs,
            2, 1, None,
        );

        assert_eq!(snap.meta.node_count, 2);
        assert_eq!(snap.meta.edge_count, 1);
        assert!(!snap.meta.partial);

        // Label prefers `path`/`name`; internal fields stripped from attrs.
        let file = snap.nodes.iter().find(|n| n.id == "files/a").unwrap();
        assert_eq!(file.label, "a.rs");
        assert_eq!(file.degree, 1);
        assert!(!file.attrs.contains_key("_id") && !file.attrs.contains_key("_rev"));
        assert_eq!(file.attrs["status"], "PROCESSED");

        // attribute_index is the union of surviving keys per collection.
        assert_eq!(snap.meta.attribute_index["files"], ["path", "status"]);
        assert_eq!(snap.meta.attribute_index["defines"], ["sym"]);

        // Edge promoted to source/target; _from/_to not in attrs.
        let edge = &snap.edges[0];
        assert_eq!(edge.source, "files/a");
        assert_eq!(edge.target, "symbols/s");
        assert!(!edge.attrs.contains_key("_from"));
    }

    #[test]
    fn build_snapshot_drops_dangling_edges() {
        let node_docs = BTreeMap::from([
            ("files".to_string(), docs(json!([{ "_id": "files/a", "_key": "a" }]))),
        ]);
        // Edge points to a symbol we never fetched -> must be dropped.
        let edge_docs = BTreeMap::from([
            ("defines".to_string(), docs(json!([
                { "_id": "defines/e", "_from": "files/a", "_to": "symbols/missing" }
            ]))),
        ]);
        let snap = build_snapshot(
            "db", "g",
            vec!["files".into()], vec!["defines".into()],
            node_docs, edge_docs, 1, 1, None,
        );
        assert_eq!(snap.edges.len(), 0, "dangling edge must be dropped");
        assert_eq!(snap.nodes[0].degree, 0);
    }

    #[test]
    fn reject_flaglike_blocks_dashes_and_empties() {
        assert!(reject_flaglike("node", "files/a").is_ok());
        assert!(reject_flaglike("node", "--database=NL").is_err());
        assert!(reject_flaglike("node", "-d").is_err());
        assert!(reject_flaglike("node", "").is_err());
    }

    #[tokio::test]
    async fn neighbors_rejects_flag_smuggling_before_spawning() {
        // bin points nowhere; if validation is skipped we'd get a spawn error,
        // if it runs we get the validation error — assert the latter.
        let b = Backend { bin: "/nonexistent/hades".into(), database: "db".into(), limit: None };
        let err = b.neighbors("g", "--database=NL", "any", 10).await.unwrap_err().to_string();
        assert!(err.contains("leading dash"), "expected validation error, got: {err}");

        let err = b.neighbors("g", "files/a", "sideways", 10).await.unwrap_err().to_string();
        assert!(err.contains("invalid direction"), "got: {err}");
    }

    #[test]
    fn parse_database_list_reads_names() {
        let bytes = serde_json::to_vec(&json!({
            "data": { "count": 2, "databases": ["bident_burn", "NestedLearning"] }
        })).unwrap();
        assert_eq!(parse_database_list(&bytes).unwrap(), ["bident_burn", "NestedLearning"]);
    }

    #[test]
    fn parse_chunk_texts_reads_ordered_results() {
        let bytes = serde_json::to_vec(&json!({
            "data": { "count": 2, "results": [
                { "chunk_index": 0, "text": "line a\n" },
                { "chunk_index": 1, "text": "line b\n" }
            ]}
        })).unwrap();
        let chunks = parse_chunk_texts(&bytes).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks.iter().map(|c| c.text.as_str()).collect::<String>(), "line a\nline b\n");
    }

    #[test]
    fn parse_neighbors_extracts_delta_and_derives_collections() {
        let bytes = serde_json::to_vec(&json!({
            "data": { "results": [
                {
                    "edge": { "_id": "imports/e1", "_from": "files/a", "_to": "symbols/s", "resolved": true },
                    "vertex": { "_id": "symbols/s", "_key": "s", "name": "Foo", "_rev": "r" }
                },
                { // duplicate vertex should be de-duped
                    "edge": { "_id": "imports/e2", "_from": "files/a", "_to": "symbols/s" },
                    "vertex": { "_id": "symbols/s", "_key": "s", "name": "Foo" }
                }
            ]}
        })).unwrap();
        let delta = parse_neighbors(&bytes).unwrap();
        assert_eq!(delta.nodes.len(), 1, "duplicate vertex de-duped");
        assert_eq!(delta.nodes[0].collection, "symbols");
        assert_eq!(delta.nodes[0].label, "Foo");
        assert!(!delta.nodes[0].attrs.contains_key("_rev"));
        assert_eq!(delta.edges.len(), 2);
        assert_eq!(delta.edges[0].collection, "imports");
        assert_eq!(delta.edges[0].source, "files/a");
        assert!(!delta.edges[0].attrs.contains_key("_from"));
    }

    #[test]
    fn build_snapshot_marks_partial_when_capped_below_total() {
        let node_docs = BTreeMap::from([
            ("files".to_string(), docs(json!([{ "_id": "files/a", "_key": "a" }]))),
        ]);
        // limit=1, but the collection truly holds 50 -> partial slice.
        let snap = build_snapshot(
            "db", "g",
            vec!["files".into()], vec![],
            node_docs, BTreeMap::new(),
            50, 0, Some(1),
        );
        assert!(snap.meta.partial);
        assert_eq!(snap.meta.node_count, 1);
        assert_eq!(snap.meta.node_total, 50);
    }
}
