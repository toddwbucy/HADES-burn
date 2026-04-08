//! `hades db` subcommands — database operations and graph traversal.

use std::path::PathBuf;

use clap::Subcommand;

#[derive(Debug, Subcommand)]
pub enum DbCmd {
    /// Semantic search across the knowledge base.
    Query {
        /// Search text (optional — interactive mode if omitted).
        search_text: Option<String>,

        /// Maximum results to return.
        #[arg(short = 'n', long, default_value_t = 10)]
        limit: u32,

        /// Collection profile to search.
        #[arg(short = 'c', long)]
        collection: Option<String>,

        /// Enable hybrid search (vector + keyword).
        #[arg(short = 'H', long)]
        hybrid: bool,

        /// Enable re-ranking of results.
        #[arg(short = 'R', long)]
        rerank: bool,

        /// Enable structural graph ranking.
        #[arg(short = 'S', long)]
        structural: bool,

        /// Output format (json, jsonl, table).
        #[arg(short = 'f', long, default_value = "json")]
        format: String,

        /// Verbose output.
        #[arg(short = 'V', long)]
        verbose: bool,
    },

    /// Execute a raw AQL query.
    Aql {
        /// AQL query string.
        aql: String,

        /// Bind variables as JSON object.
        #[arg(short = 'b', long)]
        bind: Option<String>,

        /// Output format (json, jsonl, table).
        #[arg(short = 'f', long, default_value = "json")]
        format: String,

        /// Maximum results.
        #[arg(short = 'n', long)]
        limit: Option<u32>,
    },

    /// List documents in a collection.
    List {
        /// Collection profile.
        #[arg(short = 'c', long)]
        collection: Option<String>,

        /// Maximum results.
        #[arg(short = 'n', long, default_value_t = 20)]
        limit: u32,

        /// Output format (json, jsonl, table).
        #[arg(short = 'f', long, default_value = "json")]
        format: String,

        /// Filter by paper ID.
        #[arg(short = 'p', long)]
        paper: Option<String>,
    },

    /// Show database statistics.
    Stats {
        /// Output format (json, jsonl, table).
        #[arg(short = 'f', long, default_value = "json")]
        format: String,
    },

    /// Show recently ingested papers.
    Recent {
        /// Maximum results.
        #[arg(short = 'n', long, default_value_t = 10)]
        limit: u32,

        /// Output format (json, jsonl, table).
        #[arg(short = 'f', long, default_value = "json")]
        format: String,
    },

    /// Check data integrity and health.
    Health {
        /// Verbose output.
        #[arg(short = 'V', long)]
        verbose: bool,
    },

    /// Check if a document exists.
    Check {
        /// Document ID to check.
        document_id: String,
    },

    /// Remove a document and its related data.
    Purge {
        /// Document ID to purge.
        document_id: String,

        /// Skip confirmation prompt.
        #[arg(short = 'y', long)]
        force: bool,
    },

    /// Create a new collection.
    Create {
        /// Collection name.
        name: String,

        /// Collection type (document, edge).
        #[arg(short = 't', long, default_value = "document")]
        r#type: String,
    },

    /// Delete a document from a collection.
    Delete {
        /// Collection name.
        collection: String,

        /// Document key.
        key: String,

        /// Skip confirmation prompt.
        #[arg(short = 'y', long)]
        force: bool,
    },

    /// List all collections in the database.
    Collections {
        /// Output format (json, jsonl, table).
        #[arg(short = 'f', long, default_value = "json")]
        format: String,
    },

    /// List all databases.
    Databases {
        /// Output format (json, jsonl, table).
        #[arg(short = 'f', long, default_value = "json")]
        format: String,
    },

    /// Create a new database.
    CreateDatabase {
        /// Database name.
        name: String,
    },

    /// Count documents in a collection.
    Count {
        /// Collection name.
        collection: String,
    },

    /// Get a single document by key.
    Get {
        /// Collection name.
        collection: String,

        /// Document key.
        key: String,

        /// Output format (json, jsonl, table).
        #[arg(short = 'f', long, default_value = "json")]
        format: String,
    },

    /// Insert documents into a collection.
    Insert {
        /// Collection name.
        collection: String,

        /// JSON document(s) to insert (reads from stdin if omitted).
        #[arg(long)]
        data: Option<String>,

        /// Input file path.
        #[arg(short = 'i', long)]
        input: Option<PathBuf>,
    },

    /// Update a document in a collection.
    Update {
        /// Collection name.
        collection: String,

        /// Document key.
        key: String,

        /// JSON fields to update.
        #[arg(long)]
        data: Option<String>,
    },

    /// Export a collection to file.
    Export {
        /// Collection name.
        collection: String,

        /// Output file path.
        #[arg(short = 'o', long)]
        output: Option<PathBuf>,

        /// Output format (json, jsonl, table).
        #[arg(short = 'f', long, default_value = "jsonl")]
        format: String,

        /// Maximum documents to export.
        #[arg(short = 'n', long)]
        limit: Option<u32>,
    },

    /// Create a vector index on a collection.
    CreateIndex {
        /// Collection name.
        #[arg(short = 'c', long)]
        collection: Option<String>,

        /// Vector dimension.
        #[arg(long)]
        dimension: Option<u32>,

        /// Distance metric (cosine, euclidean, dotproduct).
        #[arg(long)]
        metric: Option<String>,
    },

    /// Show vector index status.
    IndexStatus {
        /// Collection name.
        #[arg(short = 'c', long)]
        collection: Option<String>,

        /// Output format (json, jsonl, table).
        #[arg(short = 'f', long, default_value = "json")]
        format: String,
    },

    /// Backfill text fields for embedding.
    BackfillText {
        /// Collection name.
        #[arg(short = 'c', long)]
        collection: Option<String>,

        /// Preview mode — don't actually modify documents.
        #[arg(long)]
        dry_run: bool,

        /// Batch size.
        #[arg(long, default_value_t = 100)]
        batch_size: u32,
    },

    /// Graph operations.
    #[command(subcommand)]
    Graph(DbGraphCmd),

    /// Schema management — define and inspect database ontologies.
    #[command(subcommand)]
    Schema(DbSchemaCmd),
}

// ── db graph ───────────────────────────────────────────────────────

#[derive(Debug, Subcommand)]
pub enum DbGraphCmd {
    /// Create a named graph.
    Create {
        /// Graph name.
        name: String,

        /// Edge definitions as JSON.
        #[arg(long)]
        edge_definitions: Option<String>,
    },

    /// List all named graphs.
    List {
        /// Output format (json, jsonl, table).
        #[arg(short = 'f', long, default_value = "json")]
        format: String,
    },

    /// Drop a named graph.
    Drop {
        /// Graph name.
        name: String,

        /// Also drop associated collections.
        #[arg(long)]
        drop_collections: bool,

        /// Skip confirmation.
        #[arg(short = 'y', long)]
        force: bool,
    },

    /// Traverse the graph from a starting vertex.
    Traverse {
        /// Starting vertex ID.
        start: String,

        /// Traversal direction (outbound, inbound, any).
        #[arg(short = 'd', long, default_value = "outbound")]
        direction: String,

        /// Minimum depth.
        #[arg(long, default_value_t = 1)]
        min_depth: u32,

        /// Maximum depth.
        #[arg(long, default_value_t = 3)]
        max_depth: u32,

        /// Graph name.
        #[arg(short = 'g', long)]
        graph: Option<String>,

        /// Output format (json, jsonl, table).
        #[arg(short = 'f', long, default_value = "json")]
        format: String,
    },

    /// Find the shortest path between two vertices.
    ShortestPath {
        /// Source vertex ID.
        source: String,

        /// Target vertex ID.
        target: String,

        /// Graph name.
        #[arg(short = 'g', long)]
        graph: Option<String>,

        /// Output format (json, jsonl, table).
        #[arg(short = 'f', long, default_value = "json")]
        format: String,
    },

    /// Find neighbors of a vertex.
    Neighbors {
        /// Vertex ID.
        vertex: String,

        /// Traversal direction (outbound, inbound, any).
        #[arg(short = 'd', long, default_value = "any")]
        direction: String,

        /// Maximum results.
        #[arg(short = 'n', long, default_value_t = 20)]
        limit: u32,

        /// Graph name.
        #[arg(short = 'g', long)]
        graph: Option<String>,

        /// Output format (json, jsonl, table).
        #[arg(short = 'f', long, default_value = "json")]
        format: String,
    },

    /// Materialize edges from implicit cross-reference fields.
    Materialize {
        /// Filter to a single edge definition name.
        #[arg(short = 'e', long)]
        edge: Option<String>,

        /// Preview mode — count edges without inserting.
        #[arg(long)]
        dry_run: bool,

        /// Also create named graphs via the Gharial API.
        #[arg(short = 'r', long)]
        register: bool,
    },
}

// ── db schema ─────────────────────────────────────────────────────

#[derive(Debug, Subcommand)]
pub enum DbSchemaCmd {
    /// Initialize the hades_schema collection with a seed ontology.
    Init {
        /// Seed name: "nl" (Nested Learning) or "empty" (blank for new domains).
        #[arg(short = 's', long)]
        seed: String,
    },

    /// List all edge definitions and named graphs in the schema.
    List {},

    /// Show a single edge definition or named graph by name.
    Show {
        /// Name of the edge definition or named graph.
        name: String,
    },

    /// Show schema version, checksum, and metadata.
    Version {},
}
