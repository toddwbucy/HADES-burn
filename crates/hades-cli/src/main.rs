//! HADES-Burn CLI — Rust rewrite of the HADES knowledge graph system.
//!
//! During the strangler-fig migration, every command dispatches to the
//! Python `hades` CLI via subprocess.  As Rust modules mature, commands
//! switch to native implementations one at a time.

mod commands;
mod dispatch;

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process;

use clap::{Parser, Subcommand};
use hades_core::config;

use commands::{
    arxiv::ArxivCmd, codebase::CodebaseCmd, db::DbCmd,
    embed::{EmbedCmd, EmbedGpuCmd, EmbedServiceCmd},
    graph_embed::GraphEmbedCmd, smell::SmellCmd, task::TaskCmd,
};

/// HADES-Burn — AI model interface for semantic search over academic papers,
/// backed by ArangoDB with vector similarity and graph traversal.
#[derive(Parser)]
#[command(name = "hades-burn", version, about)]
struct Cli {
    /// Target ArangoDB database name (overrides config/env).
    #[arg(long = "database", alias = "db", global = true)]
    database: Option<String>,

    /// GPU device index for embedding commands.
    #[arg(short = 'g', long = "gpu", global = true)]
    gpu: Option<u32>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// System status for workspace discovery.
    Status {
        /// Output format (text, json).
        #[arg(short = 'f', long, default_value = "text")]
        format: String,

        /// Verbose output.
        #[arg(short = 'V', long)]
        verbose: bool,
    },

    /// Metadata-first context orientation for a database.
    Orient {
        /// Collection to orient on.
        #[arg(short = 'c', long)]
        collection: Option<String>,

        /// Output format (text, json).
        #[arg(short = 'f', long, default_value = "text")]
        format: String,
    },

    /// Extract text from documents (PDF, LaTeX, etc).
    Extract {
        /// File path to extract from.
        file: PathBuf,

        /// Output format (text, json).
        #[arg(short = 'f', long, default_value = "text")]
        format: String,

        /// Output file path.
        #[arg(short = 'o', long)]
        output: Option<PathBuf>,
    },

    /// Ingest documents into the knowledge base.
    Ingest {
        /// Input arXiv IDs or file paths (can mix both).
        inputs: Vec<String>,

        /// Custom document ID (single input only).
        #[arg(long)]
        id: Option<String>,

        /// Run in batch mode.
        #[arg(short = 'b', long)]
        batch: bool,

        /// Resume a previously interrupted batch.
        #[arg(short = 'r', long, conflicts_with = "reset")]
        resume: bool,

        /// Custom metadata as JSON.
        #[arg(short = 'm', long)]
        metadata: Option<String>,

        /// Embedding task type (e.g. "code" for Jina Code LoRA).
        #[arg(short = 't', long)]
        task: Option<String>,

        /// Compliance claims (CS-NN format).
        #[arg(long)]
        claims: Vec<String>,

        /// Collection profile (arxiv, sync, default).
        #[arg(short = 'c', long)]
        collection: Option<String>,

        /// Force re-processing of existing documents.
        #[arg(short = 'f', long)]
        force: bool,

        /// Reset batch state (clear previous checkpoint).
        #[arg(long, conflicts_with = "resume")]
        reset: bool,

        /// Maximum concurrent items (overrides config, must be >= 1).
        #[arg(long)]
        concurrency: Option<NonZeroUsize>,
    },

    /// Create a compliance edge linking a document to a smell node.
    Link {
        /// Source document ID.
        source_id: String,

        /// Compliance claims (CS-NN format).
        #[arg(long)]
        claims: Vec<String>,

        /// Skip confirmation.
        #[arg(short = 'y', long)]
        force: bool,
    },

    /// arXiv paper sync and status.
    #[command(subcommand)]
    Arxiv(ArxivCmd),

    /// Database operations — query, CRUD, indexes, and graph traversal.
    #[command(subcommand)]
    Db(DbCmd),

    /// Embedding generation and service management.
    #[command(subcommand)]
    Embed(EmbedCmd),

    /// Code ingestion and graph operations.
    #[command(subcommand)]
    Codebase(CodebaseCmd),

    /// Persephone task management — create, track, review, and hand off tasks.
    #[command(subcommand)]
    Task(TaskCmd),

    /// Code smell and compliance checking.
    #[command(subcommand)]
    Smell(SmellCmd),

    /// Graph embedding operations — train and query structural embeddings.
    #[command(subcommand)]
    GraphEmbed(GraphEmbedCmd),

    /// Start the HADES daemon (Unix socket query server).
    Daemon {
        /// Socket path (default: /run/hades/hades.sock).
        #[arg(long)]
        socket: Option<String>,
    },
}

/// Initialize tracing (structured logging to stderr).
fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .init();
}

/// Run `codebase ingest` (or update) with shared runtime + error handling.
fn run_codebase_ingest(
    config: &hades_core::config::HadesConfig,
    path: PathBuf,
    language: Option<&str>,
    batch: bool,
) -> anyhow::Result<()> {
    init_tracing();
    let rt = tokio::runtime::Runtime::new()?;
    let result = rt.block_on(commands::codebase_ingest::run(config, path, language, batch));
    match result {
        Ok(()) => Ok(()),
        Err(e) => {
            if e.downcast_ref::<commands::codebase_ingest::CodebaseIngestFailure>()
                .is_some()
            {
                tracing::error!(error = %e, "codebase ingest failed");
                process::exit(1);
            }
            Err(e)
        }
    }
}

fn main() -> anyhow::Result<()> {
    // Capture raw args before clap consumes them — used for pass-through dispatch.
    let raw_args: Vec<String> = std::env::args().collect();

    let cli = Cli::parse();

    let mut config = config::load_config()?;
    config.apply_cli_overrides(cli.database.as_deref(), cli.gpu);

    // Save global opts before cli.command is consumed by the match.
    let database = cli.database;
    let gpu = cli.gpu;

    // ── Native command dispatch ──────────────────────────────────────────
    // Commands with native Rust implementations are handled here.
    // All other commands fall through to the Python CLI dispatch below.
    match cli.command {
        Commands::Ingest {
            inputs, id, batch, resume, metadata, task, claims, collection, force,
            reset, concurrency,
        } => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            let result = rt.block_on(commands::ingest::run(
                &config,
                inputs.into_iter().map(PathBuf::from).collect(),
                batch,
                metadata.as_deref(),
                &claims,
                collection.as_deref(),
                force,
                task.as_deref(),
                id.as_deref(),
                resume,
                reset,
                concurrency.map(NonZeroUsize::get),
            ));
            return match result {
                Ok(()) => Ok(()),
                Err(e) => {
                    if e.downcast_ref::<commands::ingest::IngestFailure>().is_some() {
                        process::exit(1);
                    }
                    Err(e)
                }
            };
        }
        Commands::Arxiv(ArxivCmd::Sync {
            from_date, categories, max_results, batch_size, incremental,
        }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::arxiv_sync::run(
                &config,
                from_date.as_deref(),
                categories.as_deref(),
                max_results,
                batch_size,
                incremental,
            ));
        }
        Commands::Arxiv(ArxivCmd::SyncStatus { limit }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::arxiv_sync::status(&config, limit));
        }
        Commands::Codebase(CodebaseCmd::Ingest { path, language, batch }) => {
            return run_codebase_ingest(&config, path, language.as_deref(), batch);
        }
        Commands::Codebase(CodebaseCmd::Update { path }) => {
            return run_codebase_ingest(&config, path, None, false);
        }
        Commands::Codebase(CodebaseCmd::Stats) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::codebase_mgmt::run_stats(&config));
        }
        Commands::GraphEmbed(GraphEmbedCmd::Embed { node_id }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::graph_embed_query::run_embed(
                &config, &node_id,
            ));
        }
        Commands::GraphEmbed(GraphEmbedCmd::Neighbors { node_id, limit }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::graph_embed_query::run_neighbors(
                &config, &node_id, limit,
            ));
        }
        Commands::GraphEmbed(GraphEmbedCmd::Train {
            epochs, dimension, hidden_dim, num_bases, dropout, lr, weight_decay,
            patience, val_ratio, test_ratio, neg_ratio, export_to, checkpoint_dir,
            val_every, prefetch_depth, no_export,
        }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::graph_embed_train::run(
                &config,
                epochs,
                dimension,
                hidden_dim,
                num_bases,
                dropout,
                lr,
                weight_decay,
                patience,
                val_ratio,
                test_ratio,
                neg_ratio,
                export_to.as_deref(),
                &checkpoint_dir,
                val_every,
                prefetch_depth,
                no_export,
            ));
        }
        Commands::GraphEmbed(GraphEmbedCmd::Update { export_to, checkpoint_dir }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::graph_embed_update::run(
                &config,
                export_to.as_deref(),
                &checkpoint_dir,
            ));
        }
        Commands::Daemon { socket } => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::daemon::run(
                &config,
                socket.as_deref(),
            ));
        }
        // ── Native DB read commands ─────────────────────────────────────
        Commands::Db(commands::db::DbCmd::Get { collection, key, format }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_read::run_get(&config, &collection, &key, &format));
        }
        Commands::Db(commands::db::DbCmd::Count { collection }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_read::run_count(&config, &collection));
        }
        Commands::Db(commands::db::DbCmd::Collections { format }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_read::run_collections(&config, &format));
        }
        Commands::Db(commands::db::DbCmd::Check { document_id }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_read::run_check(&config, &document_id));
        }
        Commands::Db(commands::db::DbCmd::Recent { limit, format }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_read::run_recent(&config, limit, &format));
        }
        Commands::Db(commands::db::DbCmd::List { collection, limit, paper, format }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_read::run_list(
                &config,
                collection.as_deref(),
                limit,
                paper.as_deref(),
                &format,
            ));
        }
        Commands::Db(commands::db::DbCmd::Aql { aql, bind, limit, format }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_read::run_aql(
                &config,
                &aql,
                bind.as_deref(),
                limit,
                &format,
            ));
        }
        Commands::Db(commands::db::DbCmd::Health { verbose }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_read::run_health(&config, verbose));
        }
        Commands::Db(commands::db::DbCmd::Stats { format }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_read::run_stats(&config, &format));
        }
        Commands::Db(commands::db::DbCmd::Export { collection, output, format, limit }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_read::run_export(
                &config,
                &collection,
                output.as_deref(),
                &format,
                limit,
            ));
        }
        Commands::Db(commands::db::DbCmd::IndexStatus { collection, format }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_read::run_index_status(
                &config,
                collection.as_deref(),
                &format,
            ));
        }
        Commands::Db(commands::db::DbCmd::Databases { format }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_read::run_databases(&config, &format));
        }
        // ── Native DB write commands ────────────────────────────────────
        Commands::Db(commands::db::DbCmd::Insert { collection, data, input }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_write::run_insert(
                &config, &collection, data.as_deref(), input.as_deref(),
            ));
        }
        Commands::Db(commands::db::DbCmd::Update { collection, key, data }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_write::run_update(
                &config, &collection, &key, data.as_deref(),
            ));
        }
        Commands::Db(commands::db::DbCmd::Delete { collection, key, force }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_write::run_delete(
                &config, &collection, &key, force,
            ));
        }
        Commands::Db(commands::db::DbCmd::Purge { document_id, force }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_write::run_purge(
                &config, &document_id, force,
            ));
        }
        Commands::Db(commands::db::DbCmd::Create { name, r#type }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_write::run_create_collection(
                &config, &name, &r#type,
            ));
        }
        Commands::Db(commands::db::DbCmd::CreateDatabase { name }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_write::run_create_database(&config, &name));
        }
        Commands::Db(commands::db::DbCmd::CreateIndex { collection, dimension, metric }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_write::run_create_index(
                &config, collection.as_deref(), dimension, metric.as_deref(),
            ));
        }
        Commands::Db(commands::db::DbCmd::BackfillText { collection, dry_run, batch_size }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_write::run_backfill_text(
                &config, collection.as_deref(), dry_run, batch_size,
            ));
        }
        // ── Native DB graph commands ──────────────────────────────────
        Commands::Db(commands::db::DbCmd::Graph(commands::db::DbGraphCmd::Traverse {
            start, direction, min_depth, max_depth, graph, format: _,
        })) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_graph::run_traverse(
                &config, &start, &direction, min_depth, max_depth, graph.as_deref(),
            ));
        }
        Commands::Db(commands::db::DbCmd::Graph(commands::db::DbGraphCmd::ShortestPath {
            source, target, graph, format: _,
        })) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_graph::run_shortest_path(
                &config, &source, &target, graph.as_deref(),
            ));
        }
        Commands::Db(commands::db::DbCmd::Graph(commands::db::DbGraphCmd::Neighbors {
            vertex, direction, limit, graph, format: _,
        })) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_graph::run_neighbors(
                &config, &vertex, &direction, limit, graph.as_deref(),
            ));
        }
        Commands::Db(commands::db::DbCmd::Graph(commands::db::DbGraphCmd::List { format: _ })) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_graph::run_list(&config));
        }
        Commands::Db(commands::db::DbCmd::Graph(commands::db::DbGraphCmd::Create {
            name, edge_definitions,
        })) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_graph::run_create(
                &config, &name, edge_definitions.as_deref(),
            ));
        }
        Commands::Db(commands::db::DbCmd::Graph(commands::db::DbGraphCmd::Drop {
            name, drop_collections, force,
        })) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::db_graph::run_drop(
                &config, &name, drop_collections, force,
            ));
        }
        // ── Native system commands ────────────────────────────────────
        Commands::Status { format, verbose } => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::system::run_status(&config, verbose, &format));
        }
        Commands::Orient { collection, format } => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::system::run_orient(
                &config, collection.as_deref(), &format,
            ));
        }
        // ── Native task commands ───────────────────────────────────────
        Commands::Task(commands::task::TaskCmd::List {
            status, r#type, parent, limit, format,
        }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::task_mgmt::run_list(
                &config, status.as_deref(), r#type.as_deref(), parent.as_deref(), limit, &format,
            ));
        }
        Commands::Task(commands::task::TaskCmd::Show { key, format }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::task_mgmt::run_show(&config, &key, &format));
        }
        Commands::Task(commands::task::TaskCmd::Create {
            title, description, r#type, parent, priority, tags,
        }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::task_mgmt::run_create(
                &config, &title, description.as_deref(), &r#type,
                parent.as_deref(), priority.as_deref(), &tags,
            ));
        }
        Commands::Task(commands::task::TaskCmd::Update {
            key, title, description, priority, add_tags, remove_tags,
        }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::task_mgmt::run_update(
                &config,
                hades_core::dispatch::TaskUpdateParams {
                    key,
                    title,
                    description,
                    priority,
                    status: None,
                    add_tags,
                    remove_tags,
                },
            ));
        }
        Commands::Task(commands::task::TaskCmd::Close { key, message }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::task_mgmt::run_close(
                &config, &key, message.as_deref(),
            ));
        }
        Commands::Task(commands::task::TaskCmd::Start { key }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::task_mgmt::run_start(&config, &key));
        }
        Commands::Task(commands::task::TaskCmd::Review { key, message }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::task_mgmt::run_review(
                &config, &key, message.as_deref(),
            ));
        }
        Commands::Task(commands::task::TaskCmd::Approve { key, human }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::task_mgmt::run_approve(&config, &key, human));
        }
        Commands::Task(commands::task::TaskCmd::Block { key, message, blocker }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::task_mgmt::run_block(
                &config, &key, message.as_deref(), blocker.as_deref(),
            ));
        }
        Commands::Task(commands::task::TaskCmd::Unblock { key }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::task_mgmt::run_unblock(&config, &key));
        }
        Commands::Task(commands::task::TaskCmd::Handoff { key, message }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::task_mgmt::run_handoff(
                &config, &key, message.as_deref(),
            ));
        }
        Commands::Task(commands::task::TaskCmd::HandoffShow { key, format }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::task_mgmt::run_handoff_show(
                &config, &key, &format,
            ));
        }
        Commands::Task(commands::task::TaskCmd::Context { key }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::task_mgmt::run_context(&config, &key));
        }
        Commands::Task(commands::task::TaskCmd::Log { key, limit }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::task_mgmt::run_log(&config, &key, limit));
        }
        Commands::Task(commands::task::TaskCmd::Sessions { key }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::task_mgmt::run_sessions(&config, &key));
        }
        Commands::Task(commands::task::TaskCmd::Dep { key, add, remove, graph }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::task_mgmt::run_dep(
                &config, &key, add.as_deref(), remove.as_deref(), graph,
            ));
        }
        Commands::Task(commands::task::TaskCmd::Usage) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::task_mgmt::run_usage(&config));
        }
        Commands::Task(commands::task::TaskCmd::GraphIntegration { format }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::task_mgmt::run_graph_integration(
                &config, &format,
            ));
        }
        // All TaskCmd variants are handled natively above.

        // ── Embed commands ────────────────────────────────────────────
        Commands::Embed(EmbedCmd::Text { text, format }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::embed_mgmt::run_embed_text(&config, &text, &format));
        }
        Commands::Embed(EmbedCmd::Service(EmbedServiceCmd::Status)) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::embed_mgmt::run_service_status(&config));
        }
        Commands::Embed(EmbedCmd::Service(EmbedServiceCmd::Start { foreground })) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::embed_mgmt::run_service_start(&config, foreground));
        }
        Commands::Embed(EmbedCmd::Service(EmbedServiceCmd::Stop)) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::embed_mgmt::run_service_stop());
        }
        Commands::Embed(EmbedCmd::Gpu(EmbedGpuCmd::Status)) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::embed_mgmt::run_gpu_status(&config));
        }
        Commands::Embed(EmbedCmd::Gpu(EmbedGpuCmd::List)) => {
            return commands::embed_mgmt::run_gpu_list();
        }

        // ── Extract command ───────────────────────────────────────────
        Commands::Extract { file, format, output } => {
            return commands::embed_mgmt::run_extract(
                &file,
                &format,
                output.as_deref(),
            );
        }

        // ── Smell & compliance commands ──────────────────────────────
        Commands::Smell(SmellCmd::Check { path, format, verbose }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::smell_mgmt::run_smell_check(
                &config, &path, &format, verbose,
            ));
        }
        Commands::Smell(SmellCmd::Verify { path, claims }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::smell_mgmt::run_smell_verify(
                &config, &path, &claims,
            ));
        }
        Commands::Smell(SmellCmd::Report { path, output, format }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::smell_mgmt::run_smell_report(
                &config, &path, output.as_deref(), &format,
            ));
        }
        Commands::Link { source_id, claims, force } => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::smell_mgmt::run_link(
                &config, &source_id, &claims, force,
            ));
        }

        _ => {} // Remaining commands fall through to Python passthrough.
    }

    // ── Python passthrough (strangler-fig) ───────────────────────────────
    // Commands not yet ported to Rust dispatch to the Python `hades` CLI.
    let passthrough = strip_global_opts(&raw_args[1..]);
    let status = dispatch::dispatch_with_globals(
        database.as_deref(),
        gpu,
        &passthrough.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
    )?;

    if !status.success() {
        process::exit(status.code().unwrap_or(1));
    }

    Ok(())
}

/// Remove global options (--database/--db, --gpu/-g) from the raw arg list,
/// since `dispatch_with_globals` re-inserts them in the format the Python CLI expects.
fn strip_global_opts(args: &[String]) -> Vec<String> {
    let mut result = Vec::new();
    let mut skip_next = false;

    for arg in args {
        if skip_next {
            skip_next = false;
            continue;
        }
        match arg.as_str() {
            "--database" | "--db" | "--gpu" => {
                // These flags take a value in the next position
                skip_next = true;
                continue;
            }
            "-g" => {
                skip_next = true;
                continue;
            }
            _ => {}
        }
        // Handle --database=value and --db=value forms
        if arg.starts_with("--database=") || arg.starts_with("--db=") || arg.starts_with("--gpu=")
        {
            continue;
        }
        // Handle -g<value> (no space) form
        if arg.starts_with("-g") && arg.len() > 2 {
            let rest = &arg[2..];
            if rest.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
        }
        result.push(arg.clone());
    }
    result
}

// NOTE: The build_dispatch_args() functions that previously translated
// parsed clap args back into strings have been removed.  The passthrough
// approach (strip_global_opts + raw args) is more reliable during the
// strangler-fig phase because it avoids flag-name mismatches between
// the Rust clap definitions and the Python Typer CLI.
