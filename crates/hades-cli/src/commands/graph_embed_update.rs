//! Native Rust implementation of the `hades graph-embed update` command.
//!
//! Incrementally re-embeds the knowledge graph without retraining. This is the
//! inductive serving path: it works for any trained checkpoint, but pays off
//! with an inductive `hetero_sage` model, which embeds nodes added since
//! training by forward pass alone. The checkpoint records its architecture, so
//! LoadCheckpoint rebuilds the matching model (RGCN or GraphSAGE).
//!
//! 1. Load graph from ArangoDB
//! 2. Serialize graph structure to safetensors (no edge splits)
//! 3. Load the previously trained checkpoint on the GPU service
//! 4. Load the current graph onto the GPU
//! 5. Full forward pass, or a bounded-neighbourhood forward for new nodes
//! 6. Export the full or compact embedding matrix back to ArangoDB

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde_json::json;
use tracing::{info, warn};

use hades_core::config::HadesConfig;
use hades_core::db::ArangoPool;
use hades_core::db::query::{self, ExecutionTarget};
use hades_core::graph::{
    ExportConfig, IDMap, decode_f32_embeddings, export_embeddings, export_embeddings_subset,
};
use hades_core::training::{TrainingClient, TrainingClientConfig};

use super::output::{self, OutputFormat};

/// Run the `graph-embed update` command.
pub async fn run(
    config: &HadesConfig,
    export_to: Option<&str>,
    checkpoint_dir: &str,
    new_nodes: bool,
) -> Result<()> {
    // ── Source database (read-only) ─────────────────────────────────
    let source_pool =
        ArangoPool::from_config(config).context("failed to connect to source ArangoDB")?;

    info!(db = %source_pool.database(), "connected to source database");

    // ── Preflight export target ─────────────────────────────────────
    let export_pool = if let Some(target_db) = export_to {
        let mut export_config = config.clone();
        export_config.database.name = Some(target_db.to_string());
        ArangoPool::from_config(&export_config)
            .context("failed to connect to export target database")?
    } else {
        source_pool.clone()
    };

    info!(db = %export_pool.database(), "export target validated");

    // ── Load graph from ArangoDB ────────────────────────────────────
    info!("loading runtime schema");
    let schema = hades_core::graph::RuntimeSchema::load(&source_pool)
        .await
        .context("failed to load runtime schema from hades_schema")?;
    info!(
        from_db = schema.from_database,
        num_relations = schema.meta.num_relations,
        feature_dim = schema.meta.feature_dim,
        "schema loaded"
    );

    info!("loading graph from ArangoDB");
    let (graph, id_map) = hades_core::graph::load(&source_pool, &schema).await?;

    info!(
        num_nodes = graph.num_nodes,
        num_edges = graph.num_edges,
        "graph loaded"
    );

    // ── Select destination nodes for an incremental update ──────────
    let selected_indices = if new_nodes {
        if schema.meta.resolved_model_type() != "hetero_sage" {
            anyhow::bail!(
                "--new-nodes requires schema model_type 'hetero_sage'; database schema selects '{}'",
                schema.meta.resolved_model_type()
            );
        }
        let indices = find_nodes_missing_structural_embeddings(&export_pool, &id_map)
            .await
            .context("failed to select nodes missing structural embeddings")?;
        info!(
            selected = indices.len(),
            "selected new nodes for inductive update"
        );
        Some(indices)
    } else {
        None
    };

    let checkpoint_path = PathBuf::from(checkpoint_dir).join("best.pt");
    if selected_indices.as_ref().is_some_and(Vec::is_empty) {
        let result_data = json!({
            "status": "success",
            "mode": "new_nodes",
            "graph": {
                "num_nodes": graph.num_nodes,
                "num_edges": graph.num_edges,
            },
            "model": {
                "checkpoint": checkpoint_path.to_string_lossy(),
                "architecture": schema.meta.resolved_model_type(),
            },
            "embeddings": { "num_nodes": 0 },
            "export": {
                "count": 0,
                "target_db": export_pool.database(),
            },
            "message": "all graph nodes already have structural embeddings",
        });
        output::print_output("graph-embed.update", result_data, &OutputFormat::Json);
        return Ok(());
    }

    // ── Validate checkpoint and connect to training service ─────────
    if !checkpoint_path.exists() {
        anyhow::bail!(
            "no trained model found at {}. Run `graph-embed train` first.",
            checkpoint_path.display()
        );
    }
    let training_client = TrainingClient::connect(TrainingClientConfig::default())
        .await
        .context("failed to connect to HADES training service")?;

    // ── Serialize graph for inference ───────────────────────────────
    let safetensors_dir = PathBuf::from(checkpoint_dir);
    std::fs::create_dir_all(&safetensors_dir).context("failed to create checkpoint directory")?;
    let safetensors_path = safetensors_dir.join("graph_inference.safetensors");

    let graph_ref = graph.clone();
    let path = safetensors_path.clone();
    tokio::task::spawn_blocking(move || {
        hades_prefetch::serialize_graph_for_inference_to_file(&path, &graph_ref)
    })
    .await
    .map_err(|e| anyhow::anyhow!("serialization task panicked: {e}"))?
    .context("failed to serialize graph")?;

    // ── Load checkpoint + graph onto GPU ────────────────────────────
    let init = training_client
        .load_checkpoint(&checkpoint_path, Some(&config.gpu.device))
        .await
        .context("failed to load model checkpoint")?;

    info!(
        num_parameters = init.num_parameters,
        device = %init.device,
        architecture = %init.architecture,
        "model checkpoint loaded"
    );
    if new_nodes && init.architecture != "hetero_sage" {
        anyhow::bail!(
            "--new-nodes requires a hetero_sage checkpoint; loaded checkpoint architecture is '{}'",
            init.architecture
        );
    }

    training_client
        .load_graph(&safetensors_path)
        .await
        .context("failed to load graph onto GPU")?;

    // ── Generate embeddings (single forward pass) ───────────────────
    let embeddings_path = safetensors_dir.join("embeddings.bin");
    let emb_result = if let Some(indices) = &selected_indices {
        training_client
            .get_embeddings_subset(Some(&embeddings_path), indices)
            .await
            .context("failed to generate new-node embeddings")?
    } else {
        training_client
            .get_embeddings(Some(&embeddings_path))
            .await
            .context("failed to generate embeddings")?
    };

    let expected_embeddings = selected_indices.as_ref().map_or(id_map.len(), Vec::len);
    if emb_result.num_nodes as usize != expected_embeddings {
        anyhow::bail!(
            "training service returned {} embedding rows; expected {}",
            emb_result.num_nodes,
            expected_embeddings
        );
    }

    let emb_bytes = tokio::fs::read(&embeddings_path)
        .await
        .context("failed to read embeddings file")?;
    let embeddings =
        decode_f32_embeddings(&emb_bytes).context("failed to decode embedding bytes")?;

    info!(
        num_nodes = emb_result.num_nodes,
        embed_dim = emb_result.embed_dim,
        "embeddings generated"
    );

    // ── Export embeddings to ArangoDB ────────────────────────────────
    info!(db = %export_pool.database(), "exporting embeddings");

    let export_result = if let Some(indices) = &selected_indices {
        export_embeddings_subset(
            &export_pool,
            &id_map,
            indices,
            &embeddings,
            emb_result.embed_dim as usize,
            &ExportConfig::default(),
        )
        .await
    } else {
        export_embeddings(
            &export_pool,
            &id_map,
            &embeddings,
            emb_result.embed_dim as usize,
            &ExportConfig::default(),
        )
        .await
    }
    .context("embedding export failed")?;

    let export_count = export_result.total_exported;
    let expected_total = expected_embeddings;
    if export_count < expected_total {
        warn!(
            total_exported = export_count,
            expected_total,
            skipped = expected_total - export_count,
            "partial export — some documents were not updated"
        );
    } else {
        info!(total = export_count, "embeddings exported");
    }

    // ── JSON output to stdout ───────────────────────────────────────
    let result_data = json!({
        "status": "success",
        "mode": if new_nodes { "new_nodes" } else { "all_nodes" },
        "graph": {
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
        },
        "model": {
            "checkpoint": checkpoint_path.to_string_lossy(),
            "num_parameters": init.num_parameters,
            "device": init.device,
            "architecture": init.architecture,
        },
        "embeddings": {
            "num_nodes": emb_result.num_nodes,
            "embed_dim": emb_result.embed_dim,
        },
        "export": {
            "count": export_count,
            "target_db": export_pool.database(),
        },
    });

    output::print_output("graph-embed.update", result_data, &OutputFormat::Json);
    Ok(())
}

const MISSING_EMBEDDING_BATCH_SIZE: usize = 5_000;

/// Return global graph indices whose destination documents exist but do not
/// yet carry a structural embedding. Results are sorted so request and compact
/// output row order are deterministic.
async fn find_nodes_missing_structural_embeddings(
    pool: &ArangoPool,
    id_map: &IDMap,
) -> Result<Vec<u32>> {
    let mut missing = Vec::new();
    for (collection, nodes) in id_map.nodes_by_collection() {
        for batch in nodes.chunks(MISSING_EMBEDDING_BATCH_SIZE) {
            let keys: Vec<&str> = batch
                .iter()
                .filter_map(|(arango_id, _)| arango_id.split_once('/').map(|(_, key)| key))
                .collect();
            let bind_vars = json!({ "@collection": collection, "keys": keys });
            let result = query::query(
                pool,
                "FOR key IN @keys \
                 LET d = DOCUMENT(@@collection, key) \
                 FILTER d != null AND d.structural_embedding == null \
                 RETURN d._id",
                Some(&bind_vars),
                None,
                false,
                ExecutionTarget::Reader,
            )
            .await
            .with_context(|| format!("failed to inspect collection {collection}"))?;

            for value in result.results {
                let arango_id = value.as_str().with_context(|| {
                    format!("collection {collection} returned a non-string document ID")
                })?;
                let index = id_map.get_index(arango_id).with_context(|| {
                    format!("destination returned unknown graph node {arango_id}")
                })?;
                missing.push(u32::try_from(index).context("graph node index exceeds u32::MAX")?);
            }
        }
    }
    missing.sort_unstable();
    missing.dedup();
    Ok(missing)
}
