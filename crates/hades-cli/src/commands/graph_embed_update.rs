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

use std::collections::VecDeque;
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

    // Preserve the full-update fast-fail path: unlike `--new-nodes`, it has
    // no possible no-op result that would justify loading the graph first.
    let checkpoint_path = PathBuf::from(checkpoint_dir).join("best.pt");
    let preflight_training_client = if new_nodes {
        None
    } else {
        validate_checkpoint(&checkpoint_path)?;
        Some(
            TrainingClient::connect(TrainingClientConfig::default())
                .await
                .context("failed to connect to HADES training service")?,
        )
    };

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
    let selection = if new_nodes {
        if schema.meta.resolved_model_type() != "hetero_sage" {
            anyhow::bail!(
                "--new-nodes requires schema model_type 'hetero_sage'; database schema selects '{}'",
                schema.meta.resolved_model_type()
            );
        }
        let selection = find_nodes_missing_structural_embeddings(&export_pool, &id_map)
            .await
            .context("failed to select nodes missing structural embeddings")?;
        info!(
            selected = selection.indices.len(),
            absent_from_target = selection.absent_from_target,
            "selected new nodes for inductive update"
        );
        if selection.absent_from_target > 0 {
            warn!(
                absent_from_target = selection.absent_from_target,
                target_db = %export_pool.database(),
                "source graph nodes are absent from the export target"
            );
        }
        Some(selection)
    } else {
        None
    };

    if selection
        .as_ref()
        .is_some_and(|selection| selection.indices.is_empty())
    {
        let absent_from_target = selection
            .as_ref()
            .map_or(0, |selection| selection.absent_from_target);
        let message = if absent_from_target == 0 {
            "all graph nodes already have structural embeddings"
        } else {
            "no destination documents need embeddings; some source graph nodes are absent from the target"
        };
        let result_data = json!({
            "status": "success",
            "mode": "new_nodes",
            "graph": {
                "num_nodes": graph.num_nodes,
                "num_edges": graph.num_edges,
            },
            "model": {
                "schema_architecture": schema.meta.resolved_model_type(),
                "checkpoint_validated": false,
                "service_contacted": false,
            },
            "embeddings": { "num_nodes": 0 },
            "export": {
                "count": 0,
                "target_db": export_pool.database(),
                "absent_from_target": absent_from_target,
            },
            "message": message,
        });
        output::print_output("graph-embed.update", result_data, &OutputFormat::Json);
        return Ok(());
    }

    // ── Validate checkpoint and connect to training service ─────────
    let training_client = if let Some(client) = preflight_training_client {
        client
    } else {
        validate_checkpoint(&checkpoint_path)?;
        TrainingClient::connect(TrainingClientConfig::default())
            .await
            .context("failed to connect to HADES training service")?
    };

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
    let emb_result = if let Some(selection) = &selection {
        training_client
            .get_embeddings_subset(Some(&embeddings_path), &selection.indices)
            .await
            .context("failed to generate new-node embeddings")?
    } else {
        training_client
            .get_embeddings(Some(&embeddings_path))
            .await
            .context("failed to generate embeddings")?
    };

    let expected_embeddings = selection
        .as_ref()
        .map_or(id_map.len(), |selection| selection.indices.len());
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

    let export_result = if let Some(selection) = &selection {
        export_embeddings_subset(
            &export_pool,
            &id_map,
            &selection.indices,
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
            "absent_from_target": selection
                .as_ref()
                .map_or(0, |selection| selection.absent_from_target),
        },
    });

    output::print_output("graph-embed.update", result_data, &OutputFormat::Json);
    Ok(())
}

const MISSING_EMBEDDING_BATCH_SIZE: usize = 5_000;
const MISSING_EMBEDDING_QUERY_CONCURRENCY: usize = 8;

#[derive(Debug)]
struct MissingEmbeddingSelection {
    indices: Vec<u32>,
    absent_from_target: usize,
}

#[derive(Debug)]
struct MissingEmbeddingBatch {
    collection: String,
    keys: Vec<String>,
}

#[derive(Debug)]
struct MissingEmbeddingBatchResult {
    missing_ids: Vec<String>,
    absent_from_target: usize,
}

fn validate_checkpoint(checkpoint_path: &std::path::Path) -> Result<()> {
    if !checkpoint_path.exists() {
        anyhow::bail!(
            "no trained model found at {}. Run `graph-embed train` first.",
            checkpoint_path.display()
        );
    }
    Ok(())
}

/// Return global graph indices whose destination documents exist but do not
/// yet carry a structural embedding. Results are sorted so request and compact
/// output row order are deterministic.
async fn find_nodes_missing_structural_embeddings(
    pool: &ArangoPool,
    id_map: &IDMap,
) -> Result<MissingEmbeddingSelection> {
    let mut batches = VecDeque::new();
    for (collection, nodes) in id_map.nodes_by_collection() {
        for batch in nodes.chunks(MISSING_EMBEDDING_BATCH_SIZE) {
            let keys: Vec<String> = batch
                .iter()
                .filter_map(|(arango_id, _)| {
                    arango_id.split_once('/').map(|(_, key)| key.to_string())
                })
                .collect();
            batches.push_back(MissingEmbeddingBatch {
                collection: collection.to_string(),
                keys,
            });
        }
    }

    let mut tasks = tokio::task::JoinSet::new();
    for _ in 0..MISSING_EMBEDDING_QUERY_CONCURRENCY {
        if let Some(batch) = batches.pop_front() {
            spawn_missing_embedding_query(&mut tasks, pool.clone(), batch);
        }
    }

    let mut missing_ids = Vec::new();
    let mut absent_from_target = 0usize;
    while let Some(result) = tasks.join_next().await {
        let batch_result = result
            .context("missing-embedding query task panicked")?
            .context("missing-embedding query failed")?;
        missing_ids.extend(batch_result.missing_ids);
        absent_from_target += batch_result.absent_from_target;
        if let Some(batch) = batches.pop_front() {
            spawn_missing_embedding_query(&mut tasks, pool.clone(), batch);
        }
    }

    let mut missing = Vec::with_capacity(missing_ids.len());
    for arango_id in missing_ids {
        let index = id_map
            .get_index(&arango_id)
            .with_context(|| format!("destination returned unknown graph node {arango_id}"))?;
        missing.push(u32::try_from(index).context("graph node index exceeds u32::MAX")?);
    }
    missing.sort_unstable();
    missing.dedup();
    Ok(MissingEmbeddingSelection {
        indices: missing,
        absent_from_target,
    })
}

fn spawn_missing_embedding_query(
    tasks: &mut tokio::task::JoinSet<Result<MissingEmbeddingBatchResult>>,
    pool: ArangoPool,
    batch: MissingEmbeddingBatch,
) {
    tasks.spawn(async move { query_missing_embedding_batch(&pool, batch).await });
}

async fn query_missing_embedding_batch(
    pool: &ArangoPool,
    batch: MissingEmbeddingBatch,
) -> Result<MissingEmbeddingBatchResult> {
    let bind_vars = json!({
        "@collection": batch.collection,
        "keys": batch.keys,
    });
    let result = query::query(
        pool,
        "FOR key IN @keys \
         LET d = DOCUMENT(@@collection, key) \
         RETURN { \
           id: d == null ? null : d._id, \
           missing: d != null AND d.structural_embedding == null, \
           absent: d == null \
         }",
        Some(&bind_vars),
        None,
        false,
        ExecutionTarget::Reader,
    )
    .await
    .with_context(|| format!("failed to inspect collection {}", batch.collection))?;

    parse_missing_embedding_results(&batch.collection, result.results)
}

fn parse_missing_embedding_results(
    collection: &str,
    results: Vec<serde_json::Value>,
) -> Result<MissingEmbeddingBatchResult> {
    let mut missing_ids = Vec::new();
    let mut absent_from_target = 0usize;
    for value in results {
        if value.get("absent").and_then(serde_json::Value::as_bool) == Some(true) {
            absent_from_target += 1;
            continue;
        }
        if value.get("missing").and_then(serde_json::Value::as_bool) != Some(true) {
            continue;
        }
        let arango_id = value
            .get("id")
            .and_then(serde_json::Value::as_str)
            .with_context(|| format!("collection {collection} returned an invalid document ID"))?;
        missing_ids.push(arango_id.to_string());
    }
    Ok(MissingEmbeddingBatchResult {
        missing_ids,
        absent_from_target,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_missing_and_absent_destination_documents() {
        let result = parse_missing_embedding_results(
            "codebase_files",
            vec![
                json!({ "id": "codebase_files/a", "missing": true, "absent": false }),
                json!({ "id": "codebase_files/b", "missing": false, "absent": false }),
                json!({ "id": null, "missing": false, "absent": true }),
            ],
        )
        .unwrap();

        assert_eq!(result.missing_ids, vec!["codebase_files/a"]);
        assert_eq!(result.absent_from_target, 1);
    }

    #[test]
    fn missing_destination_document_requires_an_id() {
        let err = parse_missing_embedding_results(
            "codebase_files",
            vec![json!({ "id": null, "missing": true, "absent": false })],
        )
        .unwrap_err();
        assert!(err.to_string().contains("invalid document ID"));
    }

    #[test]
    fn checkpoint_validation_rejects_a_missing_file() {
        let path = tempfile::tempdir().unwrap().path().join("missing.pt");
        let err = validate_checkpoint(&path).unwrap_err();
        assert!(err.to_string().contains("graph-embed train"));
    }
}
