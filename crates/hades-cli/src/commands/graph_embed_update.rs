//! Native Rust implementation of the `hades graph-embed update` command.
//!
//! Incrementally re-embeds the knowledge graph without retraining:
//! 1. Load graph from ArangoDB
//! 2. Serialize graph structure to safetensors (no edge splits)
//! 3. Load previously trained RGCN checkpoint on the GPU service
//! 4. Load the current graph onto the GPU
//! 5. Single forward pass → embedding vectors
//! 6. Export embeddings back to ArangoDB

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde_json::json;
use tracing::{info, warn};

use hades_core::config::HadesConfig;
use hades_core::db::ArangoPool;
use hades_core::graph::{ExportConfig, decode_f32_embeddings, export_embeddings};
use hades_core::persephone::training::{TrainingClient, TrainingClientConfig};

/// Run the `graph-embed update` command.
pub async fn run(
    config: &HadesConfig,
    export_to: Option<&str>,
    checkpoint_dir: &str,
) -> Result<()> {
    // ── Source database (read-only) ─────────────────────────────────
    let source_pool = ArangoPool::from_config(config)
        .context("failed to connect to source ArangoDB")?;

    info!(db = %source_pool.database(), "connected to source database");

    // ── Preflight export target ─────────────────────────────────────
    let export_pool = if let Some(target_db) = export_to {
        let mut export_config = config.clone();
        export_config.database.name = target_db.to_string();
        export_config.require_writable_database()?;
        ArangoPool::from_config(&export_config)
            .context("failed to connect to export target database")?
    } else {
        config.require_writable_database()?;
        source_pool.clone()
    };

    info!(db = %export_pool.database(), "export target validated");

    // ── Validate checkpoint exists ──────────────────────────────────
    let checkpoint_path = PathBuf::from(checkpoint_dir).join("best.pt");
    if !checkpoint_path.exists() {
        anyhow::bail!(
            "no trained model found at {}. Run `graph-embed train` first.",
            checkpoint_path.display()
        );
    }

    // ── Connect to training service ─────────────────────────────────
    let training_client = TrainingClient::connect(TrainingClientConfig::default())
        .await
        .context("failed to connect to Persephone training service")?;

    // ── Load graph from ArangoDB ────────────────────────────────────
    info!("loading graph from ArangoDB");
    let (graph, id_map) = hades_core::graph::load(&source_pool).await?;

    info!(
        num_nodes = graph.num_nodes,
        num_edges = graph.num_edges,
        "graph loaded"
    );

    // ── Serialize graph for inference ───────────────────────────────
    let safetensors_dir = PathBuf::from(checkpoint_dir);
    std::fs::create_dir_all(&safetensors_dir)
        .context("failed to create checkpoint directory")?;
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
        "model checkpoint loaded"
    );

    training_client
        .load_graph(&safetensors_path)
        .await
        .context("failed to load graph onto GPU")?;

    // ── Generate embeddings (single forward pass) ───────────────────
    let embeddings_path = safetensors_dir.join("embeddings.bin");
    let emb_result = training_client
        .get_embeddings(Some(&embeddings_path))
        .await
        .context("failed to generate embeddings")?;

    let emb_bytes = tokio::fs::read(&embeddings_path)
        .await
        .context("failed to read embeddings file")?;
    let embeddings = decode_f32_embeddings(&emb_bytes)
        .context("failed to decode embedding bytes")?;

    info!(
        num_nodes = emb_result.num_nodes,
        embed_dim = emb_result.embed_dim,
        "embeddings generated"
    );

    // ── Export embeddings to ArangoDB ────────────────────────────────
    info!(db = %export_pool.database(), "exporting embeddings");

    let export_result = export_embeddings(
        &export_pool,
        &id_map,
        &embeddings,
        emb_result.embed_dim as usize,
        &ExportConfig::default(),
    )
    .await
    .context("embedding export failed")?;

    let export_count = export_result.total_exported;
    let expected_total = id_map.len();
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
    let output = json!({
        "status": "success",
        "graph": {
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
        },
        "model": {
            "checkpoint": checkpoint_path.to_string_lossy(),
            "num_parameters": init.num_parameters,
            "device": init.device,
        },
        "embeddings": {
            "num_nodes": emb_result.num_nodes,
            "embed_dim": emb_result.embed_dim,
        },
        "export": {
            "count": export_count,
            "target_db": export_to.unwrap_or(config.effective_database()),
        },
    });

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
