//! Native Rust implementation of the `hades graph-embed train` command.
//!
//! Replaces the Python dispatch for RGCN training.  Orchestrates:
//! 1. Graph loading from ArangoDB
//! 2. Edge splitting + safetensors serialization
//! 3. Training via Persephone gRPC (Python GPU process)
//! 4. Embedding export back to ArangoDB

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde_json::{Value, json};
use tracing::{info, warn};

use hades_core::config::HadesConfig;
use hades_core::db::ArangoPool;
use hades_core::graph::{ExportConfig, decode_f32_embeddings, export_embeddings};
use hades_core::persephone::training::{TrainingClient, TrainingClientConfig};

use hades_prefetch::{
    Orchestrator, SplitConfig, TrainConfig, prepare_training_data,
};

/// Run the `graph-embed train` command.
#[allow(clippy::too_many_arguments)]
pub async fn run(
    config: &HadesConfig,
    epochs: u32,
    dimension: u32,
    hidden_dim: u32,
    num_bases: u32,
    dropout: f32,
    lr: f32,
    weight_decay: f32,
    patience: u32,
    val_ratio: f64,
    test_ratio: f64,
    neg_ratio: f64,
    export_to: Option<&str>,
    checkpoint_dir: &str,
    val_every: usize,
    prefetch_depth: usize,
    no_export: bool,
) -> Result<()> {
    // ── Validate split ratios ───────────────────────────────────────
    if !(0.0..=1.0).contains(&val_ratio) {
        anyhow::bail!("--val-ratio must be between 0.0 and 1.0, got {val_ratio}");
    }
    if !(0.0..=1.0).contains(&test_ratio) {
        anyhow::bail!("--test-ratio must be between 0.0 and 1.0, got {test_ratio}");
    }
    if val_ratio + test_ratio >= 1.0 {
        anyhow::bail!(
            "--val-ratio ({val_ratio}) + --test-ratio ({test_ratio}) = {:.4} \
             leaves no training data (sum must be < 1.0)",
            val_ratio + test_ratio
        );
    }

    // ── Source database (read-only) ──────────────────────────────────
    let source_pool = ArangoPool::from_config(config)
        .context("failed to connect to source ArangoDB")?;

    info!(db = %source_pool.database(), "connected to source database");

    // ── Preflight export target ──────────────────────────────────────
    // Validate and connect to the export database before training so
    // misconfiguration fails fast rather than after a long training run.
    let export_pool = resolve_export_pool(config, export_to, no_export, &source_pool)?;

    // ── Prepare training data ────────────────────────────────────────
    let safetensors_dir = PathBuf::from(checkpoint_dir);
    std::fs::create_dir_all(&safetensors_dir)
        .context("failed to create checkpoint directory")?;
    let safetensors_path = safetensors_dir.join("graph.safetensors");

    let split_config = SplitConfig {
        val_ratio,
        test_ratio,
        neg_sampling_ratio: neg_ratio,
    };

    info!("loading graph and preparing training data");
    let data = prepare_training_data(&source_pool, &safetensors_path, &split_config)
        .await
        .context("failed to prepare training data")?;

    info!(
        num_nodes = data.graph.num_nodes,
        num_edges = data.graph.num_edges,
        train = data.split.train_idx.len(),
        val = data.split.val_idx.len(),
        test = data.split.test_idx.len(),
        "training data ready"
    );

    // ── Connect to training service ──────────────────────────────────
    let training_client = TrainingClient::connect(TrainingClientConfig::default())
        .await
        .context("failed to connect to Persephone training service")?;

    // ── Build orchestrator config ────────────────────────────────────
    let train_config = TrainConfig {
        hidden_dim,
        embed_dim: dimension,
        num_bases,
        dropout,
        lr,
        weight_decay,
        epochs: epochs as usize,
        patience: patience as usize,
        val_every,
        neg_sampling_ratio: neg_ratio,
        prefetch_depth,
        device: config.gpu.device.clone(),
    };

    // ── Train ────────────────────────────────────────────────────────
    let orchestrator = Orchestrator::new(training_client.clone(), train_config);
    let result = orchestrator
        .train(
            data.graph.clone(),
            data.split.clone(),
            &safetensors_path,
            &safetensors_dir,
        )
        .await
        .context("training failed")?;

    info!(
        best_epoch = result.best_epoch,
        best_val_loss = format!("{:.4}", result.best_val_loss),
        test_auc = format!("{:.3}", result.final_test.auc),
        elapsed_secs = result.elapsed.as_secs(),
        early_stopped = result.early_stopped,
        "training complete"
    );

    // ── Export embeddings ─────────────────────────────────────────────
    let mut export_count = 0;
    if let Some(pool) = &export_pool {
        // None = return embeddings inline over gRPC (no file output)
        let emb_result = training_client
            .get_embeddings(None)
            .await
            .context("failed to retrieve embeddings")?;

        let embeddings = decode_f32_embeddings(&emb_result.embeddings)
            .context("failed to decode embedding bytes")?;

        info!(
            db = %pool.database(),
            num_nodes = emb_result.num_nodes,
            embed_dim = emb_result.embed_dim,
            "exporting embeddings"
        );

        let export_result = export_embeddings(
            pool,
            &data.id_map,
            &embeddings,
            emb_result.embed_dim as usize,
            &ExportConfig::default(),
        )
        .await
        .context("embedding export failed")?;

        export_count = export_result.total_exported;
        let expected_total = data.id_map.len();
        if export_count < expected_total {
            warn!(
                db = %pool.database(),
                total_exported = export_count,
                expected_total,
                skipped = expected_total - export_count,
                "partial export — some documents were not updated"
            );
        } else {
            info!(total = export_count, "embeddings exported");
        }
    }

    // ── JSON output to stdout ────────────────────────────────────────
    let output = json!({
        "status": "success",
        "training": {
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "total_epochs": result.total_epochs,
            "early_stopped": result.early_stopped,
            "elapsed_seconds": result.elapsed.as_secs(),
            "test": {
                "loss": result.final_test.loss,
                "accuracy": result.final_test.accuracy,
                "auc": result.final_test.auc,
            },
        },
        "export": {
            "enabled": !no_export,
            "count": export_count,
            "target_db": if no_export { Value::Null } else {
                Value::String(export_to.unwrap_or(config.effective_database()).to_string())
            },
        },
        "checkpoint_path": result.checkpoint_path,
    });

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

/// Preflight the export target: validate writable DB and connect.
///
/// Returns `None` when `no_export` is true, or `Some(pool)` pointing at
/// the export target (either `export_to` DB or the source DB).
fn resolve_export_pool(
    config: &HadesConfig,
    export_to: Option<&str>,
    no_export: bool,
    source_pool: &ArangoPool,
) -> Result<Option<ArangoPool>> {
    if no_export {
        return Ok(None);
    }

    let pool = if let Some(target_db) = export_to {
        let mut export_config = config.clone();
        export_config.database.name = target_db.to_string();
        export_config.require_writable_database()?;
        ArangoPool::from_config(&export_config)
            .context("failed to connect to export target database")?
    } else {
        config.require_writable_database()?;
        // Re-use existing source connection
        source_pool.clone()
    };

    info!(db = %pool.database(), "export target validated");
    Ok(Some(pool))
}
