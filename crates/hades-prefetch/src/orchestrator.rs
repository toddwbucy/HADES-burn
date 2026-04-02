//! Training orchestrator — Rust-side epoch loop driving Python RGCN.
//!
//! Connects the [`Prefetcher`](crate::prefetcher::Prefetcher) (which
//! pre-computes negative samples) with the
//! [`TrainingClient`](hades_core::persephone::training::TrainingClient)
//! (which issues gRPC RPCs to the Python GPU process).
//!
//! ## Training lifecycle
//!
//! 1. **Init model** — create RGCN encoder + link predictor on GPU.
//! 2. **Load graph** — Python mmaps the safetensors file into GPU memory.
//! 3. **Epoch loop** — for each epoch:
//!    - Receive pre-computed negative samples from the prefetcher.
//!    - `TrainStep` RPC: forward → loss → backward → optimizer.step().
//!    - `Evaluate` RPC on validation split (every `val_every` epochs).
//!    - Early stopping: checkpoint on best val loss, stop after `patience`.
//! 4. **Restore best** — load the best checkpoint.
//! 5. **Test evaluation** — final assessment on the held-out test split.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tracing::{debug, info};

use hades_core::graph::types::GraphData;
use hades_core::persephone::training::{EvalResult, TrainingClient, TrainingError};
use hades_proto::training::{ModelConfig, OptimizerConfig};

use crate::prefetcher::{PrefetchConfig, PrefetchError, Prefetcher};
use crate::tensor::{EdgeSplit, negative_sample};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from the training orchestrator.
#[derive(Debug, thiserror::Error)]
pub enum OrchestratorError {
    /// gRPC/training service error.
    #[error("training service error: {0}")]
    Training(#[from] TrainingError),

    /// Prefetcher startup error.
    #[error("prefetcher error: {0}")]
    Prefetch(#[from] PrefetchError),

    /// Filesystem error (e.g. creating checkpoint directory).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Training configuration — mirrors Python `TrainConfig` defaults.
///
/// Covers model architecture, optimizer, training loop control, and
/// prefetcher settings. Defaults match the Python HADES trainer:
/// hidden_dim=256, embed_dim=128, num_bases=21, dropout=0.2,
/// lr=0.01, weight_decay=5e-4, epochs=200, patience=20.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    // -- Model architecture --
    /// Hidden layer dimension.
    pub hidden_dim: u32,
    /// Output structural embedding dimension.
    pub embed_dim: u32,
    /// Number of basis matrices for RGCN decomposition.
    pub num_bases: u32,
    /// Dropout rate.
    pub dropout: f32,

    // -- Optimizer --
    /// Learning rate.
    pub lr: f32,
    /// L2 regularization weight.
    pub weight_decay: f32,

    // -- Training loop --
    /// Maximum number of training epochs.
    pub epochs: usize,
    /// Early stopping patience (epochs without val improvement).
    pub patience: usize,
    /// Validate every N epochs (0 = never validate, skip early stopping).
    pub val_every: usize,

    // -- Negative sampling / prefetch --
    /// Ratio of negative to positive samples per epoch.
    pub neg_sampling_ratio: f64,
    /// Number of epochs to prefetch ahead.
    pub prefetch_depth: usize,

    // -- Device --
    /// PyTorch device string (e.g. "cuda:0", "cpu").
    pub device: String,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 256,
            embed_dim: 128,
            num_bases: 21,
            dropout: 0.2,
            lr: 0.01,
            weight_decay: 5e-4,
            epochs: 200,
            patience: 20,
            val_every: 1,
            neg_sampling_ratio: 1.0,
            prefetch_depth: 2,
            device: "cuda:0".to_string(),
        }
    }
}

impl TrainConfig {
    /// Build the protobuf `ModelConfig` from this training config.
    fn model_config(&self, num_relations: u32, num_collection_types: u32) -> ModelConfig {
        ModelConfig {
            num_relations,
            num_collection_types,
            hidden_dim: self.hidden_dim,
            embed_dim: self.embed_dim,
            num_bases: self.num_bases,
            dropout: self.dropout,
        }
    }

    /// Build the protobuf `OptimizerConfig` from this training config.
    fn optimizer_config(&self) -> OptimizerConfig {
        OptimizerConfig {
            learning_rate: self.lr,
            weight_decay: self.weight_decay,
        }
    }

    /// Build the `PrefetchConfig` from this training config.
    fn prefetch_config(&self) -> PrefetchConfig {
        PrefetchConfig {
            prefetch_depth: self.prefetch_depth,
            neg_sampling_ratio: self.neg_sampling_ratio,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-epoch metrics
// ---------------------------------------------------------------------------

/// Metrics collected for a single training epoch.
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    /// Zero-based epoch index.
    pub epoch: usize,
    /// Training loss (BCE).
    pub train_loss: f32,
    /// Training accuracy.
    pub train_accuracy: f32,
    /// Validation loss (present only on validation epochs).
    pub val_loss: Option<f32>,
    /// Validation accuracy.
    pub val_accuracy: Option<f32>,
    /// Validation AUC.
    pub val_auc: Option<f32>,
}

// ---------------------------------------------------------------------------
// Training result
// ---------------------------------------------------------------------------

/// Summary of a completed training run.
#[derive(Debug, Clone)]
pub struct TrainResult {
    /// Epoch with the best validation loss.
    pub best_epoch: usize,
    /// Best validation loss achieved.
    pub best_val_loss: f32,
    /// Test split evaluation (after restoring best checkpoint).
    pub final_test: EvalResult,
    /// Per-epoch metrics log.
    pub epoch_metrics: Vec<EpochMetrics>,
    /// Wall-clock training duration.
    pub elapsed: Duration,
    /// Path where the best checkpoint was saved.
    pub checkpoint_path: String,
    /// Whether early stopping was triggered.
    pub early_stopped: bool,
    /// Total epochs actually executed.
    pub total_epochs: usize,
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

/// Training orchestrator — drives the RGCN training loop.
///
/// Owns a [`TrainingClient`] connection and a [`TrainConfig`].
/// Call [`train()`](Self::train) with graph data and paths to execute
/// the full training lifecycle.
pub struct Orchestrator {
    client: TrainingClient,
    config: TrainConfig,
}

impl Orchestrator {
    /// Create a new orchestrator with an existing client connection.
    pub fn new(client: TrainingClient, config: TrainConfig) -> Self {
        Self { client, config }
    }

    /// Access the training configuration.
    pub fn config(&self) -> &TrainConfig {
        &self.config
    }

    /// Execute the full training lifecycle.
    ///
    /// 1. Init model + load graph onto GPU.
    /// 2. Train with early stopping and periodic validation.
    /// 3. Restore best checkpoint and evaluate on the test split.
    ///
    /// The `safetensors_path` must point to the file produced by
    /// [`prepare_and_serialize`](crate::tensor::prepare_and_serialize) or
    /// [`prepare_training_data`](crate::prefetcher::prepare_training_data).
    pub async fn train(
        &self,
        graph: Arc<GraphData>,
        split: Arc<EdgeSplit>,
        safetensors_path: &Path,
        checkpoint_dir: &Path,
    ) -> Result<TrainResult, OrchestratorError> {
        let cfg = &self.config;

        // Ensure checkpoint directory exists
        std::fs::create_dir_all(checkpoint_dir)?;

        // ── Phase 1: Init ────────────────────────────────────────────
        let num_collection_types = graph.collection_names.len() as u32;
        let init = self
            .client
            .init_model(
                cfg.model_config(graph.num_relations as u32, num_collection_types),
                cfg.optimizer_config(),
                &cfg.device,
            )
            .await?;

        info!(
            params = init.num_parameters,
            device = %init.device,
            "model initialized"
        );

        // ── Phase 2: Load graph onto GPU ─────────────────────────────
        let load = self.client.load_graph(safetensors_path).await?;
        info!(
            num_nodes = load.num_nodes,
            num_edges = load.num_edges,
            gpu_mb = load.gpu_memory_bytes as f64 / (1024.0 * 1024.0),
            "graph loaded onto GPU"
        );

        // ── Phase 3: Prefetcher ──────────────────────────────────────
        let mut prefetcher = Prefetcher::start(
            Arc::clone(&graph),
            Arc::clone(&split),
            cfg.prefetch_config(),
            Some(cfg.epochs),
        )?;

        // ── Phase 4: Epoch loop ──────────────────────────────────────
        let checkpoint_path = checkpoint_dir.join("best.pt");
        let start = Instant::now();

        let mut best_val_loss = f32::INFINITY;
        let mut best_epoch: usize = 0;
        let mut checkpoint_saved = false;
        let mut epoch_metrics = Vec::with_capacity(cfg.epochs);
        let mut early_stopped = false;
        let mut total_epochs: usize = 0;

        info!(
            epochs = cfg.epochs,
            patience = cfg.patience,
            val_every = cfg.val_every,
            "starting training"
        );

        while let Some(batch) = prefetcher.next_batch().await {
            let epoch = batch.epoch;
            total_epochs = epoch + 1;

            // Train step
            let step = self
                .client
                .train_step(
                    split.train_idx.clone(),
                    batch.train_neg.src,
                    batch.train_neg.dst,
                )
                .await?;

            // Validate (if scheduled)
            let do_val = cfg.val_every > 0 && (epoch % cfg.val_every == 0);
            let val = if do_val {
                let eval = self
                    .client
                    .evaluate(
                        split.val_idx.clone(),
                        batch.val_neg.src,
                        batch.val_neg.dst,
                    )
                    .await?;
                Some(eval)
            } else {
                None
            };

            // Record metrics
            epoch_metrics.push(EpochMetrics {
                epoch,
                train_loss: step.loss,
                train_accuracy: step.accuracy,
                val_loss: val.map(|v| v.loss),
                val_accuracy: val.map(|v| v.accuracy),
                val_auc: val.map(|v| v.auc),
            });

            // Periodic logging (epoch 0 and every 10th)
            if epoch == 0 || (epoch + 1) % 10 == 0 {
                if let Some(eval) = val {
                    info!(
                        epoch,
                        train_loss = format!("{:.4}", step.loss),
                        val_loss = format!("{:.4}", eval.loss),
                        val_auc = format!("{:.3}", eval.auc),
                        "epoch"
                    );
                } else {
                    info!(
                        epoch,
                        train_loss = format!("{:.4}", step.loss),
                        "epoch"
                    );
                }
            }

            // Early stopping check
            if let Some(eval) = val {
                if eval.loss < best_val_loss {
                    best_val_loss = eval.loss;
                    best_epoch = epoch;
                    self.client.checkpoint(&checkpoint_path).await?;
                    checkpoint_saved = true;
                    debug!(epoch, val_loss = eval.loss, "new best — checkpoint saved");
                } else if epoch.saturating_sub(best_epoch) >= cfg.patience {
                    info!(
                        epoch,
                        best_epoch,
                        patience = cfg.patience,
                        "early stopping triggered"
                    );
                    early_stopped = true;
                    break;
                }
            }
        }

        // ── Phase 5: Restore best model ──────────────────────────────
        if checkpoint_saved {
            self.client
                .load_checkpoint(&checkpoint_path, Some(&cfg.device))
                .await?;
            info!(best_epoch, "restored best checkpoint");
        }

        // ── Phase 6: Test evaluation ─────────────────────────────────
        let graph_ref = Arc::clone(&graph);
        let test_count = split.test_idx.len();
        let test_neg = tokio::task::spawn_blocking(move || {
            negative_sample(&graph_ref, test_count)
        })
        .await
        .expect("test negative sampling panicked");

        let test_eval = self
            .client
            .evaluate(split.test_idx.clone(), test_neg.src, test_neg.dst)
            .await?;

        info!(
            test_loss = format!("{:.4}", test_eval.loss),
            test_acc = format!("{:.3}", test_eval.accuracy),
            test_auc = format!("{:.3}", test_eval.auc),
            total_epochs,
            elapsed_secs = start.elapsed().as_secs(),
            "training complete"
        );

        Ok(TrainResult {
            best_epoch,
            best_val_loss,
            final_test: test_eval,
            epoch_metrics,
            elapsed: start.elapsed(),
            checkpoint_path: checkpoint_path.to_string_lossy().into_owned(),
            early_stopped,
            total_epochs,
        })
    }
}

impl std::fmt::Debug for Orchestrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Orchestrator")
            .field("config", &self.config)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use hades_core::graph::schema::NUM_RELATIONS;

    #[test]
    fn test_train_config_defaults() {
        let cfg = TrainConfig::default();

        // Match Python TrainConfig defaults
        assert_eq!(cfg.hidden_dim, 256);
        assert_eq!(cfg.embed_dim, 128);
        assert_eq!(cfg.num_bases, 21);
        assert!((cfg.dropout - 0.2).abs() < f32::EPSILON);
        assert!((cfg.lr - 0.01).abs() < f32::EPSILON);
        assert!((cfg.weight_decay - 5e-4).abs() < f32::EPSILON);
        assert_eq!(cfg.epochs, 200);
        assert_eq!(cfg.patience, 20);
        assert_eq!(cfg.val_every, 1);
        assert!((cfg.neg_sampling_ratio - 1.0).abs() < f64::EPSILON);
        assert_eq!(cfg.prefetch_depth, 2);
        assert_eq!(cfg.device, "cuda:0");
    }

    #[test]
    fn test_model_config_conversion() {
        let cfg = TrainConfig::default();
        let mc = cfg.model_config(NUM_RELATIONS as u32, 5);

        assert_eq!(mc.num_relations, NUM_RELATIONS as u32);
        assert_eq!(mc.num_collection_types, 5);
        assert_eq!(mc.hidden_dim, 256);
        assert_eq!(mc.embed_dim, 128);
        assert_eq!(mc.num_bases, 21);
        assert!((mc.dropout - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_optimizer_config_conversion() {
        let cfg = TrainConfig::default();
        let oc = cfg.optimizer_config();

        assert!((oc.learning_rate - 0.01).abs() < f32::EPSILON);
        assert!((oc.weight_decay - 5e-4).abs() < f32::EPSILON);
    }

    #[test]
    fn test_prefetch_config_conversion() {
        let cfg = TrainConfig {
            prefetch_depth: 4,
            neg_sampling_ratio: 2.0,
            ..Default::default()
        };
        let pc = cfg.prefetch_config();

        assert_eq!(pc.prefetch_depth, 4);
        assert!((pc.neg_sampling_ratio - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_epoch_metrics_construction() {
        let m = EpochMetrics {
            epoch: 42,
            train_loss: 0.5,
            train_accuracy: 0.75,
            val_loss: Some(0.6),
            val_accuracy: Some(0.72),
            val_auc: Some(0.85),
        };
        assert_eq!(m.epoch, 42);
        assert_eq!(m.val_auc, Some(0.85));
    }

    #[test]
    fn test_epoch_metrics_no_val() {
        let m = EpochMetrics {
            epoch: 1,
            train_loss: 0.7,
            train_accuracy: 0.6,
            val_loss: None,
            val_accuracy: None,
            val_auc: None,
        };
        assert!(m.val_loss.is_none());
    }

    #[test]
    fn test_train_result_fields() {
        let result = TrainResult {
            best_epoch: 50,
            best_val_loss: 0.3,
            final_test: EvalResult {
                loss: 0.35,
                accuracy: 0.82,
                auc: 0.91,
            },
            epoch_metrics: vec![],
            elapsed: Duration::from_secs(120),
            checkpoint_path: "/tmp/best.pt".into(),
            early_stopped: true,
            total_epochs: 70,
        };
        assert!(result.early_stopped);
        assert_eq!(result.total_epochs, 70);
        assert!(result.best_val_loss < result.final_test.loss);
    }

    #[test]
    fn test_train_config_custom() {
        let cfg = TrainConfig {
            hidden_dim: 512,
            embed_dim: 256,
            num_bases: 10,
            dropout: 0.3,
            lr: 0.001,
            weight_decay: 1e-3,
            epochs: 100,
            patience: 10,
            val_every: 5,
            neg_sampling_ratio: 2.0,
            prefetch_depth: 4,
            device: "cuda:2".into(),
        };
        assert_eq!(cfg.hidden_dim, 512);
        assert_eq!(cfg.val_every, 5);
        assert_eq!(cfg.device, "cuda:2");
    }

    #[test]
    fn test_orchestrator_error_variants() {
        // Verify error conversion compiles (From impls)
        let io_err: OrchestratorError = std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "missing",
        )
        .into();
        assert!(matches!(io_err, OrchestratorError::Io(_)));

        let pf_err: OrchestratorError = PrefetchError::InvalidDepth.into();
        assert!(matches!(pf_err, OrchestratorError::Prefetch(_)));
    }
}
