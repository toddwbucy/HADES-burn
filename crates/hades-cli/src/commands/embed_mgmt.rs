//! Native Rust handlers for `hades embed` and `hades extract` commands.
//!
//! All commands are CLI-only — they call the embedder HTTP service, systemctl,
//! nvidia-smi, or read files directly.  `embed text` also has a dispatch arm
//! for daemon socket clients, but the CLI adapter calls the embedder directly
//! to avoid requiring an ArangoDB connection.

use std::path::Path;
use std::process::Command;

use anyhow::{bail, Context, Result};
use serde_json::json;

use hades_core::config::HadesConfig;
use hades_core::persephone::embedder_http::EmbedderHttpClient;

// ── embed text (direct HTTP, no ArangoDB needed) ─────────────────────────

/// `hades embed text TEXT [--format F]`
pub async fn run_embed_text(config: &HadesConfig, text: &str, format: &str) -> Result<()> {
    let client = EmbedderHttpClient::new(&config.embedding.service.socket);
    let result = client
        .embed_text(text, "retrieval.passage")
        .await
        .context("embedding service error")?;

    match format {
        "raw" => {
            // Output just the embedding vector for piping
            println!("{}", serde_json::to_string(&result.embedding)?);
        }
        _ => {
            let preview_len = 10.min(result.embedding.len());
            let text_preview: String = if text.chars().count() > 100 {
                let mut s: String = text.chars().take(100).collect();
                s.push_str("...");
                s
            } else {
                text.to_string()
            };
            let output = json!({
                "text": text_preview,
                "dimension": result.dimension,
                "model": result.model,
                "embedding": result.embedding,
                "embedding_preview": &result.embedding[..preview_len],
                "embedding_truncated": result.embedding.len() > preview_len,
                "duration_ms": result.duration_ms,
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
    }
    Ok(())
}

// ── embed service status (CLI-only) ──────────────────────────────────────

/// `hades embed service status`
pub async fn run_service_status(config: &HadesConfig) -> Result<()> {
    let socket = &config.embedding.service.socket;
    let client = EmbedderHttpClient::new(socket);

    match client.health().await {
        Ok(health) => {
            let result = json!({
                "service": "hades-embedder",
                "status": health.status,
                "model_loaded": health.model_loaded,
                "device": health.device,
                "model_name": health.model_name,
                "uptime_seconds": health.uptime_seconds,
                "socket_path": socket,
            });
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Err(_) => {
            let result = json!({
                "service": "hades-embedder",
                "status": "stopped",
                "model_loaded": false,
                "socket_path": socket,
                "hint": "Start with: sudo systemctl start hades-embedder",
            });
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
    }
    Ok(())
}

// ── embed service start (CLI-only) ───────────────────────────────────────

/// `hades embed service start [--foreground]`
pub async fn run_service_start(config: &HadesConfig, foreground: bool) -> Result<()> {
    if foreground {
        bail!("--foreground is not yet supported; start the embedder via systemctl");
    }

    let output = Command::new("systemctl")
        .args(["start", "hades-embedder"])
        .output()
        .context("failed to run systemctl")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("failed to start service: {}", stderr.trim());
    }

    // Wait briefly for the service to initialize, then probe health
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    let client = EmbedderHttpClient::new(&config.embedding.service.socket);
    let available = client.health().await.is_ok();

    let result = json!({
        "action": "started",
        "service": "hades-embedder",
        "available": available,
        "note": if available { None } else { Some("Model loading may take 10-30 seconds") },
    });
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

// ── embed service stop (CLI-only) ──��─────────────────────────────────────

/// `hades embed service stop`
pub async fn run_service_stop() -> Result<()> {
    let output = Command::new("systemctl")
        .args(["stop", "hades-embedder"])
        .output()
        .context("failed to run systemctl")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("failed to stop service: {}", stderr.trim());
    }

    let result = json!({
        "action": "stopped",
        "service": "hades-embedder",
        "method": "systemctl",
    });
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

// ── embed gpu status (CLI-only) ──────────────────────────────────────────

/// `hades embed gpu status`
pub async fn run_gpu_status(config: &HadesConfig) -> Result<()> {
    let gpus = parse_nvidia_smi_output(&run_nvidia_smi()?);

    // Try to get embedder device from service health
    let client = EmbedderHttpClient::new(&config.embedding.service.socket);
    let embedder_device = client
        .health()
        .await
        .ok()
        .map(|h| h.device);

    let result = json!({
        "cuda_available": !gpus.is_empty(),
        "device_count": gpus.len(),
        "gpus": gpus,
        "embedder_service_device": embedder_device,
    });
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

// ── embed gpu list (CLI-only) ────────────────────────────────────────────

/// `hades embed gpu list`
pub fn run_gpu_list() -> Result<()> {
    let gpus = parse_nvidia_smi_output(&run_nvidia_smi()?);

    let result = json!({ "gpus": gpus });
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

// ── extract (CLI-only) ──────────────────────────────────────────────────

/// `hades extract FILE [--format F] [--output O]`
pub fn run_extract(file: &Path, format: &str, output: Option<&Path>) -> Result<()> {
    if !file.exists() {
        bail!("file not found: {}", file.display());
    }

    let ext = file
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let text = match ext.as_str() {
        "pdf" => extract_pdf(file)?,
        "txt" | "md" | "rst" | "text" | "markdown" => {
            std::fs::read_to_string(file)
                .with_context(|| format!("failed to read {}", file.display()))?
        }
        "html" | "htm" => {
            let raw = std::fs::read_to_string(file)
                .with_context(|| format!("failed to read {}", file.display()))?;
            strip_html_tags(&raw)
        }
        "tex" | "latex" => extract_latex(file)?,
        _ => {
            // Attempt direct read for unknown formats
            std::fs::read_to_string(file)
                .with_context(|| format!("failed to read {}", file.display()))?
        }
    };

    // Write to file if requested
    if let Some(out_path) = output {
        match format {
            "json" => {
                let json_out = json!({
                    "text": text,
                    "source_path": file.display().to_string(),
                    "format_detected": ext,
                    "text_length": text.len(),
                });
                std::fs::write(out_path, serde_json::to_string_pretty(&json_out)?)?;
            }
            _ => {
                std::fs::write(out_path, &text)?;
            }
        }
        eprintln!("Written to {}", out_path.display());
    }

    match format {
        "json" => {
            let result = json!({
                "text": text,
                "source_path": file.display().to_string(),
                "format_detected": ext,
                "text_length": text.len(),
            });
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        _ => {
            print!("{text}");
        }
    }
    Ok(())
}

// ── Shared helpers ───────────────────────────────────────────────────────

/// Run nvidia-smi and return its raw stdout.
fn run_nvidia_smi() -> Result<String> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output();

    match output {
        Ok(o) if o.status.success() => {
            Ok(String::from_utf8_lossy(&o.stdout).into_owned())
        }
        Ok(o) => {
            anyhow::bail!("nvidia-smi failed: {}", String::from_utf8_lossy(&o.stderr));
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // nvidia-smi not installed — return empty so caller can handle
            Ok(String::new())
        }
        Err(e) => Err(e.into()),
    }
}

/// Parse nvidia-smi CSV output into structured GPU info.
pub(crate) fn parse_nvidia_smi_output(csv: &str) -> Vec<serde_json::Value> {
    csv.lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| {
            let parts: Vec<&str> = line.split(',').map(|p| p.trim()).collect();
            if parts.len() >= 6 {
                Some(json!({
                    "index": safe_parse_int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": safe_parse_int(parts[2]),
                    "memory_used_mb": safe_parse_int(parts[3]),
                    "memory_free_mb": safe_parse_int(parts[4]),
                    "utilization_percent": safe_parse_int(parts[5]),
                }))
            } else {
                None
            }
        })
        .collect()
}

fn safe_parse_int(s: &str) -> serde_json::Value {
    let s = s.trim();
    if s.is_empty() || s.eq_ignore_ascii_case("n/a") {
        serde_json::Value::Null
    } else {
        s.parse::<i64>()
            .map(serde_json::Value::from)
            .unwrap_or(serde_json::Value::Null)
    }
}

/// Extract text from PDF via `pdftotext`.
fn extract_pdf(file: &Path) -> Result<String> {
    let output = Command::new("pdftotext")
        .args([file.as_os_str(), std::ffi::OsStr::new("-")])
        .output();

    match output {
        Ok(o) if o.status.success() => Ok(String::from_utf8_lossy(&o.stdout).into_owned()),
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            anyhow::bail!("pdftotext failed: {}", stderr.trim())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            anyhow::bail!(
                "pdftotext not found — install poppler-utils: sudo apt install poppler-utils"
            )
        }
        Err(e) => Err(e.into()),
    }
}

/// Extract text from LaTeX via `detex`, falling back to raw read.
fn extract_latex(file: &Path) -> Result<String> {
    let output = Command::new("detex").arg(file).output();
    match output {
        Ok(o) if o.status.success() => Ok(String::from_utf8_lossy(&o.stdout).into_owned()),
        _ => {
            // Fallback: read raw LaTeX
            std::fs::read_to_string(file)
                .with_context(|| format!("failed to read {}", file.display()))
        }
    }
}

/// Strip HTML tags with a simple regex-free approach.
fn strip_html_tags(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    for ch in html.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => result.push(ch),
            _ => {}
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_nvidia_smi_typical() {
        let csv = "0, NVIDIA RTX A6000, 49140, 1234, 47906, 15\n\
                   1, NVIDIA RTX A6000, 49140, 256, 48884, 3\n\
                   2, NVIDIA RTX 2000 Ada Generation, 16376, 8192, 8184, 45\n";
        let gpus = parse_nvidia_smi_output(csv);
        assert_eq!(gpus.len(), 3);
        assert_eq!(gpus[0]["index"], 0);
        assert_eq!(gpus[0]["name"], "NVIDIA RTX A6000");
        assert_eq!(gpus[0]["memory_total_mb"], 49140);
        assert_eq!(gpus[2]["name"], "NVIDIA RTX 2000 Ada Generation");
        assert_eq!(gpus[2]["utilization_percent"], 45);
    }

    #[test]
    fn test_parse_nvidia_smi_empty() {
        let gpus = parse_nvidia_smi_output("");
        assert!(gpus.is_empty());
    }

    #[test]
    fn test_parse_nvidia_smi_na_values() {
        let csv = "0, Tesla T4, 15360, N/A, N/A, N/A\n";
        let gpus = parse_nvidia_smi_output(csv);
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0]["memory_used_mb"], serde_json::Value::Null);
        assert_eq!(gpus[0]["utilization_percent"], serde_json::Value::Null);
    }

    #[test]
    fn test_strip_html_tags() {
        assert_eq!(
            strip_html_tags("<html><body><p>Hello <b>world</b></p></body></html>"),
            "Hello world"
        );
    }

    #[test]
    fn test_strip_html_empty() {
        assert_eq!(strip_html_tags(""), "");
        assert_eq!(strip_html_tags("plain text"), "plain text");
    }

    #[test]
    fn test_safe_parse_int() {
        assert_eq!(safe_parse_int("42"), serde_json::Value::from(42));
        assert_eq!(safe_parse_int(" 100 "), serde_json::Value::from(100));
        assert_eq!(safe_parse_int("N/A"), serde_json::Value::Null);
        assert_eq!(safe_parse_int(""), serde_json::Value::Null);
    }
}
