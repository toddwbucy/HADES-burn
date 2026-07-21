//! `hades tools` — external analyzer inventory and health.
//!
//! The enrichment analyzers (rust-analyzer, gopls) are external binaries, and
//! "is it installed" has no directory-independent answer on a rustup system:
//! the shim resolves per rust-toolchain.toml, so the same command can work in
//! one directory and die in another (#164, #167). `tools status` answers the
//! question the way ingest will actually ask it: resolve each analyzer through
//! the same precedence ingest uses (config/env override, then PATH) and probe
//! it FROM a stated workspace directory.
//!
//! `tools install` puts HADES-owned, standalone release binaries in the
//! managed tools directory (`HADES_TOOLS_DIR` or
//! `~/.local/share/hades/tools`). Standalone binaries are not rustup
//! shims, so they resolve identically from every directory — the whole
//! point of #167. Resolution order everywhere: config/env pin → managed
//! dir → PATH.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::json;

use hades_core::config::HadesConfig;

use super::output::{self, OutputFormat};

#[derive(Debug, clap::Subcommand)]
pub enum ToolsCmd {
    /// Report each analyzer's resolution and a live probe from a workspace.
    Status {
        /// Directory to probe FROM (the shim resolves per-directory). Defaults
        /// to the current directory — pass the ingest root you intend to use.
        #[arg(long)]
        workspace: Option<PathBuf>,
    },

    /// Install a HADES-managed analyzer binary into the tools directory.
    Install {
        /// Which analyzer: rust-analyzer (GitHub release download) or
        /// gopls (built via `go install`; requires a Go toolchain).
        tool: String,

        /// Release tag (rust-analyzer, e.g. 2026-07-14) or module version
        /// (gopls, e.g. v0.21.0). Defaults to latest.
        #[arg(long)]
        version: Option<String>,

        /// Install a rust-analyzer asset even when the release carries no
        /// sha256 digest (default: refuse — no integrity check at all).
        #[arg(long)]
        allow_unverified: bool,
    },
}

pub fn run_status(config: &HadesConfig, workspace: Option<PathBuf>) -> Result<()> {
    let ws = match workspace {
        Some(w) => w
            .canonicalize()
            .with_context(|| format!("workspace not found: {}", w.display()))?,
        None => std::env::current_dir().context("cannot read current directory")?,
    };

    // Failure is fatal only for a CONFIGURED analyzer: an explicit pin that
    // does not probe is broken operator intent. An unconfigured analyzer
    // missing from PATH is inventory (a Rust-only box without gopls is
    // healthy), reported but not fatal — enforcement for analyzers a
    // workspace actually needs lives in the ingest preflight.
    let mut fatal = false;
    let mut report = serde_json::Map::new();
    for (name, configured) in [
        ("rust-analyzer", config.analyzers.rust_analyzer.as_deref()),
        ("gopls", config.analyzers.gopls.as_deref()),
    ] {
        let probe = hades_core::code::lsp::resolve_and_probe(name, configured, &ws);
        let entry = match probe.outcome {
            Ok(version) => json!({
                "command": probe.command, "source": probe.source, "ok": true,
                "version": version,
            }),
            Err(e) => {
                if probe.configured {
                    fatal = true;
                }
                json!({
                    "command": probe.command, "source": probe.source, "ok": false,
                    "fatal": probe.configured, "error": e,
                })
            }
        };
        report.insert(name.to_string(), entry);
    }
    // libclang is dlopen'd by the clang crate at analysis time, not spawned,
    // so a spawn probe cannot exercise it. Reported for inventory honesty.
    report.insert(
        "libclang".to_string(),
        json!({ "source": "dlopen at analysis time", "ok": null,
                "note": "not probed by tools status; C/C++/CUDA analysis degrades gracefully when absent" }),
    );

    output::print_output_with_success(
        "tools.status",
        json!({ "workspace": ws.display().to_string(), "analyzers": report }),
        &OutputFormat::Json,
        !fatal,
    );
    if fatal {
        anyhow::bail!(
            "a configured analyzer failed its probe from {}",
            ws.display()
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// tools install
// ---------------------------------------------------------------------------

/// The release-asset target triple for this build.
fn target_triple() -> Result<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => Ok("x86_64-unknown-linux-gnu"),
        ("linux", "aarch64") => Ok("aarch64-unknown-linux-gnu"),
        ("macos", "x86_64") => Ok("x86_64-apple-darwin"),
        ("macos", "aarch64") => Ok("aarch64-apple-darwin"),
        (os, arch) => anyhow::bail!("no known rust-analyzer release asset for {os}/{arch}"),
    }
}

/// Extract the hex digest from a GitHub asset `digest` field ("sha256:HEX").
fn parse_sha256_digest(digest: &str) -> Option<&str> {
    digest.strip_prefix("sha256:").filter(|h| h.len() == 64)
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Atomically place `bytes` as an executable at `dest`.
fn install_executable(dest: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = dest.with_extension("tmp");
    std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755))?;
    }
    std::fs::rename(&tmp, dest).with_context(|| format!("installing {}", dest.display()))?;
    Ok(())
}

/// Read-modify-write the manifest recording what is installed and from where.
///
/// A present-but-unparseable manifest is an error, not an empty map —
/// silently defaulting would drop the OTHER tool's record on the next
/// install. The write goes through tmp+rename like the binary itself,
/// so a crash mid-write cannot produce the corrupt file that path
/// would then destroy.
fn record_manifest(dir: &Path, tool: &str, entry: serde_json::Value) -> Result<()> {
    let path = dir.join("manifest.json");
    let mut manifest: serde_json::Map<String, serde_json::Value> = match std::fs::read(&path) {
        Ok(bytes) => serde_json::from_slice(&bytes).with_context(|| {
            format!(
                "{} exists but is not valid JSON — refusing to overwrite it \
                 (move it aside and re-run to rebuild)",
                path.display()
            )
        })?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Default::default(),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    manifest.insert(tool.to_string(), entry);
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&manifest)?)
        .with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, &path).with_context(|| format!("installing {}", path.display()))?;
    Ok(())
}

/// Download and install a rust-analyzer release binary.
///
/// The asset digest comes from the same GitHub release API as the
/// download URL, so verification catches corruption and truncation, not
/// a compromised GitHub. The installed tag is recorded in the manifest —
/// determinism comes from `--version <tag>` plus the manifest, not from
/// a hardcoded default that would go stale weekly.
async fn install_rust_analyzer(
    dir: &Path,
    version: Option<&str>,
    allow_unverified: bool,
) -> Result<serde_json::Value> {
    let triple = target_triple()?;
    let release_url = match version {
        Some(tag) => {
            format!("https://api.github.com/repos/rust-lang/rust-analyzer/releases/tags/{tag}")
        }
        None => "https://api.github.com/repos/rust-lang/rust-analyzer/releases/latest".to_string(),
    };
    let client = reqwest::Client::builder()
        .user_agent("hades-tools-install")
        .build()?;
    let release: serde_json::Value = client
        .get(&release_url)
        .send()
        .await?
        .error_for_status()
        .with_context(|| format!("fetching release metadata from {release_url}"))?
        .json()
        .await?;
    let tag = release["tag_name"]
        .as_str()
        .context("release metadata has no tag_name")?
        .to_string();
    let asset_name = format!("rust-analyzer-{triple}.gz");
    let asset = release["assets"]
        .as_array()
        .and_then(|a| {
            a.iter()
                .find(|x| x["name"].as_str() == Some(asset_name.as_str()))
        })
        .with_context(|| format!("release {tag} has no asset {asset_name}"))?;
    let url = asset["browser_download_url"]
        .as_str()
        .context("asset has no download url")?;

    tracing::info!(%tag, %url, "downloading rust-analyzer");
    let resp = client.get(url).send().await?.error_for_status()?;
    // Sanity-cap the download: release assets are tens of MiB, and
    // `bytes()` buffers the whole body in memory.
    const MAX_ASSET_BYTES: u64 = 256 * 1024 * 1024;
    if let Some(len) = resp.content_length() {
        anyhow::ensure!(
            len <= MAX_ASSET_BYTES,
            "release asset {asset_name} reports {len} bytes, over the {MAX_ASSET_BYTES} byte cap"
        );
    }
    let bytes = resp.bytes().await?;
    anyhow::ensure!(
        bytes.len() as u64 <= MAX_ASSET_BYTES,
        "release asset {asset_name} is {} bytes, over the {MAX_ASSET_BYTES} byte cap",
        bytes.len()
    );

    // Verify against the API-reported digest. Absent digest fails
    // closed: an executable with no corruption check at all is exactly
    // the case that must not install silently. --allow-unverified opts
    // in explicitly, and the manifest records the downgrade.
    let actual = sha256_hex(&bytes);
    let verified = match asset["digest"].as_str().and_then(parse_sha256_digest) {
        Some(expected) => {
            anyhow::ensure!(
                expected == actual,
                "sha256 mismatch for {asset_name}: expected {expected}, got {actual}"
            );
            true
        }
        None if allow_unverified => {
            tracing::warn!(
                "release asset carries no digest; installing UNVERIFIED per --allow-unverified"
            );
            false
        }
        None => anyhow::bail!(
            "release {tag} asset {asset_name} carries no sha256 digest — refusing to \
             install an executable with no integrity check. Re-run with \
             --allow-unverified to accept it, or pin a release tag that has one"
        ),
    };

    let mut decoder = flate2::read::GzDecoder::new(&bytes[..]);
    let mut binary = Vec::new();
    std::io::Read::read_to_end(&mut decoder, &mut binary)
        .context("decompressing rust-analyzer release asset")?;

    let dest = dir.join("rust-analyzer");
    install_executable(&dest, &binary)?;

    Ok(json!({
        "version": tag,
        "asset": asset_name,
        "sha256": actual,
        "digest_verified": verified,
        "source_url": url,
        "installed": chrono::Utc::now().to_rfc3339(),
        "path": dest.display().to_string(),
    }))
}

/// Build gopls into the tools directory via `go install`.
///
/// gopls publishes no prebuilt binaries, so this needs a Go toolchain —
/// which is the #167 boundary working as intended: HADES owns where its
/// analysis tools live, the machine that ingests Go code owns Go.
fn install_gopls(dir: &Path, version: Option<&str>) -> Result<serde_json::Value> {
    let spec = format!("golang.org/x/tools/gopls@{}", version.unwrap_or("latest"));
    tracing::info!(%spec, "building gopls via go install");
    let output = std::process::Command::new("go")
        .args(["install", &spec])
        .env("GOBIN", dir)
        .output()
        .context(
            "failed to run `go install` — installing gopls requires a Go toolchain \
             (gopls publishes no prebuilt binaries). Install Go, or pin an existing \
             gopls via analyzers.gopls in hades.yaml",
        )?;
    anyhow::ensure!(
        output.status.success(),
        "go install {spec} failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let dest = dir.join("gopls");
    // Capture the actual built version from the binary itself.
    let version_line = hades_core::code::lsp::preflight_binary(
        &dest.display().to_string(),
        hades_core::code::lsp::version_args_for("gopls"),
        dir,
    )
    .map_err(|e| anyhow::anyhow!("gopls built but failed its probe: {e}"))?;

    Ok(json!({
        "version": version_line,
        "requested": spec,
        "installed": chrono::Utc::now().to_rfc3339(),
        "path": dest.display().to_string(),
    }))
}

pub fn run_install(tool: &str, version: Option<&str>, allow_unverified: bool) -> Result<()> {
    // Validate before touching the filesystem, so an unknown tool name
    // has no side effects.
    anyhow::ensure!(
        matches!(tool, "rust-analyzer" | "gopls"),
        "unknown tool '{tool}': supported tools are rust-analyzer, gopls"
    );
    let dir = hades_core::code::lsp::managed_tools_dir();
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("creating tools directory {}", dir.display()))?;

    let entry = match tool {
        "rust-analyzer" => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(install_rust_analyzer(&dir, version, allow_unverified))?
        }
        "gopls" => install_gopls(&dir, version)?,
        _ => unreachable!("validated above"),
    };
    record_manifest(&dir, tool, entry.clone())?;

    // Post-install probe through the real resolution path: the managed
    // binary must now win (unless the operator has an explicit pin).
    let probe = hades_core::code::lsp::resolve_and_probe(tool, None, &dir);
    let ok = probe.outcome.is_ok();
    output::print_output_with_success(
        "tools.install",
        json!({
            "tool": tool,
            "tools_dir": dir.display().to_string(),
            "installed": entry,
            "probe": {
                "command": probe.command,
                "source": probe.source,
                "outcome": match probe.outcome { Ok(v) => v, Err(e) => e },
            },
        }),
        &OutputFormat::Json,
        ok,
    );
    anyhow::ensure!(ok, "{tool} installed but failed its post-install probe");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triple_is_known_on_this_host() {
        // The test host is one of the supported platforms.
        assert!(target_triple().is_ok());
    }

    #[test]
    fn digest_parsing() {
        let hex = "a".repeat(64);
        assert_eq!(
            parse_sha256_digest(&format!("sha256:{hex}")),
            Some(hex.as_str())
        );
        assert_eq!(parse_sha256_digest("sha512:abc"), None);
        assert_eq!(parse_sha256_digest("sha256:short"), None);
    }

    #[test]
    fn sha256_matches_known_vector() {
        // sha256("abc")
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn gunzip_roundtrip() {
        use std::io::{Read, Write};
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(b"fake analyzer binary").unwrap();
        let gz = enc.finish().unwrap();
        let mut dec = flate2::read::GzDecoder::new(&gz[..]);
        let mut out = Vec::new();
        dec.read_to_end(&mut out).unwrap();
        assert_eq!(out, b"fake analyzer binary");
    }

    #[test]
    fn manifest_roundtrip_preserves_other_tools() {
        let dir = tempfile::tempdir().unwrap();
        record_manifest(
            dir.path(),
            "rust-analyzer",
            json!({"version": "2026-07-14"}),
        )
        .unwrap();
        record_manifest(dir.path(), "gopls", json!({"version": "v0.21.0"})).unwrap();
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["rust-analyzer"]["version"], "2026-07-14");
        assert_eq!(manifest["gopls"]["version"], "v0.21.0");
    }

    #[test]
    fn install_executable_is_atomic_and_executable() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("tool");
        install_executable(&dest, b"#!/bin/sh\necho hi\n").unwrap();
        assert!(dest.is_file());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&dest).unwrap().permissions().mode();
            assert_eq!(mode & 0o111, 0o111, "must be executable");
        }
        assert!(
            !dest.with_extension("tmp").exists(),
            "no tmp file left behind"
        );
    }

    #[test]
    fn corrupt_manifest_is_refused_not_wiped() {
        let dir = tempfile::tempdir().unwrap();
        record_manifest(dir.path(), "gopls", json!({"version": "v0.21.0"})).unwrap();
        std::fs::write(dir.path().join("manifest.json"), b"{not json").unwrap();
        let err =
            record_manifest(dir.path(), "rust-analyzer", json!({"version": "x"})).unwrap_err();
        assert!(err.to_string().contains("refusing to overwrite"));
        // The corrupt file is left in place for the operator, not clobbered.
        assert_eq!(
            std::fs::read(dir.path().join("manifest.json")).unwrap(),
            b"{not json"
        );
        assert!(
            !dir.path().join("manifest.json.tmp").exists(),
            "no tmp residue"
        );
    }

    #[test]
    fn unknown_tool_is_refused() {
        let err = run_install("clangd", None, false).unwrap_err();
        assert!(err.to_string().contains("supported tools"));
    }
}
