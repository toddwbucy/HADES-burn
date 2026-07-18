//! Workstation-capability probe for #99 using a self-contained Go module.

use std::collections::HashMap;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use hades_core::code::lsp::go_symbols::GoSymbolExtractor;
use hades_core::code::lsp::{EdgeKind, GoplsSession, LspEdgeResolver};

fn gopls_available() -> bool {
    std::process::Command::new("gopls")
        .arg("version")
        .output()
        .is_ok_and(|output| output.status.success())
        || std::env::var_os("HOME")
            .map(PathBuf::from)
            .is_some_and(|home| home.join("go/bin/gopls").exists())
}

fn gopls_path() -> Option<PathBuf> {
    std::process::Command::new("sh")
        .args(["-c", "command -v gopls"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|path| PathBuf::from(path.trim()))
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join("go/bin/gopls"))
                .filter(|path| path.exists())
        })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gopls_resolves_cross_package_calls_and_interface_satisfaction() {
    if !gopls_available() {
        eprintln!("SKIP: gopls is unavailable");
        return;
    }

    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    std::fs::create_dir_all(root.join("api")).unwrap();
    std::fs::create_dir_all(root.join("worker")).unwrap();
    std::fs::write(
        root.join("go.mod"),
        "module example.test/semantic\n\ngo 1.22\n",
    )
    .unwrap();
    std::fs::write(
        root.join("api/api.go"),
        r#"package api

type Runner interface {
	Run() string
}

func Execute(r Runner) string {
	return r.Run()
}
"#,
    )
    .unwrap();
    std::fs::write(
        root.join("worker/worker.go"),
        r#"package worker

import "example.test/semantic/api"

type Worker struct{}

func (Worker) Run() string { return "done" }

func Use() string {
	return api.Execute(Worker{})
}
"#,
    )
    .unwrap();

    // Keep gopls/Go build artifacts inside the disposable fixture. This also
    // makes the probe work in read-only-home CI and sandbox environments.
    let gopls = gopls_path().expect("gopls path should be discoverable");
    let cache = root.join("gocache");
    std::fs::create_dir_all(&cache).unwrap();
    let wrapper = root.join("gopls-with-cache");
    std::fs::write(
        &wrapper,
        format!(
            "#!/bin/sh\nGOCACHE='{}' exec '{}' \"$@\"\n",
            cache.display(),
            gopls.display()
        ),
    )
    .unwrap();
    #[cfg(unix)]
    std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o755)).unwrap();

    let session = GoplsSession::start_with_options(root, wrapper.to_str(), 30)
        .await
        .expect("gopls should initialize for the fixture module");
    let extractor = GoSymbolExtractor::new(&session, true);
    let files = [root.join("api/api.go"), root.join("worker/worker.go")];
    let refs: Vec<&Path> = files.iter().map(PathBuf::as_path).collect();
    let raw = extractor.extract_module(&refs).await;

    let mut relative = HashMap::new();
    for (path, extraction) in raw {
        let path = Path::new(&path)
            .strip_prefix(root)
            .unwrap()
            .to_string_lossy()
            .into_owned();
        relative.insert(path, extraction);
    }

    let resolver = LspEdgeResolver::new(relative, "gopls");
    let documents = resolver.build_symbol_documents();
    let edges = resolver.build_edges();
    eprintln!(
        "gopls fixture: {} symbols, {} calls, {} implements",
        documents.len(),
        edges
            .iter()
            .filter(|edge| edge.kind == EdgeKind::Calls)
            .count(),
        edges
            .iter()
            .filter(|edge| edge.kind == EdgeKind::Implements)
            .count()
    );

    assert!(documents.iter().any(|document| document.name == "Execute"));
    assert!(documents.iter().any(|document| document.name == "Worker"));
    assert!(
        documents
            .iter()
            .all(|document| document.analyzer == "gopls")
    );
    assert!(
        edges.iter().any(|edge| edge.kind == EdgeKind::Calls),
        "expected at least one semantic gopls call edge"
    );
    assert!(
        edges.iter().any(|edge| edge.kind == EdgeKind::Implements),
        "expected Worker -> Runner interface satisfaction"
    );

    session.shutdown().await.unwrap();
}
