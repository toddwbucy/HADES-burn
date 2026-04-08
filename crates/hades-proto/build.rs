use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR not set");
    let proto_root = PathBuf::from(manifest_dir)
        .join("../../proto")
        .canonicalize()
        .expect("proto/ directory not found — expected at workspace root");

    let protos: Vec<PathBuf> = [
        "persephone/common/common.proto",
        "persephone/embedding/embedding.proto",
        "persephone/extraction/extraction.proto",
        "persephone/training/training.proto",
    ]
    .iter()
    .map(|p| proto_root.join(p))
    .collect();

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&protos, &[&proto_root])?;

    // Re-run if any proto file changes
    println!("cargo:rerun-if-changed={}", proto_root.display());

    Ok(())
}
