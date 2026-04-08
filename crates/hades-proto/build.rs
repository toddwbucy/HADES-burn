fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR not set");
    let proto_root = std::path::Path::new(&manifest_dir)
        .join("../../proto")
        .canonicalize()
        .expect("proto/ directory not found — expected at workspace root");

    let proto_root_str = proto_root.to_str().expect("proto path is not valid UTF-8");

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(
            &[
                format!("{proto_root_str}/persephone/common/common.proto"),
                format!("{proto_root_str}/persephone/embedding/embedding.proto"),
                format!("{proto_root_str}/persephone/extraction/extraction.proto"),
                format!("{proto_root_str}/persephone/training/training.proto"),
            ],
            &[proto_root_str],
        )?;

    // Re-run if any proto file changes
    println!("cargo:rerun-if-changed={proto_root_str}/");

    Ok(())
}
