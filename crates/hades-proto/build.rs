fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_root = "../../proto";

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(
            &[
                format!("{proto_root}/persephone/common/common.proto"),
                format!("{proto_root}/persephone/embedding/embedding.proto"),
                format!("{proto_root}/persephone/extraction/extraction.proto"),
                format!("{proto_root}/persephone/training/training.proto"),
            ],
            &[proto_root],
        )?;

    // Re-run if any proto file changes
    println!("cargo:rerun-if-changed={proto_root}/");

    Ok(())
}
