fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_dir = "../../proto/persephone";

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(
            &[
                format!("{proto_dir}/common.proto"),
                format!("{proto_dir}/embedding.proto"),
                format!("{proto_dir}/extraction.proto"),
            ],
            &["../../proto"],
        )?;

    // Re-run if any proto file changes
    println!("cargo:rerun-if-changed=../../proto/");

    Ok(())
}
