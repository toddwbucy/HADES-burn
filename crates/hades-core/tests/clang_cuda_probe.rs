//! End-to-end probe for #149 against a real CUDA kernel file.
//!
//! Skips (does not fail) if libclang or the fixture is unavailable.

use std::path::Path;

use hades_core::code;

#[test]
fn clang_extracts_cuda_kernel_symbols() {
    let fixture = "/home/todd/olympus/NL_Hecate/core/kernels/elementwise.cu";
    if !Path::new(fixture).exists() {
        eprintln!("SKIP: fixture not present: {fixture}");
        return;
    }

    match clang::Clang::new() {
        Ok(clang) => drop(clang),
        Err(e) => {
            eprintln!("SKIP: libclang unavailable: {e}");
            return;
        }
    }

    let source = std::fs::read_to_string(fixture).unwrap();
    let analysis = code::analyze(&source, fixture).expect("CUDA analysis should succeed");
    let names: Vec<&str> = analysis.symbols.iter().map(|s| s.name.as_str()).collect();
    eprintln!("extracted {} symbols from elementwise.cu", names.len());

    assert!(
        !analysis.symbols.is_empty(),
        "no functions extracted from the CUDA file"
    );
    assert!(
        names.contains(&"sigmoid_kernel"),
        "missing __global__ kernel `sigmoid_kernel`: {names:?}"
    );
    assert!(
        names.contains(&"sigmoid_cuda"),
        "missing extern \"C\" host wrapper `sigmoid_cuda`: {names:?}"
    );
    // Every symbol carries a line span -- the anchor the #148 identity needs.
    assert!(
        analysis.symbols.iter().all(|symbol| symbol.start_line > 0),
        "every symbol must have a 1-based line"
    );

    let wrapper = analysis
        .symbols
        .iter()
        .find(|symbol| symbol.name == "sigmoid_cuda")
        .expect("missing sigmoid_cuda wrapper");
    let calls = wrapper.metadata["calls"]
        .as_array()
        .expect("wrapper should carry resolved calls");
    assert!(
        calls
            .iter()
            .any(|call| { call["name"] == "sigmoid_kernel" && call["is_kernel_launch"] == true }),
        "missing resolved CUDA launch edge: {calls:?}"
    );
}
