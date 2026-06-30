//! Feasibility probe for #149: does the `clang` crate load libclang and extract
//! function symbols (with spans, for the #148 identity contract) from a real
//! CUDA kernel file?
//!
//! Skips (does not fail) if libclang or the fixture is unavailable.

use std::path::Path;

/// Recursively collect (name, line) for FunctionDecls written in the main file
/// (not the pulled-in CUDA headers). `extern "C"` wraps its FunctionDecls in a
/// LinkageSpec, so we must recurse rather than only look at top-level children.
fn collect_functions(entity: clang::Entity, out: &mut Vec<(String, u32)>) {
    for child in entity.get_children() {
        let in_main = child
            .get_location()
            .map(|l| l.is_in_main_file())
            .unwrap_or(false);
        if in_main && child.get_kind() == clang::EntityKind::FunctionDecl {
            if let Some(name) = child.get_name() {
                let line = child
                    .get_location()
                    .map(|l| l.get_file_location().line)
                    .unwrap_or(0);
                out.push((name, line));
            }
        }
        collect_functions(child, out);
    }
}

#[test]
fn clang_extracts_cuda_kernel_symbols() {
    let fixture = "/home/todd/olympus/NL_Hecate/core/kernels/elementwise.cu";
    if !Path::new(fixture).exists() {
        eprintln!("SKIP: fixture not present: {fixture}");
        return;
    }

    let clang = match clang::Clang::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: libclang unavailable: {e}");
            return;
        }
    };
    let index = clang::Index::new(&clang, false, false);

    let tu = match index
        .parser(fixture)
        .arguments(&[
            "-x",
            "cuda",
            "--cuda-host-only",
            "--cuda-path=/usr/local/cuda-12.6",
            "--no-cuda-version-check",
            "-std=c++17",
        ])
        .skip_function_bodies(true)
        .parse()
    {
        Ok(tu) => tu,
        Err(e) => {
            eprintln!("SKIP: parse failed: {e:?}");
            return;
        }
    };

    let mut funcs = Vec::new();
    collect_functions(tu.get_entity(), &mut funcs);

    let names: Vec<&str> = funcs.iter().map(|(n, _)| n.as_str()).collect();
    eprintln!("extracted {} functions from elementwise.cu:", funcs.len());
    for (n, line) in &funcs {
        eprintln!("  {line:>4}  {n}");
    }

    assert!(!funcs.is_empty(), "no functions extracted from the CUDA file");
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
        funcs.iter().all(|(_, line)| *line > 0),
        "every symbol must have a 1-based line"
    );
}
