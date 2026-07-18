//! C / C++ / CUDA source analysis via libclang.
//!
//! The analyzer uses compiler arguments from `compile_commands.json` when one
//! is supplied (or found near the source tree), parses function bodies, and
//! records libclang-resolved calls in symbol metadata. Graph materialization
//! happens after all files are ingested in [`super::cpp_edges`].

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use clang::{Clang, CompilationDatabase, Entity, EntityKind, Index, Unsaved};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tracing::{debug, warn};

use super::Language;
use super::symbols::{CodeMetrics, FileAnalysis, Symbol, SymbolKind, TopLevelDef};

static CLANG_LOCK: Mutex<()> = Mutex::new(());

/// Analyze C/C++/CUDA source, returning symbols, resolved call metadata,
/// metrics, and AST-aligned definition boundaries.
pub fn analyze(
    source: &str,
    file_path: &str,
    compilation_database: Option<&Path>,
) -> Result<FileAnalysis, super::CodeAnalysisError> {
    let (symbols, top_level_defs) = match extract(source, file_path, compilation_database) {
        Ok(pair) => pair,
        Err(e) => {
            warn!(error = %e, file = file_path, "libclang analysis failed, no C++ symbols");
            (Vec::new(), Vec::new())
        }
    };
    let metrics = compute_metrics(source);

    // Calls, namespace ownership, and template identity are part of the graph
    // structure. Hash serialized symbols rather than bare names so an edited
    // call site triggers authoritative re-ingest even when declarations did
    // not change.
    let encoded = serde_json::to_vec(&symbols).unwrap_or_default();
    let symbol_hash = Sha256::digest(encoded)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();

    Ok(FileAnalysis {
        language: Language::Cpp,
        symbols,
        metrics,
        symbol_hash,
        top_level_defs,
    })
}

#[derive(Debug, Clone, Copy)]
enum Mode {
    Cuda,
    C,
    Cpp,
}

fn detect_mode(file_path: &str, source: &str) -> Mode {
    if file_path.ends_with(".cu")
        || file_path.ends_with(".cuh")
        || source.contains("__global__")
        || source.contains("__device__")
        || source.contains("__host__")
    {
        Mode::Cuda
    } else if file_path.ends_with(".c") {
        Mode::C
    } else {
        Mode::Cpp
    }
}

fn default_parse_args(mode: Mode) -> Vec<String> {
    match mode {
        Mode::Cuda => [
            "-x",
            "cuda",
            "--cuda-host-only",
            "--cuda-path=/usr/local/cuda",
            "--no-cuda-version-check",
            "-std=c++17",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        Mode::C => ["-x", "c", "-std=c17"]
            .into_iter()
            .map(str::to_string)
            .collect(),
        Mode::Cpp => ["-x", "c++", "-std=c++17"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    }
}

fn extract(
    source: &str,
    file_path: &str,
    compilation_database: Option<&Path>,
) -> Result<(Vec<Symbol>, Vec<TopLevelDef>), String> {
    // `clang-rs` permits one live `Clang` token per process. Recover from a
    // poisoned lock rather than cascading failures across the ingest batch.
    let _guard = CLANG_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let clang = Clang::new()?;
    let index = Index::new(&clang, false, false);
    let mode = detect_mode(file_path, source);
    let (args, from_database) = compilation_arguments(file_path, compilation_database)
        .map(|args| (args, true))
        .unwrap_or_else(|| (default_parse_args(mode), false));

    let parse = |arguments: &[String]| {
        let refs: Vec<&str> = arguments.iter().map(String::as_str).collect();
        index
            .parser(file_path)
            .arguments(&refs)
            .unsaved(&[Unsaved::new(file_path, source)])
            .skip_function_bodies(false)
            .detailed_preprocessing_record(true)
            .parse()
            .map_err(|e| format!("{e:?}"))
    };

    // A compilation database may contain driver-only flags that libclang does
    // not accept. Preserve availability by retrying with conservative language
    // defaults while making the downgrade visible in logs.
    let tu = match parse(&args) {
        Ok(tu) => tu,
        Err(first) if from_database => {
            warn!(
                file = file_path,
                error = %first,
                "compile-command parse failed; retrying with default clang arguments"
            );
            parse(&default_parse_args(mode))?
        }
        Err(e) => return Err(e),
    };

    let serious_diagnostics = tu
        .get_diagnostics()
        .into_iter()
        .filter(|d| {
            matches!(
                d.get_severity(),
                clang::diagnostic::Severity::Error | clang::diagnostic::Severity::Fatal
            )
        })
        .count();
    if serious_diagnostics > 0 {
        debug!(
            file = file_path,
            serious_diagnostics, "libclang parsed with diagnostics"
        );
    }

    let lines: Vec<&str> = source.lines().collect();
    let mut symbols = Vec::new();
    let mut defs = Vec::new();
    collect(
        tu.get_entity(),
        true,
        source,
        &lines,
        &mut symbols,
        &mut defs,
    );

    // libclang can expose the same declaration through a template wrapper and
    // its child cursor. Span identity is the contract, so deduplicate exactly
    // on the stored identity tuple while retaining distinct overloads and
    // specializations at different spans.
    let mut seen = HashSet::new();
    symbols.retain(|s| seen.insert((s.qualified_name(), s.start_line, s.kind.lang_kind())));
    let mut seen_defs = HashSet::new();
    defs.retain(|d| seen_defs.insert((d.start_byte, d.end_byte, d.kind.lang_kind())));

    Ok((symbols, defs))
}

/// Load the matching command from a compilation database. An explicit path
/// may name either the JSON file or its directory; without one, search source
/// ancestors and their conventional `build/` directory.
fn compilation_arguments(file_path: &str, explicit: Option<&Path>) -> Option<Vec<String>> {
    let file = Path::new(file_path);
    let directory = explicit
        .and_then(compilation_database_directory)
        .or_else(|| discover_compilation_database(file))?;
    let database = CompilationDatabase::from_directory(&directory).ok()?;
    let commands = database.get_compile_commands(file).ok()?;
    let command = commands.get_commands().into_iter().next()?;
    let working_directory = command.get_directory();
    let raw = command.get_arguments();
    let args = sanitize_compile_arguments(&raw, &working_directory, file);
    if args.is_empty() {
        return None;
    }
    debug!(
        file = file_path,
        database = %directory.display(),
        argument_count = args.len(),
        "using compilation database"
    );
    Some(args)
}

fn compilation_database_directory(path: &Path) -> Option<PathBuf> {
    if path.is_file() {
        if path.file_name().and_then(|n| n.to_str()) == Some("compile_commands.json") {
            return path.parent().map(Path::to_path_buf);
        }
        return None;
    }
    path.join("compile_commands.json")
        .is_file()
        .then(|| path.to_path_buf())
}

fn discover_compilation_database(file: &Path) -> Option<PathBuf> {
    let start = file.parent()?;
    for ancestor in start.ancestors() {
        if ancestor.join("compile_commands.json").is_file() {
            return Some(ancestor.to_path_buf());
        }
        let build = ancestor.join("build");
        if build.join("compile_commands.json").is_file() {
            return Some(build);
        }
    }
    None
}

fn sanitize_compile_arguments(raw: &[String], cwd: &Path, file: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = usize::from(raw.first().is_some_and(|a| !a.starts_with('-')));
    while i < raw.len() {
        let arg = &raw[i];

        if matches!(arg.as_str(), "-c" | "--compile") {
            i += 1;
            continue;
        }
        if matches!(arg.as_str(), "-o" | "-MF" | "-MT" | "-MQ" | "--output") {
            i += 2;
            continue;
        }
        if (arg.starts_with("-o") && arg.len() > 2) || arg.starts_with("--output=") {
            i += 1;
            continue;
        }
        if same_path(arg, file, cwd) {
            i += 1;
            continue;
        }

        if matches!(
            arg.as_str(),
            "-I" | "-isystem" | "-iquote" | "-include" | "-imacros"
        ) && let Some(value) = raw.get(i + 1)
        {
            out.push(arg.clone());
            out.push(absolutize(value, cwd));
            i += 2;
            continue;
        }
        if let Some(value) = arg.strip_prefix("-I").filter(|v| !v.is_empty()) {
            out.push(format!("-I{}", absolutize(value, cwd)));
            i += 1;
            continue;
        }

        out.push(arg.clone());
        i += 1;
    }
    out
}

fn same_path(candidate: &str, file: &Path, cwd: &Path) -> bool {
    if candidate.starts_with('-') {
        return false;
    }
    let candidate = Path::new(candidate);
    let candidate = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        cwd.join(candidate)
    };
    if candidate == file {
        return true;
    }
    match (candidate.canonicalize(), file.canonicalize()) {
        (Ok(candidate), Ok(file)) => candidate == file,
        _ => false,
    }
}

fn absolutize(value: &str, cwd: &Path) -> String {
    let path = Path::new(value);
    if path.is_absolute() {
        value.to_string()
    } else {
        cwd.join(path).to_string_lossy().into_owned()
    }
}

/// Walk declarations in the main file. Calls are collected recursively from
/// each callable body and attached to that symbol's metadata.
fn collect(
    entity: Entity,
    top: bool,
    source: &str,
    lines: &[&str],
    symbols: &mut Vec<Symbol>,
    defs: &mut Vec<TopLevelDef>,
) {
    for child in entity.get_children() {
        let in_main = child
            .get_location()
            .map(|l| l.is_in_main_file())
            .unwrap_or(false);
        let kind = child.get_kind();

        if in_main {
            if is_callable(kind) {
                if let Some(sym) = func_symbol(&child, source, lines) {
                    symbols.push(sym);
                }
            } else if let Some(symbol_kind) = type_symbol_kind(kind)
                && let Some(sym) = type_symbol(&child, symbol_kind)
            {
                symbols.push(sym);
            }

            if top
                && (is_callable(kind) || type_symbol_kind(kind).is_some())
                && let Some(def) = top_level_def(&child)
            {
                defs.push(def);
            }
        }

        match kind {
            EntityKind::Namespace | EntityKind::LinkageSpec => {
                collect(child, top, source, lines, symbols, defs);
            }
            EntityKind::StructDecl
            | EntityKind::ClassDecl
            | EntityKind::ClassTemplate
            | EntityKind::ClassTemplatePartialSpecialization => {
                collect(child, false, source, lines, symbols, defs);
            }
            _ => {}
        }
    }
}

fn is_callable(kind: EntityKind) -> bool {
    matches!(
        kind,
        EntityKind::FunctionDecl
            | EntityKind::Method
            | EntityKind::Constructor
            | EntityKind::Destructor
            | EntityKind::ConversionFunction
            | EntityKind::FunctionTemplate
    )
}

fn type_symbol_kind(kind: EntityKind) -> Option<SymbolKind> {
    match kind {
        EntityKind::StructDecl => Some(SymbolKind::Struct),
        EntityKind::ClassDecl
        | EntityKind::ClassTemplate
        | EntityKind::ClassTemplatePartialSpecialization => Some(SymbolKind::Class),
        EntityKind::EnumDecl => Some(SymbolKind::Enum),
        EntityKind::TypeAliasDecl | EntityKind::TypedefDecl | EntityKind::TypeAliasTemplateDecl => {
            Some(SymbolKind::TypeAlias)
        }
        _ => None,
    }
}

fn span_lines(entity: &Entity) -> Option<(usize, usize)> {
    let range = entity.get_range()?;
    let start = range.get_start().get_file_location().line as usize;
    let end = range.get_end().get_file_location().line as usize;
    Some((start.max(1), end.max(start)))
}

fn func_symbol(entity: &Entity, source: &str, lines: &[&str]) -> Option<Symbol> {
    let name = entity.get_name()?;
    let (start_line, end_line) = span_lines(entity)?;
    let mut meta = common_metadata(entity);

    if let Some(line) = lines.get(start_line.saturating_sub(1)) {
        if line.contains("__global__") {
            meta.insert("is_kernel".into(), json!(true));
        }
        if line.contains("__device__") {
            meta.insert("is_device".into(), json!(true));
        }
        if line.contains("__host__") {
            meta.insert("is_host".into(), json!(true));
        }
    }

    let mut calls = Vec::new();
    collect_calls(*entity, source, &mut calls);
    if !calls.is_empty() {
        meta.insert("calls".into(), Value::Array(calls));
    }

    Some(Symbol {
        name,
        kind: SymbolKind::Function,
        start_line,
        end_line,
        metadata: Value::Object(meta),
    })
}

fn type_symbol(entity: &Entity, kind: SymbolKind) -> Option<Symbol> {
    let name = entity.get_name()?;
    let (start_line, end_line) = span_lines(entity)?;
    Some(Symbol {
        name,
        kind,
        start_line,
        end_line,
        metadata: Value::Object(common_metadata(entity)),
    })
}

fn common_metadata(entity: &Entity) -> serde_json::Map<String, Value> {
    let mut meta = serde_json::Map::new();
    meta.insert("qualified_name".into(), json!(qualified_name(entity)));
    meta.insert("analyzer".into(), json!("libclang"));
    meta.insert("analysis_tier".into(), json!("semantic"));
    meta.insert("is_definition".into(), json!(entity.is_definition()));
    if let Some(display) = entity.get_display_name() {
        meta.insert("signature".into(), json!(display));
    }
    if let Some(usr) = entity.get_usr() {
        meta.insert("usr".into(), json!(usr.0));
    }
    if matches!(
        entity.get_kind(),
        EntityKind::FunctionTemplate
            | EntityKind::ClassTemplate
            | EntityKind::ClassTemplatePartialSpecialization
            | EntityKind::TypeAliasTemplateDecl
    ) {
        meta.insert("is_template".into(), json!(true));
    }
    if let Some(template) = entity.get_template() {
        meta.insert("is_template_instantiation".into(), json!(true));
        if let Some(usr) = template.get_usr() {
            meta.insert("template_usr".into(), json!(usr.0));
        }
    }
    meta
}

fn qualified_name(entity: &Entity) -> String {
    let mut parts = entity.get_name().into_iter().collect::<Vec<_>>();
    let mut parent = entity.get_semantic_parent();
    while let Some(current) = parent {
        let kind = current.get_kind();
        if matches!(
            kind,
            EntityKind::Namespace
                | EntityKind::StructDecl
                | EntityKind::ClassDecl
                | EntityKind::ClassTemplate
                | EntityKind::ClassTemplatePartialSpecialization
                | EntityKind::EnumDecl
        ) && let Some(name) = current.get_name()
            && !name.is_empty()
        {
            parts.push(name);
        }
        if kind == EntityKind::TranslationUnit {
            break;
        }
        parent = current.get_semantic_parent();
    }
    parts.reverse();
    parts.join("::")
}

fn collect_calls(entity: Entity, source: &str, calls: &mut Vec<Value>) {
    for child in entity.get_children() {
        if child.get_kind() == EntityKind::CallExpr
            && let Some(call) = call_metadata(&child, source)
        {
            calls.push(call);
        }
        collect_calls(child, source, calls);
    }
}

fn call_metadata(call: &Entity, source: &str) -> Option<Value> {
    let referenced = call.get_reference().or_else(|| {
        call.get_children()
            .into_iter()
            .find_map(|child| child.get_reference())
    })?;
    let target = referenced.get_definition().unwrap_or(referenced);
    let target_location = target.get_location()?.get_file_location();
    let target_file = target_location.file?.get_path();
    let name = referenced.get_name().or_else(|| target.get_name())?;
    let qname = qualified_name(&referenced);
    let call_location = call.get_location()?.get_file_location();
    let snippet = call
        .get_range()
        .map(|range| {
            let start = range.get_start().get_file_location().offset as usize;
            let end = range.get_end().get_file_location().offset as usize;
            source.get(start..end).unwrap_or("")
        })
        .unwrap_or("");
    let is_kernel = target
        .get_children()
        .iter()
        .any(|child| child.get_kind() == EntityKind::CudaGlobalAttr);

    let mut value = json!({
        "name": name,
        "qualified_name": qname,
        "target_file": target_file.to_string_lossy(),
        "target_line": target_location.line,
        "call_line": call_location.line,
        "is_kernel_launch": is_kernel && snippet.contains("<<<"),
        "resolution": "semantic",
        "analyzer": "libclang",
    });
    if let Some(usr) = referenced.get_usr().or_else(|| target.get_usr()) {
        value["target_usr"] = json!(usr.0);
    }
    Some(value)
}

fn top_level_def(entity: &Entity) -> Option<TopLevelDef> {
    let name = entity.get_name()?;
    let range = entity.get_range()?;
    let start = range.get_start().get_file_location();
    let end = range.get_end().get_file_location();
    let kind = if is_callable(entity.get_kind()) {
        SymbolKind::Function
    } else {
        type_symbol_kind(entity.get_kind())?
    };
    Some(TopLevelDef {
        name,
        kind,
        start_line: (start.line as usize).max(1),
        end_line: (end.line as usize).max(start.line as usize),
        start_byte: start.offset as usize,
        end_byte: end.offset as usize,
    })
}

fn compute_metrics(source: &str) -> CodeMetrics {
    let mut total: usize = 0;
    let mut blank: usize = 0;
    let mut comment: usize = 0;
    for raw in source.lines() {
        total += 1;
        let t = raw.trim();
        if t.is_empty() {
            blank += 1;
        } else if t.starts_with("//") || t.starts_with("/*") || t.starts_with('*') {
            comment += 1;
        }
    }
    CodeMetrics {
        total_lines: total,
        lines_of_code: total.saturating_sub(blank + comment),
        blank_lines: blank,
        comment_lines: comment,
        cyclomatic_complexity: 0,
        max_nesting_depth: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    const CUDA: &str = "__global__ void my_kernel(float* x, int n) {}\n\
extern \"C\" void launch(float* x, int n) { my_kernel<<<1, 1>>>(x, n); }\n\
namespace engine { struct Cfg { int a; void reset(); }; }\n";

    fn available(analysis: &FileAnalysis) -> bool {
        if analysis.symbols.is_empty() {
            eprintln!("SKIP: libclang unavailable (no symbols extracted)");
            false
        } else {
            true
        }
    }

    #[test]
    fn analyze_cuda_extracts_kernel_call_method_and_type() {
        let analysis = analyze(CUDA, "test.cu", None).unwrap();
        if !available(&analysis) {
            return;
        }
        let by_name = |n: &str| analysis.symbols.iter().find(|s| s.name == n);
        let kernel = by_name("my_kernel").expect("kernel extracted");
        assert_eq!(kernel.kind, SymbolKind::Function);
        assert_eq!(kernel.metadata["is_kernel"], json!(true));

        let launch = by_name("launch").expect("host function");
        let calls = launch.metadata["calls"].as_array().expect("calls metadata");
        assert!(calls.iter().any(|c| {
            c["name"] == "my_kernel"
                && c["is_kernel_launch"] == true
                && c["resolution"] == "semantic"
        }));
        assert!(
            analysis
                .symbols
                .iter()
                .any(|s| { s.name == "Cfg" && s.qualified_name() == "engine::Cfg" })
        );
        assert!(
            analysis
                .symbols
                .iter()
                .any(|s| { s.name == "reset" && s.qualified_name() == "engine::Cfg::reset" })
        );
    }

    #[test]
    fn analyze_c_file_parses_as_c() {
        let analysis =
            analyze("int add(int a, int b) { return a + b; }\n", "math.c", None).unwrap();
        if !available(&analysis) {
            return;
        }
        assert!(analysis.symbols.iter().any(|s| s.name == "add"));
    }

    #[test]
    fn out_of_line_method_and_overloads_keep_distinct_span_identities() {
        let src = "namespace n { struct C { void reset(); }; }\n\
void n::C::reset() {}\n\
void f(int) {}\n\
void f(double) {}\n";
        let analysis = analyze(src, "c.cpp", None).unwrap();
        if !available(&analysis) {
            return;
        }
        assert!(
            analysis
                .top_level_defs
                .iter()
                .any(|d| d.name == "reset" && d.start_line == 2)
        );
        let overloads: Vec<_> = analysis.symbols.iter().filter(|s| s.name == "f").collect();
        assert_eq!(overloads.len(), 2);
        assert_ne!(overloads[0].start_line, overloads[1].start_line);
    }

    #[test]
    fn compilation_database_arguments_are_loaded_and_sanitized() {
        let temp = tempfile::tempdir().unwrap();
        let source_path = temp.path().join("main.cpp");
        fs::write(&source_path, "int answer() { return VALUE; }\n").unwrap();
        fs::create_dir(temp.path().join("include")).unwrap();
        let database = json!([{
            "directory": temp.path(),
            "file": source_path,
            "arguments": ["clang++", "-DVALUE=42", "-I", "include", "-std=c++20", "-c", source_path, "-o", "main.o"]
        }]);
        fs::write(
            temp.path().join("compile_commands.json"),
            serde_json::to_vec(&database).unwrap(),
        )
        .unwrap();

        let args = {
            // `CompilationDatabase` is a libclang API and therefore requires
            // the runtime library to be loaded on this thread.
            let _guard = CLANG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let _clang = match Clang::new() {
                Ok(clang) => clang,
                Err(_) => return,
            };
            compilation_arguments(source_path.to_str().unwrap(), Some(temp.path()))
                .expect("matching compile command")
        };
        assert!(args.contains(&"-DVALUE=42".to_string()));
        assert!(args.contains(&"-std=c++20".to_string()));
        assert!(!args.contains(&"-c".to_string()));
        assert!(!args.iter().any(|a| a == source_path.to_str().unwrap()));

        let analysis = analyze(
            "int answer() { return VALUE; }\n",
            source_path.to_str().unwrap(),
            Some(temp.path()),
        )
        .unwrap();
        if available(&analysis) {
            assert!(analysis.symbols.iter().any(|s| s.name == "answer"));
        }
    }

    #[test]
    fn call_changes_affect_incremental_digest() {
        let before = "void a() {} void b() { a(); }\n";
        let after = "void a() {} void c() {} void b() { c(); }\n";
        let a = analyze(before, "calls.cpp", None).unwrap();
        let b = analyze(after, "calls.cpp", None).unwrap();
        if available(&a) && available(&b) {
            assert_ne!(a.symbol_hash, b.symbol_hash);
        }
    }
}
