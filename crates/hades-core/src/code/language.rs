//! Language detection from file extensions.

use std::path::Path;

use serde::Serialize;

/// Supported source languages for AST analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[non_exhaustive]
pub enum Language {
    Python,
    Rust,
    /// C, C++, and CUDA (analyzed via libclang).
    Cpp,
    /// Go (gopls is authoritative when available; Tree-sitter is the fallback).
    Go,
}

impl Language {
    /// Detect language from a file path's extension.
    ///
    /// Returns `None` for unsupported or extensionless files.
    pub fn from_path(path: &str) -> Option<Self> {
        let ext = Path::new(path).extension()?.to_str()?;
        Self::from_extension(ext)
    }

    /// Detect language from a bare file extension (without the dot).
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            "py" | "pyi" | "pyw" => Some(Self::Python),
            "rs" => Some(Self::Rust),
            "c" | "cc" | "cpp" | "cxx" | "c++" | "h" | "hh" | "hpp" | "hxx" | "cu" | "cuh" => {
                Some(Self::Cpp)
            }
            "go" => Some(Self::Go),
            _ => None,
        }
    }

    /// Detect language from a shebang line (`#!/usr/bin/env python3`).
    ///
    /// Extensionless scripts are identified only by their first line, so an
    /// extension-keyed lookup cannot see them at all. Returns `None` both for
    /// non-shebang lines and for interpreters HADES has no analyzer for (shell,
    /// perl, …) — the caller still wants those files ingested, just through the
    /// parser-free raw-text path.
    pub fn from_shebang(first_line: &str) -> Option<Self> {
        let rest = first_line.strip_prefix("#!")?;
        // `#!/usr/bin/env python3` -> the interpreter is the argument to `env`.
        let mut words = rest.split_whitespace();
        let first = words.next()?;
        let interpreter = if Path::new(first).file_name()?.to_str()? == "env" {
            words.next()?
        } else {
            first
        };
        // Strip the path and any version suffix: /usr/bin/python3.11 -> python
        let stem = Path::new(interpreter).file_name()?.to_str()?;
        let base = stem.trim_end_matches(|c: char| c.is_ascii_digit() || c == '.');
        match base {
            "python" => Some(Self::Python),
            _ => None,
        }
    }

    /// Whether a first line is a shebang at all, regardless of interpreter.
    ///
    /// Discovery uses this to decide an extensionless file is *source* even when
    /// no analyzer matches, so it lands in the raw-text path instead of being
    /// dropped without a trace.
    pub fn is_shebang(first_line: &str) -> bool {
        first_line.starts_with("#!")
    }

    /// Human-readable name for this language.
    pub fn name(self) -> &'static str {
        match self {
            Self::Python => "Python",
            Self::Rust => "Rust",
            Self::Cpp => "C++",
            Self::Go => "Go",
        }
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shebang_maps_python_interpreters() {
        assert_eq!(
            Language::from_shebang("#!/usr/bin/env python3"),
            Some(Language::Python)
        );
        assert_eq!(
            Language::from_shebang("#!/usr/bin/python"),
            Some(Language::Python)
        );
        assert_eq!(
            Language::from_shebang("#!/usr/bin/python3.11"),
            Some(Language::Python)
        );
    }

    #[test]
    fn shebang_without_analyzer_is_none_but_still_a_shebang() {
        // Shell has no analyzer: the file must still be recognized as source so
        // it reaches the raw-text path rather than being dropped silently.
        assert_eq!(Language::from_shebang("#!/bin/bash"), None);
        assert!(Language::is_shebang("#!/bin/bash"));
        assert!(Language::is_shebang("#!/usr/bin/env perl"));
    }

    #[test]
    fn non_shebang_first_lines_are_not_shebangs() {
        assert!(!Language::is_shebang("import os"));
        assert!(!Language::is_shebang("# regular comment"));
        assert_eq!(Language::from_shebang("import os"), None);
    }
}

#[cfg(test)]
mod path_tests {
    use super::*;

    #[test]
    fn test_from_extension() {
        assert_eq!(Language::from_extension("py"), Some(Language::Python));
        assert_eq!(Language::from_extension("pyi"), Some(Language::Python));
        assert_eq!(Language::from_extension("pyw"), Some(Language::Python));
        assert_eq!(Language::from_extension("rs"), Some(Language::Rust));
        assert_eq!(Language::from_extension("go"), Some(Language::Go));
        assert_eq!(Language::from_extension("js"), None);
        assert_eq!(Language::from_extension(""), None);
    }

    #[test]
    fn test_from_path() {
        assert_eq!(Language::from_path("src/main.rs"), Some(Language::Rust));
        assert_eq!(
            Language::from_path("core/models.py"),
            Some(Language::Python)
        );
        assert_eq!(Language::from_path("README.md"), None);
        assert_eq!(Language::from_path("Makefile"), None);
        assert_eq!(
            Language::from_path("src/__init__.pyi"),
            Some(Language::Python)
        );
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Language::Python), "Python");
        assert_eq!(format!("{}", Language::Rust), "Rust");
    }
}
