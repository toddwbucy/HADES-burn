//! AST-aligned code chunking.
//!
//! Chunks source code at function/class/struct boundaries rather than
//! arbitrary token counts.  Module-level code (imports, constants,
//! globals) becomes chunk 0, then each top-level definition becomes
//! its own chunk.
//!
//! Oversized chunks (>8000 chars by default) are split at line
//! boundaries to stay within embedding model context windows.

use crate::chunking::{ChunkingStrategy, TextChunk};

use super::symbols::TopLevelDef;

/// Maximum characters per chunk before line-boundary splitting.
const DEFAULT_MAX_CHUNK_CHARS: usize = 8000;

/// AST-aware chunking strategy for source code.
///
/// Call [`AstChunking::new`] with a language and pre-extracted top-level
/// definitions (from [`crate::code::analyze`]).  The chunker uses those
/// boundaries to produce semantically meaningful chunks.
pub struct AstChunking {
    max_chunk_chars: usize,
    defs: Vec<TopLevelDef>,
}

impl AstChunking {
    /// Create a new AST chunker from pre-extracted definitions.
    pub fn new(defs: Vec<TopLevelDef>) -> Self {
        Self {
            max_chunk_chars: DEFAULT_MAX_CHUNK_CHARS,
            defs,
        }
    }

    /// Override the maximum chunk size (in characters).
    pub fn with_max_chars(mut self, max: usize) -> Self {
        self.max_chunk_chars = max;
        self
    }
}

impl ChunkingStrategy for AstChunking {
    fn chunk(&self, text: &str) -> Vec<TextChunk> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut raw_chunks = Vec::new();

        if self.defs.is_empty() {
            // No definitions found — treat entire file as one chunk.
            raw_chunks.push((text.to_string(), 0, text.len()));
        } else {
            // Walk the file once, emitting gaps and defs in source order.
            let mut cursor = 0usize;

            for def in &self.defs {
                // Emit gap (module-level code) before this def.
                let def_start = def.start_byte.min(text.len());
                if def_start > cursor {
                    let gap = &text[cursor..def_start];
                    if !gap.trim().is_empty() {
                        raw_chunks.push((gap.to_string(), cursor, def_start));
                    }
                }

                // Emit the def itself.
                let start = def_start;
                let end = def.end_byte.min(text.len());
                if start < end {
                    raw_chunks.push((text[start..end].to_string(), start, end));
                }

                cursor = def.end_byte.min(text.len());
            }

            // Trailing module code after last def.
            if cursor < text.len() {
                let tail = &text[cursor..];
                if !tail.trim().is_empty() {
                    raw_chunks.push((tail.to_string(), cursor, text.len()));
                }
            }
        }

        // Split oversized chunks at line boundaries.
        let mut final_chunks = Vec::new();
        for (chunk_text, start_byte, _end_byte) in raw_chunks {
            if chunk_text.len() <= self.max_chunk_chars {
                final_chunks.push((chunk_text, start_byte));
            } else {
                split_at_lines(&chunk_text, start_byte, self.max_chunk_chars, &mut final_chunks);
            }
        }

        // Build TextChunk values with correct indices.
        let total = final_chunks.len();
        final_chunks
            .into_iter()
            .enumerate()
            .map(|(i, (text_content, start))| TextChunk {
                text: text_content.clone(),
                start_char: start,
                end_char: start + text_content.len(),
                chunk_index: i,
                total_chunks: total,
            })
            .collect()
    }
}

/// Split a chunk at line boundaries, keeping each sub-chunk under `max_chars`.
fn split_at_lines(
    text: &str,
    base_offset: usize,
    max_chars: usize,
    out: &mut Vec<(String, usize)>,
) {
    let mut current = String::new();
    let mut current_start = base_offset;
    let mut byte_offset = 0usize;

    for line in text.lines() {
        let line_with_newline_len = line.len() + 1; // approximate

        if !current.is_empty() && current.len() + line_with_newline_len > max_chars {
            out.push((current.clone(), current_start));
            current.clear();
            current_start = base_offset + byte_offset;
        }

        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
        byte_offset += line.len() + 1; // +1 for newline
    }

    if !current.is_empty() {
        out.push((current, current_start));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code::symbols::SymbolKind;

    #[test]
    fn test_empty_source() {
        let chunker = AstChunking::new(vec![]);
        let chunks = chunker.chunk("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_no_defs_single_chunk() {
        let source = "import os\nimport sys\n\nprint('hello')\n";
        let chunker = AstChunking::new(vec![]);
        let chunks = chunker.chunk(source);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, source);
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[0].total_chunks, 1);
    }

    #[test]
    fn test_module_plus_function() {
        let source = "import os\n\ndef hello():\n    print('hi')\n";
        let def_start = source.find("def").unwrap();
        let defs = vec![TopLevelDef {
            name: "hello".into(),
            kind: SymbolKind::Function,
            start_line: 3,
            end_line: 4,
            start_byte: def_start,
            end_byte: source.len(),
        }];
        let chunker = AstChunking::new(defs);
        let chunks = chunker.chunk(source);

        assert_eq!(chunks.len(), 2, "expected module chunk + function chunk");
        // Chunk 0: module-level imports.
        assert!(chunks[0].text.contains("import os"));
        // Chunk 1: function.
        assert!(chunks[1].text.contains("def hello"));
    }

    #[test]
    fn test_multiple_defs() {
        let source = "X = 1\n\ndef foo():\n    pass\n\ndef bar():\n    pass\n";
        let foo_start = source.find("def foo").unwrap();
        let foo_end = source.find("\ndef bar").unwrap() + 1;
        let bar_start = source.find("def bar").unwrap();

        let defs = vec![
            TopLevelDef {
                name: "foo".into(),
                kind: SymbolKind::Function,
                start_line: 3,
                end_line: 4,
                start_byte: foo_start,
                end_byte: foo_end,
            },
            TopLevelDef {
                name: "bar".into(),
                kind: SymbolKind::Function,
                start_line: 6,
                end_line: 7,
                start_byte: bar_start,
                end_byte: source.len(),
            },
        ];
        let chunker = AstChunking::new(defs);
        let chunks = chunker.chunk(source);

        assert_eq!(chunks.len(), 3, "module + foo + bar");
        assert!(chunks[0].text.contains("X = 1"));
        assert!(chunks[1].text.contains("def foo"));
        assert!(chunks[2].text.contains("def bar"));
    }

    #[test]
    fn test_oversized_chunk_splits() {
        // Create a source with a very long function.
        let mut source = String::from("def big():\n");
        for i in 0..200 {
            source.push_str(&format!("    x_{i} = {i}\n"));
        }
        let defs = vec![TopLevelDef {
            name: "big".into(),
            kind: SymbolKind::Function,
            start_line: 1,
            end_line: 201,
            start_byte: 0,
            end_byte: source.len(),
        }];
        let chunker = AstChunking::new(defs).with_max_chars(500);
        let chunks = chunker.chunk(&source);

        assert!(chunks.len() > 1, "should have split into multiple chunks");
        for chunk in &chunks {
            assert!(
                chunk.text.len() <= 600, // some slack for line boundaries
                "chunk too large: {} chars",
                chunk.text.len()
            );
        }
    }

    #[test]
    fn test_chunk_indices_correct() {
        let source = "A = 1\n\ndef f():\n    pass\n\ndef g():\n    pass\n";
        let f_start = source.find("def f").unwrap();
        let f_end = source.find("\ndef g").unwrap() + 1;
        let g_start = source.find("def g").unwrap();

        let defs = vec![
            TopLevelDef { name: "f".into(), kind: SymbolKind::Function, start_line: 3, end_line: 4, start_byte: f_start, end_byte: f_end },
            TopLevelDef { name: "g".into(), kind: SymbolKind::Function, start_line: 6, end_line: 7, start_byte: g_start, end_byte: source.len() },
        ];
        let chunker = AstChunking::new(defs);
        let chunks = chunker.chunk(source);

        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_index, i);
            assert_eq!(chunk.total_chunks, chunks.len());
        }
    }
}
