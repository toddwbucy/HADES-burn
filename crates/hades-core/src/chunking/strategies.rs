//! Chunking strategy implementations.

use super::{ChunkingStrategy, TextChunk};

// ---------------------------------------------------------------------------
// Token-based chunking
// ---------------------------------------------------------------------------

/// Splits text into chunks of approximately `chunk_size` whitespace-delimited
/// tokens with `overlap` tokens of overlap between consecutive chunks.
///
/// Uses a simple whitespace tokenizer.  For ML-accurate token counts, the
/// Persephone service handles BPE tokenization; this is a fast, deterministic
/// approximation suitable for chunk boundary calculation.
pub struct TokenChunking {
    /// Target tokens per chunk.
    pub chunk_size: usize,
    /// Overlap tokens between consecutive chunks.
    pub overlap: usize,
}

impl Default for TokenChunking {
    fn default() -> Self {
        Self {
            chunk_size: 500,
            overlap: 200,
        }
    }
}

impl ChunkingStrategy for TokenChunking {
    fn chunk(&self, text: &str) -> Vec<TextChunk> {
        if text.is_empty() {
            return Vec::new();
        }

        // Collect (byte_start, byte_end) for each whitespace-delimited token.
        let tokens: Vec<(usize, usize)> = token_spans(text);
        if tokens.is_empty() {
            return Vec::new();
        }

        let step = self.chunk_size.saturating_sub(self.overlap).max(1);
        let mut chunks = Vec::new();
        let mut start = 0;

        while start < tokens.len() {
            let end = (start + self.chunk_size).min(tokens.len());
            let char_start = tokens[start].0;
            let char_end = tokens[end - 1].1;

            chunks.push(TextChunk {
                text: text[char_start..char_end].to_string(),
                start_char: char_start,
                end_char: char_end,
                chunk_index: chunks.len(),
                total_chunks: 0, // filled below
            });

            if end >= tokens.len() {
                break;
            }
            start += step;
        }

        let total = chunks.len();
        for c in &mut chunks {
            c.total_chunks = total;
        }
        chunks
    }
}

// ---------------------------------------------------------------------------
// Sliding window chunking (character-based)
// ---------------------------------------------------------------------------

/// Character-based sliding window with configurable window size and step.
pub struct SlidingWindowChunking {
    /// Window size in characters.
    pub window_size: usize,
    /// Step size in characters (window_size - overlap).
    pub step_size: usize,
}

impl Default for SlidingWindowChunking {
    fn default() -> Self {
        Self {
            window_size: 512,
            step_size: 256,
        }
    }
}

impl ChunkingStrategy for SlidingWindowChunking {
    fn chunk(&self, text: &str) -> Vec<TextChunk> {
        if text.is_empty() || self.window_size == 0 {
            return Vec::new();
        }

        let step = self.step_size.max(1);
        let mut chunks = Vec::new();

        // Work on character indices for correct Unicode handling.
        let char_indices: Vec<(usize, char)> = text.char_indices().collect();
        let total_chars = char_indices.len();
        let mut char_pos = 0;

        while char_pos < total_chars {
            let end_char_pos = (char_pos + self.window_size).min(total_chars);
            let byte_start = char_indices[char_pos].0;
            let byte_end = if end_char_pos < total_chars {
                char_indices[end_char_pos].0
            } else {
                text.len()
            };

            chunks.push(TextChunk {
                text: text[byte_start..byte_end].to_string(),
                start_char: char_pos,
                end_char: end_char_pos,
                chunk_index: chunks.len(),
                total_chunks: 0,
            });

            if end_char_pos >= total_chars {
                break;
            }
            char_pos += step;
        }

        let total = chunks.len();
        for c in &mut chunks {
            c.total_chunks = total;
        }
        chunks
    }
}

// ---------------------------------------------------------------------------
// Sentence-based chunking
// ---------------------------------------------------------------------------

/// Splits text on sentence boundaries, grouping sentences into chunks
/// that don't exceed `max_chunk_size` characters.
pub struct SentenceChunking {
    /// Maximum characters per chunk.
    pub max_chunk_size: usize,
    /// Minimum characters per chunk (avoids tiny trailing chunks).
    pub min_chunk_size: usize,
}

impl Default for SentenceChunking {
    fn default() -> Self {
        Self {
            max_chunk_size: 1500,
            min_chunk_size: 100,
        }
    }
}

impl ChunkingStrategy for SentenceChunking {
    fn chunk(&self, text: &str) -> Vec<TextChunk> {
        if text.is_empty() {
            return Vec::new();
        }

        let sentences = split_sentences(text);
        if sentences.is_empty() {
            return Vec::new();
        }

        let mut chunks = Vec::new();
        let mut current_start: usize = sentences[0].0;
        let mut current_end: usize = sentences[0].0;

        for &(sent_start, sent_end) in &sentences {
            let current_len = current_end - current_start;
            let sent_len = sent_end - sent_start;

            if current_len > 0 && current_len + sent_len > self.max_chunk_size {
                // Flush current chunk.
                chunks.push(TextChunk {
                    text: text[current_start..current_end].to_string(),
                    start_char: current_start,
                    end_char: current_end,
                    chunk_index: chunks.len(),
                    total_chunks: 0,
                });
                current_start = sent_start;
            }
            current_end = sent_end;
        }

        // Flush remaining text.
        if current_end > current_start {
            // Merge tiny trailing chunk into previous if possible.
            let remaining_len = current_end - current_start;
            if remaining_len < self.min_chunk_size && !chunks.is_empty() {
                let last = chunks.last_mut().unwrap();
                last.text = text[last.start_char..current_end].to_string();
                last.end_char = current_end;
            } else {
                chunks.push(TextChunk {
                    text: text[current_start..current_end].to_string(),
                    start_char: current_start,
                    end_char: current_end,
                    chunk_index: chunks.len(),
                    total_chunks: 0,
                });
            }
        }

        let total = chunks.len();
        for c in &mut chunks {
            c.total_chunks = total;
        }
        chunks
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return byte-offset spans for whitespace-delimited tokens.
fn token_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut in_token = false;
    let mut start = 0;

    for (i, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if in_token {
                spans.push((start, i));
                in_token = false;
            }
        } else if !in_token {
            start = i;
            in_token = true;
        }
    }
    if in_token {
        spans.push((start, text.len()));
    }
    spans
}

/// Simple sentence splitter based on punctuation followed by whitespace.
///
/// Returns byte-offset spans `(start, end)` for each sentence.
fn split_sentences(text: &str) -> Vec<(usize, usize)> {
    let mut sentences = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        // Sentence-ending punctuation followed by whitespace or end-of-text.
        if matches!(bytes[i], b'.' | b'!' | b'?') {
            let after = i + 1;
            if after >= len || bytes[after].is_ascii_whitespace() {
                let end = after.min(len);
                // Include trailing whitespace in the sentence span
                // so the next sentence starts at a non-space character.
                let mut trim_end = end;
                while trim_end < len && bytes[trim_end].is_ascii_whitespace() {
                    trim_end += 1;
                }
                sentences.push((start, end));
                start = trim_end;
                i = trim_end;
                continue;
            }
        }
        i += 1;
    }

    // Trailing text without sentence-ending punctuation.
    if start < len {
        sentences.push((start, len));
    }

    sentences
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Token chunking ---------------------------------------------------

    #[test]
    fn test_token_chunking_basic() {
        let text = "one two three four five six seven eight nine ten";
        let chunker = TokenChunking {
            chunk_size: 4,
            overlap: 2,
        };
        let chunks = chunker.chunk(text);

        assert!(chunks.len() >= 2);
        // First chunk has 4 tokens
        assert_eq!(chunks[0].text, "one two three four");
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[0].start_char, 0);
        // Overlap: step = 4-2 = 2, so second chunk starts at token index 2
        assert_eq!(chunks[1].text, "three four five six");
        // All chunks know the total
        for c in &chunks {
            assert_eq!(c.total_chunks, chunks.len());
        }
    }

    #[test]
    fn test_token_chunking_empty() {
        let chunker = TokenChunking::default();
        assert!(chunker.chunk("").is_empty());
        assert!(chunker.chunk("   ").is_empty());
    }

    #[test]
    fn test_token_chunking_single_chunk() {
        let text = "hello world";
        let chunker = TokenChunking {
            chunk_size: 100,
            overlap: 10,
        };
        let chunks = chunker.chunk(text);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "hello world");
        assert_eq!(chunks[0].total_chunks, 1);
    }

    // -- Sliding window ---------------------------------------------------

    #[test]
    fn test_sliding_window_basic() {
        let text = "abcdefghij"; // 10 chars
        let chunker = SlidingWindowChunking {
            window_size: 4,
            step_size: 2,
        };
        let chunks = chunker.chunk(text);
        assert_eq!(chunks[0].text, "abcd");
        assert_eq!(chunks[1].text, "cdef");
        assert!(chunks.len() >= 3);
    }

    #[test]
    fn test_sliding_window_empty() {
        let chunker = SlidingWindowChunking::default();
        assert!(chunker.chunk("").is_empty());
    }

    // -- Sentence chunking ------------------------------------------------

    #[test]
    fn test_sentence_chunking_basic() {
        let text = "First sentence. Second sentence. Third sentence.";
        let chunker = SentenceChunking {
            max_chunk_size: 35,
            min_chunk_size: 5,
        };
        let chunks = chunker.chunk(text);
        assert!(chunks.len() >= 2);
        // First chunk should contain at least the first sentence
        assert!(chunks[0].text.contains("First"));
    }

    #[test]
    fn test_sentence_chunking_tiny_trailing_merged() {
        // Short trailing text should be merged into previous chunk.
        let text = "A long enough sentence here. OK.";
        let chunker = SentenceChunking {
            max_chunk_size: 30,
            min_chunk_size: 10,
        };
        let chunks = chunker.chunk(text);
        // "OK." is only 3 chars < min_chunk_size, merged into previous
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_sentence_chunking_empty() {
        let chunker = SentenceChunking::default();
        assert!(chunker.chunk("").is_empty());
    }

    // -- Helpers ----------------------------------------------------------

    #[test]
    fn test_token_spans() {
        let spans = token_spans("hello  world  foo");
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0], (0, 5));   // "hello"
        assert_eq!(spans[1], (7, 12));  // "world"
        assert_eq!(spans[2], (14, 17)); // "foo"
    }

    #[test]
    fn test_split_sentences() {
        let text = "Hello world. This is a test! And more? Final.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 4);
        assert_eq!(&text[sentences[0].0..sentences[0].1], "Hello world.");
        assert_eq!(&text[sentences[1].0..sentences[1].1], "This is a test!");
    }

    // -- Character offsets are correct ------------------------------------

    #[test]
    fn test_chunk_offsets_roundtrip() {
        let text = "The quick brown fox jumps over the lazy dog and more words here";
        let chunker = TokenChunking {
            chunk_size: 3,
            overlap: 1,
        };
        let chunks = chunker.chunk(text);
        for c in &chunks {
            assert_eq!(&text[c.start_char..c.end_char], c.text);
        }
    }
}
