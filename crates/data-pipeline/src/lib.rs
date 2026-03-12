/// OCRLLM Data Pipeline — Text Chunking
///
/// Converts a UnifiedDocument (paragraph-level blocks) into fixed-size
/// overlapping TextChunks suitable for embedding and vector storage.
///
/// Chunking invariants:
/// - max_chars: 512 characters per chunk  
/// - overlap:   50 characters carried forward to the next chunk
/// - Splits are made at the nearest whitespace boundary to avoid cutting mid-word
/// - chunk_id is deterministic: "{document_id}_p{page_num}_c{chunk_idx}"

use protocol::{TextChunk, UnifiedDocument};

const MAX_CHUNK_CHARS: usize = 512;
const OVERLAP_CHARS: usize = 50;

/// Chunk a full UnifiedDocument into overlapping TextChunks.
///
/// Pages are processed in reading order. All paragraph blocks on a page
/// are concatenated (separated by "\n\n") and then chunked together,
/// preserving page attribution in each chunk.
pub fn chunk_document(doc: &UnifiedDocument) -> Vec<TextChunk> {
    let mut all_chunks: Vec<TextChunk> = Vec::new();

    for page in &doc.pages {
        // Build the full text for this page by concatenating all paragraph blocks
        let page_text: String = page
            .blocks
            .iter()
            .filter_map(|b| b.text.as_deref())
            .filter(|t| !t.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");

        if page_text.trim().is_empty() {
            continue;
        }

        let chunks = split_into_chunks(&page_text, MAX_CHUNK_CHARS, OVERLAP_CHARS);

        for (chunk_idx, chunk_text) in chunks.into_iter().enumerate() {
            let char_count = chunk_text.chars().count();
            all_chunks.push(TextChunk {
                chunk_id: format!(
                    "{}_p{}_c{}",
                    doc.document_id, page.page_num, chunk_idx
                ),
                document_id: doc.document_id.clone(),
                page_num: page.page_num,
                text: chunk_text,
                char_count,
            });
        }
    }

    all_chunks
}

/// Split `text` into overlapping windows of at most `max_chars` characters,
/// with `overlap` characters of carry-forward between consecutive chunks.
/// Splits are aligned to whitespace boundaries where possible.
fn split_into_chunks(text: &str, max_chars: usize, overlap: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let total = chars.len();

    if total <= max_chars {
        return vec![text.to_owned()];
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut start = 0usize;

    while start < total {
        let raw_end = (start + max_chars).min(total);

        // Walk back from raw_end to the nearest whitespace to avoid mid-word splits
        let end = if raw_end < total {
            find_split_boundary(&chars, raw_end)
        } else {
            raw_end
        };

        let chunk: String = chars[start..end].iter().collect();
        chunks.push(chunk);

        if end >= total {
            break;
        }

        // Advance start, carrying `overlap` chars back for context continuity
        let next_start = end.saturating_sub(overlap);
        // Find the next word boundary after next_start so we don't start mid-word
        start = find_next_word_start(&chars, next_start).unwrap_or(end);
    }

    chunks
}

/// Walk backward from `pos` to find the nearest whitespace character.
/// Returns `pos` unchanged if no whitespace found within 100 chars.
fn find_split_boundary(chars: &[char], pos: usize) -> usize {
    let search_start = pos.saturating_sub(100);
    for i in (search_start..pos).rev() {
        if chars[i].is_whitespace() {
            return i + 1; // split after the whitespace
        }
    }
    pos // fallback: hard split at pos
}

/// Find the start of the next word at or after `pos`.
fn find_next_word_start(chars: &[char], pos: usize) -> Option<usize> {
    // Skip any whitespace at pos
    let skip_ws = chars[pos..].iter().position(|c| !c.is_whitespace())?;
    Some(pos + skip_ws)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_text_single_chunk() {
        let result = split_into_chunks("Hello world", 512, 50);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "Hello world");
    }

    #[test]
    fn test_long_text_splits_at_whitespace() {
        // Build a text of 600 chars split into discrete words of 10 chars each
        let word = "abcdefghi "; // 10 chars including space
        let text: String = word.repeat(60); // 600 chars
        let chunks = split_into_chunks(&text, 512, 50);
        assert!(chunks.len() >= 2, "Should produce at least 2 chunks");
        for chunk in &chunks {
            assert!(
                chunk.chars().count() <= 512,
                "Each chunk must be <= 512 chars, got {}",
                chunk.chars().count()
            );
        }
    }

    #[test]
    fn test_overlap_provides_continuity() {
        let text = "a ".repeat(300); // 600 chars
        let chunks = split_into_chunks(&text, 100, 20);
        assert!(chunks.len() >= 2);
        // The tail of chunk[0] should appear at the start of chunk[1]
        let c0_end: String = chunks[0].chars().rev().take(20).collect::<String>().chars().rev().collect();
        assert!(chunks[1].starts_with(c0_end.trim()));
    }


    #[test]
    fn test_chunk_empty_document() {
        use protocol::{UnifiedDocument, MetaParams, PrimaryLanguage};
        let doc = UnifiedDocument {
            document_id: "empty-doc".to_string(),
            document_type: "pdf".to_string(),
            page_count: 0,
            meta_params: MetaParams {
                primary_language: PrimaryLanguage::En,
                fallback_languages: vec![],
            },
            pages: vec![],
        };
        let chunks = chunk_document(&doc);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_hard_split_no_whitespace() {
        // A single string of 600 'a's with no spaces
        let text = "a".repeat(600);
        let chunks = split_into_chunks(&text, 512, 50);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].len(), 512);
    }

    #[test]
    fn test_multi_page_chunking() {
        use protocol::{BlockType, Page, TextBlock, UnifiedDocument, MetaParams, PrimaryLanguage};
        let doc = UnifiedDocument {
            document_id: "multi-page".to_string(),
            document_type: "pdf".to_string(),
            page_count: 2,
            meta_params: MetaParams {
                primary_language: PrimaryLanguage::En,
                fallback_languages: vec![],
            },
            pages: vec![
                Page {
                    page_num: 1,
                    blocks: vec![TextBlock {
                        block_id: "p1".to_string(),
                        block_type: BlockType::Text,
                        text: Some("Page 1 content".to_string()),
                        html_representation: None,
                        bounding_box: [0.0, 0.0, 0.0, 0.0],
                        confidence: 1.0,
                        reading_order: 0,
                        lang_detected: None,
                        metadata: None,
                    }],
                },
                Page {
                    page_num: 2,
                    blocks: vec![TextBlock {
                        block_id: "p2".to_string(),
                        block_type: BlockType::Text,
                        text: Some("Page 2 content".to_string()),
                        html_representation: None,
                        bounding_box: [0.0, 0.0, 0.0, 0.0],
                        confidence: 1.0,
                        reading_order: 0,
                        lang_detected: None,
                        metadata: None,
                    }],
                },
            ],
        };
        let chunks = chunk_document(&doc);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].page_num, 1);
        assert_eq!(chunks[1].page_num, 2);
    }
}
