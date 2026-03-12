use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The primary language target as a hint for OCR and Data Pipeline
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum PrimaryLanguage {
    Ja,
    Zh,
    En,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MetaParams {
    pub primary_language: PrimaryLanguage,
    pub fallback_languages: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum BlockType {
    Text,
    Title,
    Table,
    ImageCaption,
    Header,
    Footer,
    MathematicalFormula,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BlockMetadata {
    pub font_size: Option<f32>,
    pub is_vertical: Option<bool>,
    pub char_distribution: Option<HashMap<String, f32>>,
}

/// A universally mapped block of text from an OCR or PDF extraction pipeline
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TextBlock {
    pub block_id: String,
    pub block_type: BlockType,
    pub text: Option<String>,
    pub html_representation: Option<String>,
    pub bounding_box: [f32; 4], // [x0, y0, x1, y1]
    pub confidence: f32,
    pub reading_order: i32,
    pub lang_detected: Option<String>,
    pub metadata: Option<BlockMetadata>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Page {
    pub page_num: i32,
    pub blocks: Vec<TextBlock>,
}

/// The Root Document structure connecting Rust and Python
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UnifiedDocument {
    pub document_id: String,
    pub document_type: String, // e.g., "scanned_pdf", "digital_pdf"
    pub page_count: i32,
    pub meta_params: MetaParams,
    pub pages: Vec<Page>,
}

// ---------------------------------------------------------------------------
// RAG Pipeline Types
// ---------------------------------------------------------------------------

/// A semantically coherent text chunk derived from a UnifiedDocument.
/// This is the unit of embedding and vector storage in Qdrant.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TextChunk {
    /// Deterministic ID: "{document_id}_p{page_num}_c{chunk_idx}"
    pub chunk_id: String,
    pub document_id: String,
    pub page_num: i32,
    pub text: String,
    pub char_count: usize,
}

/// Request sent to the Embedding Service (port 8003).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EmbedRequest {
    pub texts: Vec<String>,
}

/// Response from the Embedding Service.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EmbedResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub dim: usize,
}

/// A single candidate chunk sent to the Reranker Service (port 8004).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RerankCandidate {
    pub chunk_id: String,
    pub text: String,
    pub page_num: i32,
}

/// Request sent to the Reranker Service.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RerankRequest {
    pub question: String,
    pub candidates: Vec<RerankCandidate>,
}

/// A single ranked result returned by the Reranker Service.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RankedChunk {
    pub chunk_id: String,
    pub text: String,
    pub page_num: i32,
    /// Relevance score (higher = more relevant)
    pub score: f32,
}

/// Response from the Reranker Service.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RerankResponse {
    /// Ordered by descending relevance score
    pub ranked: Vec<RankedChunk>,
}

/// Request body for the /api/v1/query gateway endpoint.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct QueryRequest {
    pub question: String,
    /// Optional: restrict search to a single ingested document
    pub doc_id: Option<String>,
    /// Number of results to include in LLM context (after reranking)
    pub top_k: Option<usize>,
}

/// Response from the /api/v1/query endpoint.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct QueryResponse {
    pub answer: String,
    pub sources: Vec<RankedChunk>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_unified_document_serialization() {
        let doc = UnifiedDocument {
            document_id: "test-id".to_string(),
            document_type: "pdf".to_string(),
            page_count: 1,
            meta_params: MetaParams {
                primary_language: PrimaryLanguage::Ja,
                fallback_languages: vec!["en".to_string()],
            },
            pages: vec![Page {
                page_num: 1,
                blocks: vec![TextBlock {
                    block_id: "b1".to_string(),
                    block_type: BlockType::Text,
                    text: Some("Hello".to_string()),
                    html_representation: None,
                    bounding_box: [0.0, 0.0, 1.0, 1.0],
                    confidence: 0.95,
                    reading_order: 0,
                    lang_detected: Some("ja".to_string()),
                    metadata: None,
                }],
            }],
        };

        let serialized = serde_json::to_string(&doc).unwrap();
        let deserialized: UnifiedDocument = serde_json::from_str(&serialized).unwrap();
        assert_eq!(doc.document_id, deserialized.document_id);
        assert_eq!(doc.pages[0].blocks[0].text, deserialized.pages[0].blocks[0].text);
    }

    #[test]
    fn test_primary_language_enum() {
        let lang = PrimaryLanguage::Ja;
        let serialized = serde_json::to_string(&lang).unwrap();
        assert_eq!(serialized, "\"ja\"");
    }
}
