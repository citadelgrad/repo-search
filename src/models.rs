use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A tracked repository
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Repo {
    pub id: Uuid,
    pub url: String,
    pub name: String,
    pub status: RepoStatus,
    pub added_at: DateTime<Utc>,
    pub indexed_at: Option<DateTime<Utc>>,
    pub file_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RepoStatus {
    Cloning,
    Indexing,
    Ready,
    Error(String),
}

/// A single indexed file chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChunk {
    pub repo_id: Uuid,
    pub file_path: String,
    pub chunk_index: usize,
    pub content: String,
    pub language: String,
    pub start_line: usize,
    pub end_line: usize,
}

/// A search result before re-ranking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub repo_id: Uuid,
    pub repo_name: String,
    pub file_path: String,
    pub chunk_index: usize,
    pub content: String,
    pub language: String,
    pub start_line: usize,
    pub end_line: usize,
    pub bm25_score: f32,
    pub vector_score: f32,
    pub combined_score: f32,
    pub rerank_score: Option<f32>,
}

/// Search request
#[derive(Debug, Clone, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default = "default_true")]
    pub use_bm25: bool,
    #[serde(default = "default_true")]
    pub use_vector: bool,
    #[serde(default)]
    pub use_rerank: bool,
    /// Filter by repo IDs
    pub repo_ids: Option<Vec<Uuid>>,
}

fn default_limit() -> usize {
    20
}

fn default_true() -> bool {
    true
}

/// Add-repo request
#[derive(Debug, Clone, Deserialize)]
pub struct AddRepoRequest {
    pub url: String,
}

/// Search response
#[derive(Debug, Clone, Serialize)]
pub struct SearchResponse {
    pub query: String,
    pub results: Vec<SearchHit>,
    pub total_bm25_hits: usize,
    pub total_vector_hits: usize,
}

/// LLM config update request
#[derive(Debug, Clone, Deserialize)]
pub struct LlmConfigUpdate {
    pub provider: Option<String>,
    // base_url intentionally omitted: immutable at runtime to prevent SSRF
    pub chat_model: Option<String>,
    pub embedding_model: Option<String>,
    pub api_key: Option<String>,
    pub embedding_dim: Option<usize>,
}
