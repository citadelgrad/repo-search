use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Where cloned repos and index data are stored
    pub data_dir: PathBuf,
    /// Server bind address
    pub bind_addr: String,
    /// LLM provider configuration
    pub llm: LlmConfig,
    /// Cross-encoder reranker configuration
    pub reranker: RerankerConfig,
    /// Maximum number of repos allowed
    pub max_repos: usize,
    /// Maximum concurrent clone operations
    pub max_concurrent_clones: usize,
    /// Clone timeout in seconds
    pub clone_timeout_secs: u64,
    /// Maximum repo size in MB (checked after clone)
    pub max_repo_size_mb: u64,
    /// Maximum total vector entries in memory (0 = unlimited)
    pub max_vector_entries: usize,
    /// Git personal access token for cloning private repos
    pub git_token: Option<String>,
}

/// Configuration for the cross-encoder reranker sidecar (e.g. llama-server with Qwen3-Reranker).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankerConfig {
    /// Base URL for the reranker API (e.g. "http://127.0.0.1:8082").
    /// If None, reranking falls back to RRF-only.
    pub base_url: Option<String>,
    /// Model name to send in the rerank request.
    pub model: Option<String>,
    /// Request timeout in seconds (capped at 30).
    pub timeout_secs: u64,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            base_url: None,
            model: None,
            timeout_secs: 10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// "ollama" or "openai"
    pub provider: String,
    /// Base URL for the LLM API
    pub base_url: String,
    /// Model name for chat/reranking
    pub chat_model: String,
    /// Model name for embeddings
    pub embedding_model: String,
    /// API key (only needed for cloud providers)
    pub api_key: Option<String>,
    /// Embedding vector dimension
    pub embedding_dim: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./data"),
            bind_addr: "127.0.0.1:9000".to_string(),
            llm: LlmConfig::default(),
            reranker: RerankerConfig::default(),
            max_repos: 50,
            max_concurrent_clones: 2,
            clone_timeout_secs: 300,
            max_repo_size_mb: 500,
            max_vector_entries: 500_000,
            git_token: None,
        }
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: "ollama".to_string(),
            base_url: "http://localhost:11434".to_string(),
            chat_model: "llama3.2".to_string(),
            embedding_model: "nomic-embed-text".to_string(),
            api_key: None,
            embedding_dim: 768,
        }
    }
}

impl Config {
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(dir) = std::env::var("REPO_SEARCH_DATA_DIR") {
            config.data_dir = PathBuf::from(dir);
        }
        if let Ok(addr) = std::env::var("REPO_SEARCH_BIND_ADDR") {
            config.bind_addr = addr;
        }
        if let Ok(provider) = std::env::var("LLM_PROVIDER") {
            config.llm.provider = provider;
        }
        if let Ok(url) = std::env::var("LLM_BASE_URL") {
            config.llm.base_url = url;
        }
        if let Ok(model) = std::env::var("LLM_CHAT_MODEL") {
            config.llm.chat_model = model;
        }
        if let Ok(model) = std::env::var("LLM_EMBEDDING_MODEL") {
            config.llm.embedding_model = model;
        }
        if let Ok(key) = std::env::var("LLM_API_KEY") {
            config.llm.api_key = Some(key);
        }
        if let Ok(dim) = std::env::var("LLM_EMBEDDING_DIM") {
            if let Ok(d) = dim.parse() {
                config.llm.embedding_dim = d;
            }
        }
        if let Ok(val) = std::env::var("REPO_SEARCH_MAX_REPOS") {
            if let Ok(v) = val.parse() {
                config.max_repos = v;
            }
        }
        if let Ok(val) = std::env::var("REPO_SEARCH_MAX_CONCURRENT_CLONES") {
            if let Ok(v) = val.parse() {
                config.max_concurrent_clones = v;
            }
        }
        if let Ok(val) = std::env::var("REPO_SEARCH_CLONE_TIMEOUT_SECS") {
            if let Ok(v) = val.parse() {
                config.clone_timeout_secs = v;
            }
        }
        if let Ok(val) = std::env::var("REPO_SEARCH_MAX_REPO_SIZE_MB") {
            if let Ok(v) = val.parse() {
                config.max_repo_size_mb = v;
            }
        }
        if let Ok(val) = std::env::var("REPO_SEARCH_MAX_VECTOR_ENTRIES") {
            if let Ok(v) = val.parse() {
                config.max_vector_entries = v;
            }
        }
        if let Ok(token) = std::env::var("REPO_SEARCH_GIT_TOKEN") {
            config.git_token = Some(token);
        }

        // Reranker config
        if let Ok(url) = std::env::var("RERANKER_BASE_URL") {
            config.reranker.base_url = Some(url);
        }
        if let Ok(model) = std::env::var("RERANKER_MODEL") {
            config.reranker.model = Some(model);
        }
        if let Ok(val) = std::env::var("RERANKER_TIMEOUT_SECS") {
            if let Ok(v) = val.parse::<u64>() {
                config.reranker.timeout_secs = v.min(30); // Cap at 30s
            }
        }

        config
    }

    pub fn repos_dir(&self) -> PathBuf {
        self.data_dir.join("repos")
    }

    pub fn index_dir(&self) -> PathBuf {
        self.data_dir.join("index")
    }

    pub fn vector_dir(&self) -> PathBuf {
        self.data_dir.join("vectors")
    }

    pub fn db_path(&self) -> PathBuf {
        self.data_dir.join("repos.json")
    }
}
