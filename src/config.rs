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
            bind_addr: "127.0.0.1:7000".to_string(),
            llm: LlmConfig::default(),
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
