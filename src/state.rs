use parking_lot::RwLock;
use std::sync::Arc;

use crate::config::{Config, LlmConfig};
use crate::models::Repo;
use crate::search::bm25::Bm25Index;
use crate::search::vector::VectorStore;

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    pub config: Config,
    pub repos: Arc<RwLock<Vec<Repo>>>,
    pub bm25: Arc<Bm25Index>,
    pub vectors: Arc<VectorStore>,
    pub http_client: reqwest::Client,
    pub llm_config: Arc<RwLock<LlmConfig>>,
    pub clone_semaphore: Arc<tokio::sync::Semaphore>,
    pub chat_semaphore: Arc<tokio::sync::Semaphore>,
}

impl AppState {
    pub fn new(config: Config) -> anyhow::Result<Self> {
        // Ensure data directories exist
        std::fs::create_dir_all(config.repos_dir())?;
        std::fs::create_dir_all(config.index_dir())?;
        std::fs::create_dir_all(config.vector_dir())?;

        // Load persisted repos
        let repos = if config.db_path().exists() {
            let data = std::fs::read_to_string(config.db_path())?;
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            Vec::new()
        };

        let bm25 = Bm25Index::open_or_create(&config.index_dir())?;
        let vectors =
            VectorStore::open_or_create_with_limit(&config.vector_dir(), config.max_vector_entries)?;

        let llm_config = config.llm.clone();
        let max_concurrent_clones = config.max_concurrent_clones;

        Ok(Self {
            config,
            repos: Arc::new(RwLock::new(repos)),
            bm25: Arc::new(bm25),
            vectors: Arc::new(vectors),
            http_client: reqwest::Client::builder()
                .connect_timeout(std::time::Duration::from_secs(10))
                .timeout(std::time::Duration::from_secs(120))
                .build()?,
            llm_config: Arc::new(RwLock::new(llm_config)),
            clone_semaphore: Arc::new(tokio::sync::Semaphore::new(max_concurrent_clones)),
            chat_semaphore: Arc::new(tokio::sync::Semaphore::new(3)),
        })
    }

    /// Persist repos list to disk (atomic write via temp file + rename).
    pub fn persist_repos(&self) {
        let repos = self.repos.read();
        if let Ok(data) = serde_json::to_string_pretty(&*repos) {
            let db_path = self.config.db_path();
            let tmp_path = db_path.with_extension("json.tmp");
            if std::fs::write(&tmp_path, &data).is_ok() {
                let _ = std::fs::rename(&tmp_path, &db_path);
            }
        }
    }
}
