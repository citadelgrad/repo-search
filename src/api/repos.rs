use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use chrono::Utc;
use uuid::Uuid;

use crate::models::{AddRepoRequest, FileChunk, LlmConfigUpdate, Repo, RepoStatus};
use crate::state::AppState;

/// GET /api/repos - List all repos
pub async fn list_repos(State(state): State<AppState>) -> Json<Vec<Repo>> {
    let repos = state.repos.read();
    Json(repos.clone())
}

/// POST /api/repos - Add a new repo (clone + index in background)
pub async fn add_repo(
    State(state): State<AppState>,
    Json(req): Json<AddRepoRequest>,
) -> Result<(StatusCode, Json<Repo>), (StatusCode, String)> {
    let url = req.url.trim().to_string();
    if url.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "URL is required".to_string()));
    }

    // Derive repo name from URL
    let name = url
        .rsplit('/')
        .next()
        .unwrap_or("repo")
        .trim_end_matches(".git")
        .to_string();

    let repo = Repo {
        id: Uuid::new_v4(),
        url: url.clone(),
        name: name.clone(),
        status: RepoStatus::Cloning,
        added_at: Utc::now(),
        indexed_at: None,
        file_count: 0,
    };

    // Save repo to state
    {
        let mut repos = state.repos.write();
        repos.push(repo.clone());
        drop(repos);
        state.persist_repos();
    }

    // Spawn background task to clone and index
    let repo_id = repo.id;
    let state_clone = state.clone();
    tokio::spawn(async move {
        if let Err(e) = clone_and_index(state_clone, repo_id, &url, &name).await {
            tracing::error!("Failed to clone and index {url}: {e:#}");
        }
    });

    Ok((StatusCode::CREATED, Json(repo)))
}

/// DELETE /api/repos/:id - Remove a repo and its index data
pub async fn delete_repo(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, (StatusCode, String)> {
    let exists = {
        let repos = state.repos.read();
        repos.iter().any(|r| r.id == id)
    };

    if !exists {
        return Err((StatusCode::NOT_FOUND, "Repo not found".to_string()));
    }

    // Remove from BM25 index
    if let Err(e) = state.bm25.delete_repo(&id) {
        tracing::warn!("Failed to delete BM25 data for {id}: {e}");
    }

    // Remove from vector store
    if let Err(e) = state.vectors.delete_repo(&id) {
        tracing::warn!("Failed to delete vector data for {id}: {e}");
    }

    // Remove cloned files
    let repo_dir = state.config.repos_dir().join(id.to_string());
    if repo_dir.exists() {
        let _ = std::fs::remove_dir_all(&repo_dir);
    }

    // Remove from state
    {
        let mut repos = state.repos.write();
        repos.retain(|r| r.id != id);
        drop(repos);
        state.persist_repos();
    }

    Ok(StatusCode::NO_CONTENT)
}

/// GET /api/config - Get current LLM config
pub async fn get_config(State(state): State<AppState>) -> Json<crate::config::LlmConfig> {
    let config = state.llm_config.read();
    Json(config.clone())
}

/// PUT /api/config - Update LLM config
pub async fn update_config(
    State(state): State<AppState>,
    Json(update): Json<LlmConfigUpdate>,
) -> Json<crate::config::LlmConfig> {
    let mut config = state.llm_config.write();

    if let Some(provider) = update.provider {
        config.provider = provider;
    }
    if let Some(base_url) = update.base_url {
        config.base_url = base_url;
    }
    if let Some(chat_model) = update.chat_model {
        config.chat_model = chat_model;
    }
    if let Some(embedding_model) = update.embedding_model {
        config.embedding_model = embedding_model;
    }
    if let Some(api_key) = update.api_key {
        config.api_key = Some(api_key);
    }
    if let Some(embedding_dim) = update.embedding_dim {
        config.embedding_dim = embedding_dim;
    }

    Json(config.clone())
}

/// Clone a repo and index its files.
async fn clone_and_index(
    state: AppState,
    repo_id: Uuid,
    url: &str,
    repo_name: &str,
) -> anyhow::Result<()> {
    let repo_dir = state.config.repos_dir().join(repo_id.to_string());

    // Clone
    update_repo_status(&state, repo_id, RepoStatus::Cloning);
    let url_owned = url.to_string();
    let repo_dir_clone = repo_dir.clone();
    tokio::task::spawn_blocking(move || crate::git::clone_repo(&url_owned, &repo_dir_clone))
        .await??;

    // Walk files and chunk them
    update_repo_status(&state, repo_id, RepoStatus::Indexing);
    let repo_dir_walk = repo_dir.clone();
    let files =
        tokio::task::spawn_blocking(move || crate::git::walk_repo_files(&repo_dir_walk)).await?;

    tracing::info!("Found {} indexable files in {repo_name}", files.len());

    let mut all_chunks = Vec::new();
    for file in &files {
        let chunks = chunk_file(repo_id, &file.relative_path, &file.content, &file.language);
        all_chunks.extend(chunks);
    }

    tracing::info!("Created {} chunks for {repo_name}", all_chunks.len());

    // BM25 index
    let chunks_for_bm25 = all_chunks.clone();
    let repo_name_bm25 = repo_name.to_string();
    let bm25 = state.bm25.clone();
    tokio::task::spawn_blocking(move || bm25.index_chunks(&repo_name_bm25, &chunks_for_bm25))
        .await??;
    tracing::info!("BM25 indexing complete for {repo_name}");

    // Vector embeddings (best effort - may fail if LLM not available)
    let texts: Vec<String> = all_chunks
        .iter()
        .map(|c| {
            // Prepend file path for better embedding context
            format!("File: {}\n{}", c.file_path, c.content)
        })
        .collect();

    let chunk_metadata: Vec<(String, usize, String, String, usize, usize)> = all_chunks
        .iter()
        .map(|c| {
            (
                c.file_path.clone(),
                c.chunk_index,
                c.content.clone(),
                c.language.clone(),
                c.start_line,
                c.end_line,
            )
        })
        .collect();

    let llm_config = state.llm_config.read().clone();
    match crate::llm::embeddings::embed_batch(&state.http_client, &llm_config, &texts).await {
        Ok(embeddings) => {
            state
                .vectors
                .add_chunks(repo_id, repo_name, &chunk_metadata, embeddings)?;
            tracing::info!("Vector indexing complete for {repo_name}");
        }
        Err(e) => {
            tracing::warn!(
                "Vector embedding failed for {repo_name} (LLM may not be running): {e:#}"
            );
        }
    }

    // Mark as ready
    {
        let mut repos = state.repos.write();
        if let Some(repo) = repos.iter_mut().find(|r| r.id == repo_id) {
            repo.status = RepoStatus::Ready;
            repo.indexed_at = Some(Utc::now());
            repo.file_count = files.len();
        }
        drop(repos);
        state.persist_repos();
    }

    tracing::info!("Repo {repo_name} is ready for search");
    Ok(())
}

fn update_repo_status(state: &AppState, repo_id: Uuid, status: RepoStatus) {
    let mut repos = state.repos.write();
    if let Some(repo) = repos.iter_mut().find(|r| r.id == repo_id) {
        repo.status = status;
    }
    drop(repos);
    state.persist_repos();
}

/// Split a file into overlapping chunks of ~100 lines each.
fn chunk_file(
    repo_id: Uuid,
    file_path: &str,
    content: &str,
    language: &str,
) -> Vec<FileChunk> {
    let lines: Vec<&str> = content.lines().collect();

    if lines.is_empty() {
        return Vec::new();
    }

    let chunk_size = 100;
    let overlap = 20;
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < lines.len() {
        let end = (start + chunk_size).min(lines.len());
        let chunk_content = lines[start..end].join("\n");

        chunks.push(FileChunk {
            repo_id,
            file_path: file_path.to_string(),
            chunk_index: chunks.len(),
            content: chunk_content,
            language: language.to_string(),
            start_line: start + 1,
            end_line: end,
        });

        if end >= lines.len() {
            break;
        }

        start += chunk_size - overlap;
    }

    chunks
}
