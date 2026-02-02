use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use chrono::Utc;
use uuid::Uuid;

use crate::models::{AddRepoRequest, FileChunk, LlmConfigUpdate, ReorderRequest, Repo, RepoStatus};
use crate::state::AppState;

/// GET /api/repos - List all repos (pinned first, preserving relative order)
pub async fn list_repos(State(state): State<AppState>) -> Json<Vec<Repo>> {
    let repos = state.repos.read();
    let mut sorted = repos.clone();
    sorted.sort_by_key(|r| !r.pinned);
    Json(sorted)
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

    // Security: only allow https:// and git:// URLs to prevent SSRF and local file access
    if !url.starts_with("https://") && !url.starts_with("git://") && !url.starts_with("http://") {
        return Err((
            StatusCode::BAD_REQUEST,
            "Only https://, http://, and git:// URLs are allowed".to_string(),
        ));
    }

    // Reject duplicate URLs and enforce repo limit
    {
        let repos = state.repos.read();
        if repos.iter().any(|r| r.url == url) {
            return Err((
                StatusCode::CONFLICT,
                "A repo with this URL has already been added".to_string(),
            ));
        }
        if repos.len() >= state.config.max_repos {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "Maximum number of repos ({}) reached",
                    state.config.max_repos
                ),
            ));
        }
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
        head_commit: None,
        pinned: false,
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
        let err_state = state_clone.clone();
        if let Err(e) = clone_and_index(state_clone, repo_id, &url, &name).await {
            tracing::error!("Failed to clone and index {url}: {e:#}");
            update_repo_status(&err_state, repo_id, RepoStatus::Error(format!("{e:#}")));
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

/// PATCH /api/repos/:id/pin - Toggle pin status
pub async fn pin_repo(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<Repo>, (StatusCode, String)> {
    let mut repos = state.repos.write();
    let repo = repos.iter_mut().find(|r| r.id == id);
    match repo {
        None => return Err((StatusCode::NOT_FOUND, "Repo not found".to_string())),
        Some(r) => {
            r.pinned = !r.pinned;
        }
    }

    // Reposition: move the toggled repo to end of pinned group (if pinning)
    // or to the start of unpinned group (if unpinning).
    // Find the repo's current index and its new pinned state.
    let idx = repos.iter().position(|r| r.id == id).unwrap();
    let is_now_pinned = repos[idx].pinned;
    let repo_moved = repos.remove(idx);

    if is_now_pinned {
        // Insert at end of pinned group (before first unpinned)
        let insert_at = repos.iter().position(|r| !r.pinned).unwrap_or(repos.len());
        repos.insert(insert_at, repo_moved);
    } else {
        // Insert at start of unpinned group (after last pinned)
        let insert_at = repos.iter().rposition(|r| r.pinned).map(|i| i + 1).unwrap_or(0);
        repos.insert(insert_at, repo_moved);
    }

    let updated = repos.iter().find(|r| r.id == id).cloned().unwrap();
    drop(repos);
    state.persist_repos();
    Ok(Json(updated))
}

/// PUT /api/repos/order - Bulk reorder repos
pub async fn reorder_repos(
    State(state): State<AppState>,
    Json(req): Json<ReorderRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    let mut repos = state.repos.write();

    if req.order.len() != repos.len() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "Order list has {} IDs but there are {} repos",
                req.order.len(),
                repos.len()
            ),
        ));
    }

    // Validate all IDs exist
    for id in &req.order {
        if !repos.iter().any(|r| r.id == *id) {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("Unknown repo ID: {id}"),
            ));
        }
    }

    // Reorder the Vec to match the requested order
    let mut reordered = Vec::with_capacity(repos.len());
    for id in &req.order {
        let repo = repos.iter().find(|r| r.id == *id).cloned().unwrap();
        reordered.push(repo);
    }
    *repos = reordered;

    drop(repos);
    state.persist_repos();
    Ok(StatusCode::NO_CONTENT)
}

/// GET /api/config - Get current LLM config (API key redacted)
pub async fn get_config(State(state): State<AppState>) -> Json<LlmConfigResponse> {
    let config = state.llm_config.read();
    Json(LlmConfigResponse {
        provider: config.provider.clone(),
        base_url: config.base_url.clone(),
        chat_model: config.chat_model.clone(),
        embedding_model: config.embedding_model.clone(),
        embedding_dim: config.embedding_dim,
        has_api_key: config.api_key.is_some(),
    })
}

/// Config response with API key redacted
#[derive(serde::Serialize)]
pub struct LlmConfigResponse {
    pub provider: String,
    pub base_url: String,
    pub chat_model: String,
    pub embedding_model: String,
    pub embedding_dim: usize,
    pub has_api_key: bool,
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
    // base_url is immutable at runtime (set via LLM_BASE_URL env var only)
    // to prevent SSRF: an attacker changing it could exfiltrate the API key
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

/// POST /api/repos/:id/reindex - Re-generate vector embeddings for an already-indexed repo
pub async fn reindex_repo(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, (StatusCode, String)> {
    // Validate repo exists and is in a re-indexable state (Ready or Error)
    {
        let repos = state.repos.read();
        let repo = repos.iter().find(|r| r.id == id);
        match repo {
            None => return Err((StatusCode::NOT_FOUND, "Repo not found".to_string())),
            Some(r)
                if !matches!(
                    r.status,
                    RepoStatus::Ready | RepoStatus::Error(_) | RepoStatus::Embedding
                ) =>
            {
                return Err((
                    StatusCode::CONFLICT,
                    "Repo must be in ready, error, or embedding state to re-index".to_string(),
                ));
            }
            _ => {}
        }
    }

    // Set status to Embedding
    update_repo_status(&state, id, RepoStatus::Embedding);

    // Spawn background task
    let state_clone = state.clone();
    tokio::spawn(async move {
        let err_state = state_clone.clone();
        if let Err(e) = reindex_embeddings(state_clone, id).await {
            tracing::error!("Failed to re-index embeddings for {id}: {e:#}");
            update_repo_status(&err_state, id, RepoStatus::Error(format!("{e:#}")));
        }
    });

    Ok(StatusCode::OK)
}

/// Re-generate vector embeddings for an already-cloned repo.
async fn reindex_embeddings(state: AppState, repo_id: Uuid) -> anyhow::Result<()> {
    let (repo_name, repo_dir) = {
        let repos = state.repos.read();
        let repo = repos
            .iter()
            .find(|r| r.id == repo_id)
            .ok_or_else(|| anyhow::anyhow!("Repo not found"))?;
        (
            repo.name.clone(),
            state.config.repos_dir().join(repo_id.to_string()),
        )
    };

    if !repo_dir.exists() {
        update_repo_status(
            &state,
            repo_id,
            RepoStatus::Error("Cloned repo directory not found".to_string()),
        );
        anyhow::bail!("Cloned repo directory not found for {repo_name}");
    }

    // Walk files and chunk them
    let repo_dir_walk = repo_dir.clone();
    let files =
        tokio::task::spawn_blocking(move || crate::git::walk_repo_files(&repo_dir_walk)).await?;

    tracing::info!(
        "Re-indexing: found {} files in {repo_name}",
        files.len()
    );

    let mut all_chunks = Vec::new();
    for file in &files {
        let chunks = chunk_file(repo_id, &file.relative_path, &file.content, &file.language);
        all_chunks.extend(chunks);
    }

    // Clear stale vectors
    if let Err(e) = state.vectors.delete_repo(&repo_id) {
        tracing::warn!("Failed to delete old vectors for {repo_name}: {e}");
    }

    // Generate new embeddings
    let texts: Vec<String> = all_chunks
        .iter()
        .map(|c| format!("File: {}\n{}", c.file_path, c.content))
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
    match crate::llm::embeddings::embed_batch(&state.http_client, &llm_config, &texts, crate::llm::embeddings::EmbedTask::SearchDocument).await {
        Ok(embeddings) => {
            state
                .vectors
                .add_chunks(repo_id, &repo_name, &chunk_metadata, embeddings)?;
            tracing::info!("Re-index: vector embeddings complete for {repo_name}");
        }
        Err(e) => {
            update_repo_status(
                &state,
                repo_id,
                RepoStatus::Error(format!("Embedding failed: {e:#}")),
            );
            anyhow::bail!("Embedding failed for {repo_name}: {e:#}");
        }
    }

    // Mark as ready again
    {
        let mut repos = state.repos.write();
        if let Some(repo) = repos.iter_mut().find(|r| r.id == repo_id) {
            repo.status = RepoStatus::Ready;
            repo.indexed_at = Some(Utc::now());
        }
        drop(repos);
        state.persist_repos();
    }

    tracing::info!("Re-index complete for {repo_name}");
    Ok(())
}

/// POST /api/repos/:id/sync - Fetch latest commits and re-index
pub async fn sync_repo(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<(StatusCode, Json<Repo>), (StatusCode, String)> {
    let repo = {
        let repos = state.repos.read();
        let repo = repos.iter().find(|r| r.id == id);
        match repo {
            None => return Err((StatusCode::NOT_FOUND, "Repo not found".to_string())),
            Some(r) => {
                match &r.status {
                    RepoStatus::Ready | RepoStatus::Error(_) => {}
                    _ => {
                        return Err((
                            StatusCode::CONFLICT,
                            "Repo is already being processed".to_string(),
                        ));
                    }
                }
                r.clone()
            }
        }
    };

    // Check that the repo directory still exists on disk
    let repo_dir = state.config.repos_dir().join(id.to_string());
    if !repo_dir.exists() {
        return Err((
            StatusCode::BAD_REQUEST,
            "Repo directory not found on disk. Delete and re-add the repo.".to_string(),
        ));
    }

    // Set status to Indexing and return immediately
    update_repo_status(&state, id, RepoStatus::Indexing);
    let updated_repo = {
        let repos = state.repos.read();
        repos.iter().find(|r| r.id == id).cloned().unwrap()
    };

    // Spawn background sync task
    let state_clone = state.clone();
    let repo_url = repo.url.clone();
    let repo_name = repo.name.clone();
    let old_head = repo.head_commit.clone();
    tokio::spawn(async move {
        let err_state = state_clone.clone();
        if let Err(e) = sync_and_reindex(state_clone, id, &repo_url, &repo_name, old_head).await {
            tracing::error!("Failed to sync {repo_name}: {e:#}");
            update_repo_status(&err_state, id, RepoStatus::Error(format!("{e:#}")));
        }
    });

    Ok((StatusCode::OK, Json(updated_repo)))
}

/// Fetch latest commits and re-index if HEAD changed.
async fn sync_and_reindex(
    state: AppState,
    repo_id: Uuid,
    url: &str,
    repo_name: &str,
    old_head: Option<String>,
) -> anyhow::Result<()> {
    let repo_dir = state.config.repos_dir().join(repo_id.to_string());

    // Acquire clone permit (shared with clone operations)
    let _permit = state
        .clone_semaphore
        .acquire()
        .await
        .map_err(|_| anyhow::anyhow!("Clone semaphore closed"))?;

    // Pull with timeout
    let url_owned = url.to_string();
    let repo_dir_pull = repo_dir.clone();
    let git_token = state.config.git_token.clone();
    let timeout = std::time::Duration::from_secs(state.config.clone_timeout_secs);

    let pull_result = tokio::time::timeout(
        timeout,
        tokio::task::spawn_blocking(move || {
            crate::git::pull_repo(&repo_dir_pull, &url_owned, git_token.as_deref())
        }),
    )
    .await;

    let new_head = match pull_result {
        Ok(Ok(Ok(sha))) => sha,
        Ok(Ok(Err(e))) => {
            update_repo_status(
                &state,
                repo_id,
                RepoStatus::Error(format!("Fetch failed: {e:#}")),
            );
            anyhow::bail!("Fetch failed for {repo_name}: {e:#}");
        }
        Ok(Err(e)) => {
            update_repo_status(
                &state,
                repo_id,
                RepoStatus::Error("Sync task failed".to_string()),
            );
            anyhow::bail!("Sync task failed: {e}");
        }
        Err(_) => {
            update_repo_status(
                &state,
                repo_id,
                RepoStatus::Error(format!(
                    "Sync timed out after {}s",
                    state.config.clone_timeout_secs
                )),
            );
            anyhow::bail!("Sync timed out");
        }
    };

    // Skip re-index if HEAD unchanged
    if old_head.as_deref() == Some(&new_head) {
        tracing::info!("Sync {repo_name}: HEAD unchanged ({new_head}), skipping re-index");
        update_repo_status(&state, repo_id, RepoStatus::Ready);
        return Ok(());
    }

    tracing::info!(
        "Sync {repo_name}: HEAD changed {} -> {new_head}, re-indexing",
        old_head.as_deref().unwrap_or("(unknown)")
    );

    // Delete existing index data
    if let Err(e) = state.bm25.delete_repo(&repo_id) {
        tracing::warn!("Failed to delete BM25 data for {repo_name}: {e}");
    }
    if let Err(e) = state.vectors.delete_repo(&repo_id) {
        tracing::warn!("Failed to delete vector data for {repo_name}: {e}");
    }

    // Re-walk and re-chunk
    let repo_dir_walk = repo_dir.clone();
    let files =
        tokio::task::spawn_blocking(move || crate::git::walk_repo_files(&repo_dir_walk)).await?;

    tracing::info!(
        "Sync: found {} indexable files in {repo_name}",
        files.len()
    );

    let mut all_chunks = Vec::new();
    for file in &files {
        let chunks = chunk_file(repo_id, &file.relative_path, &file.content, &file.language);
        all_chunks.extend(chunks);
    }

    tracing::info!("Sync: created {} chunks for {repo_name}", all_chunks.len());

    // BM25 index
    let chunks_for_bm25 = all_chunks.clone();
    let repo_name_bm25 = repo_name.to_string();
    let bm25 = state.bm25.clone();
    tokio::task::spawn_blocking(move || bm25.index_chunks(&repo_name_bm25, &chunks_for_bm25))
        .await??;
    tracing::info!("Sync: BM25 indexing complete for {repo_name}");

    // Vector embeddings (best effort)
    let texts: Vec<String> = all_chunks
        .iter()
        .map(|c| format!("File: {}\n{}", c.file_path, c.content))
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
    match crate::llm::embeddings::embed_batch(&state.http_client, &llm_config, &texts, crate::llm::embeddings::EmbedTask::SearchDocument).await {
        Ok(embeddings) => {
            state
                .vectors
                .add_chunks(repo_id, repo_name, &chunk_metadata, embeddings)?;
            tracing::info!("Sync: vector indexing complete for {repo_name}");
        }
        Err(e) => {
            tracing::warn!(
                "Sync: vector embedding failed for {repo_name} (LLM may not be running): {e:#}"
            );
        }
    }

    // Update repo metadata
    {
        let mut repos = state.repos.write();
        if let Some(repo) = repos.iter_mut().find(|r| r.id == repo_id) {
            repo.status = RepoStatus::Ready;
            repo.indexed_at = Some(Utc::now());
            repo.file_count = files.len();
            repo.head_commit = Some(new_head);
        }
        drop(repos);
        state.persist_repos();
    }

    tracing::info!("Sync complete for {repo_name}");
    Ok(())
}

/// Clone a repo and index its files.
async fn clone_and_index(
    state: AppState,
    repo_id: Uuid,
    url: &str,
    repo_name: &str,
) -> anyhow::Result<()> {
    let repo_dir = state.config.repos_dir().join(repo_id.to_string());

    // Acquire clone permit (limits concurrent clones)
    let _permit = state
        .clone_semaphore
        .acquire()
        .await
        .map_err(|_| anyhow::anyhow!("Clone semaphore closed"))?;

    // Clone with timeout
    update_repo_status(&state, repo_id, RepoStatus::Cloning);
    let url_owned = url.to_string();
    let repo_dir_clone = repo_dir.clone();
    let git_token = state.config.git_token.clone();
    let timeout = std::time::Duration::from_secs(state.config.clone_timeout_secs);

    let clone_result = tokio::time::timeout(
        timeout,
        tokio::task::spawn_blocking(move || {
            crate::git::clone_repo(&url_owned, &repo_dir_clone, git_token.as_deref())
        }),
    )
    .await;

    match clone_result {
        Ok(Ok(Ok(()))) => {}
        Ok(Ok(Err(e))) => {
            set_error_and_cleanup(&state, repo_id, &repo_dir, &format!("Clone failed: {e:#}"));
            return Err(e);
        }
        Ok(Err(e)) => {
            set_error_and_cleanup(&state, repo_id, &repo_dir, "Clone task failed");
            anyhow::bail!("Clone task failed: {e}");
        }
        Err(_) => {
            set_error_and_cleanup(
                &state,
                repo_id,
                &repo_dir,
                &format!(
                    "Clone timed out after {}s",
                    state.config.clone_timeout_secs
                ),
            );
            anyhow::bail!("Clone timed out");
        }
    }

    // Check repo size limit
    let max_bytes = state.config.max_repo_size_mb * 1024 * 1024;
    let size_check_dir = repo_dir.clone();
    let repo_size =
        tokio::task::spawn_blocking(move || crate::git::dir_size_bytes(&size_check_dir)).await?;
    if repo_size > max_bytes {
        set_error_and_cleanup(
            &state,
            repo_id,
            &repo_dir,
            &format!(
                "Repo size ({} MB) exceeds limit ({} MB)",
                repo_size / (1024 * 1024),
                state.config.max_repo_size_mb
            ),
        );
        anyhow::bail!("Repo exceeds size limit");
    }

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
    match crate::llm::embeddings::embed_batch(&state.http_client, &llm_config, &texts, crate::llm::embeddings::EmbedTask::SearchDocument).await {
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

    // Read HEAD commit SHA
    let head_commit = match crate::git::head_commit_sha(&repo_dir) {
        Ok(sha) => Some(sha),
        Err(e) => {
            tracing::warn!("Could not read HEAD for {repo_name}: {e}");
            None
        }
    };

    // Mark as ready
    {
        let mut repos = state.repos.write();
        if let Some(repo) = repos.iter_mut().find(|r| r.id == repo_id) {
            repo.status = RepoStatus::Ready;
            repo.indexed_at = Some(Utc::now());
            repo.file_count = files.len();
            repo.head_commit = head_commit;
        }
        drop(repos);
        state.persist_repos();
    }

    tracing::info!("Repo {repo_name} is ready for search");
    Ok(())
}

fn set_error_and_cleanup(
    state: &AppState,
    repo_id: Uuid,
    repo_dir: &std::path::Path,
    message: &str,
) {
    tracing::error!("{message}");
    update_repo_status(state, repo_id, RepoStatus::Error(message.to_string()));
    let _ = std::fs::remove_dir_all(repo_dir);
}

fn update_repo_status(state: &AppState, repo_id: Uuid, status: RepoStatus) {
    let mut repos = state.repos.write();
    if let Some(repo) = repos.iter_mut().find(|r| r.id == repo_id) {
        repo.status = status;
    }
    drop(repos);
    state.persist_repos();
}

/// Split a file into chunks using AST-aware chunking (for supported languages)
/// or a smart line-based fallback.
fn chunk_file(
    repo_id: Uuid,
    file_path: &str,
    content: &str,
    language: &str,
) -> Vec<FileChunk> {
    let raw_chunks = crate::chunking::chunk_code(content, language);

    raw_chunks
        .into_iter()
        .enumerate()
        .map(|(i, c)| FileChunk {
            repo_id,
            file_path: file_path.to_string(),
            chunk_index: i,
            content: c.content,
            language: language.to_string(),
            start_line: c.start_line,
            end_line: c.end_line,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_chunk_empty_file() {
        let id = Uuid::new_v4();
        let chunks = chunk_file(id, "empty.rs", "", "rust");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_small_rust_file_single_chunk() {
        let id = Uuid::new_v4();
        let content = "fn hello() {\n    println!(\"hello\");\n}\n";
        let chunks = chunk_file(id, "small.rs", content, "rust");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].repo_id, id);
        assert_eq!(chunks[0].file_path, "small.rs");
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[0].language, "rust");
        assert_eq!(chunks[0].start_line, 1);
        assert!(chunks[0].content.contains("fn hello"));
    }

    #[test]
    fn test_chunk_content_integrity() {
        let id = Uuid::new_v4();
        let content = "fn main() {\n    println!(\"hello\");\n}";
        let chunks = chunk_file(id, "main.rs", content, "rust");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("fn main()"));
        assert!(chunks[0].content.contains("println!"));
    }

    #[test]
    fn test_chunk_sequential_indices() {
        let id = Uuid::new_v4();
        // Create content large enough to produce multiple chunks
        let funcs: String = (0..100)
            .map(|i| format!("fn func_{i}() {{\n    let x = {i};\n    let y = x * 2;\n    let z = y + 1;\n}}\n\n"))
            .collect();
        let chunks = chunk_file(id, "many.rs", &funcs, "rust");
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_index, i);
        }
    }

    #[test]
    fn test_chunk_fallback_for_unknown_language() {
        let id = Uuid::new_v4();
        let content = "some text\nmore text\n";
        let chunks = chunk_file(id, "readme.md", content, "markdown");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].language, "markdown");
    }
}
