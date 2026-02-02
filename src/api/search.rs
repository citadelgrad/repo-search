use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;

use uuid::Uuid;

use crate::models::{SearchRequest, SearchResponse};
use crate::search::hybrid::{multi_query_rrf_fusion, QueryResults};
use crate::state::AppState;

/// Reusable search pipeline: query expansion → multi-query BM25+vector → RRF fusion → optional reranking.
/// Called by both the search handler and the chat handler.
pub async fn run_search(
    state: &AppState,
    query: &str,
    repo_ids: Option<&[Uuid]>,
    use_bm25: bool,
    use_vector: bool,
    use_rerank: bool,
    limit: usize,
) -> Result<SearchResponse, (StatusCode, String)> {
    let query = query.trim().to_string();
    if query.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Query is required".to_string()));
    }

    let limit = limit.min(200); // Cap at 200 to prevent resource exhaustion
    let fetch_limit = limit * 3; // Fetch more than needed for fusion

    // ── Step 1: Query Expansion ──────────────────────────────
    let mut queries: Vec<(String, f32)> = vec![(query.clone(), 2.0)]; // Original with ×2 weight

    if use_rerank {
        // Only expand if LLM features are enabled (rerank implies LLM availability)
        let llm_config = state.llm_config.read().clone();
        match crate::llm::query_expand::expand_query(&state.http_client, &llm_config, &query)
            .await
        {
            Ok(expanded) => {
                tracing::info!("Query expanded: {:?}", expanded);
                for eq in expanded {
                    if !eq.is_empty() {
                        queries.push((eq, 1.0));
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Query expansion failed: {e}");
            }
        }
    }

    // ── Step 2: Multi-query BM25 + Vector Search ─────────────
    let repo_ids_vec: Option<Vec<Uuid>> = repo_ids.map(|ids| ids.to_vec());
    let mut all_query_results = Vec::new();
    let mut total_bm25 = 0usize;
    let mut total_vector = 0usize;

    for (q, weight) in &queries {
        let bm25_hits = if use_bm25 {
            let bm25 = state.bm25.clone();
            let q = q.clone();
            let ids = repo_ids_vec.clone();
            let fl = fetch_limit;
            tokio::task::spawn_blocking(move || bm25.search(&q, fl, ids.as_deref()))
                .await
                .map_err(|e| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("BM25 search error: {e}"),
                    )
                })?
                .map_err(|e| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("BM25 search error: {e}"),
                    )
                })?
        } else {
            Vec::new()
        };

        let vector_hits = if use_vector {
            let llm_config = state.llm_config.read().clone();
            match crate::llm::embeddings::embed_single(&state.http_client, &llm_config, q, crate::llm::embeddings::EmbedTask::SearchQuery).await {
                Ok(query_embedding) => state
                    .vectors
                    .search(&query_embedding, fetch_limit, repo_ids_vec.as_deref()),
                Err(e) => {
                    tracing::warn!("Vector search skipped for variant '{q}': {e}");
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };

        total_bm25 += bm25_hits.len();
        total_vector += vector_hits.len();

        all_query_results.push(QueryResults {
            bm25_hits,
            vector_hits,
            weight: *weight,
        });
    }

    // ── Step 3: RRF Fusion (top 30) ─────────────────────────
    let rrf_limit = 30;
    let mut results = multi_query_rrf_fusion(&all_query_results, rrf_limit);

    // ── Step 4: LLM Re-ranking with position-aware blend ────
    if use_rerank && !results.is_empty() {
        let llm_config = state.llm_config.read().clone();
        match crate::llm::rerank::rerank(&state.http_client, &llm_config, &query, &mut results)
            .await
        {
            Ok(()) => {
                tracing::info!("Re-ranking applied to {} results", results.len());
            }
            Err(e) => {
                tracing::warn!("Re-ranking failed: {e}");
            }
        }
    }

    // Trim to requested limit
    results.truncate(limit);

    Ok(SearchResponse {
        query,
        results,
        total_bm25_hits: total_bm25,
        total_vector_hits: total_vector,
    })
}

/// POST /api/search - Full hybrid search pipeline
pub async fn search(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    let response = run_search(
        &state,
        &req.query,
        req.repo_ids.as_deref(),
        req.use_bm25,
        req.use_vector,
        req.use_rerank,
        req.limit,
    )
    .await?;
    Ok(Json(response))
}
