use std::collections::HashMap;
use uuid::Uuid;

use crate::models::SearchHit;
use crate::search::bm25::Bm25Hit;
use crate::search::vector::VectorHit;

/// A set of BM25 + vector results from a single query variant.
pub struct QueryResults {
    pub bm25_hits: Vec<Bm25Hit>,
    pub vector_hits: Vec<VectorHit>,
    /// Weight multiplier for this query variant (original = 2.0, expanded = 1.0)
    pub weight: f32,
}

/// Multi-query RRF fusion with weighted query variants and top-rank bonus.
///
/// Pipeline:
/// 1. For each query variant (original ×2 weight, expanded ×1 each), compute RRF
///    scores from its BM25 and vector result lists.
/// 2. Sum weighted RRF scores across all variants.
/// 3. Apply a top-rank bonus of +0.05 for results that appeared in rank 1 of any list.
/// 4. Keep top 30 results.
pub fn multi_query_rrf_fusion(
    query_results: &[QueryResults],
    limit: usize,
) -> Vec<SearchHit> {
    let k = 60.0f32; // RRF constant
    let top_rank_bonus = 0.05f32;

    // Key: (repo_id, file_path, chunk_index)
    type Key = (Uuid, String, usize);
    let mut score_map: HashMap<Key, SearchHit> = HashMap::new();
    let mut top_ranked: HashMap<Key, bool> = HashMap::new();

    for qr in query_results {
        // Process BM25 results for this query variant
        for (rank, hit) in qr.bm25_hits.iter().enumerate() {
            let key: Key = (hit.repo_id, hit.file_path.clone(), hit.chunk_index);
            let rrf_score = qr.weight * (1.0 / (k + rank as f32 + 1.0));

            if rank == 0 {
                top_ranked.insert(key.clone(), true);
            }

            let entry = score_map.entry(key).or_insert_with(|| SearchHit {
                repo_id: hit.repo_id,
                repo_name: hit.repo_name.clone(),
                file_path: hit.file_path.clone(),
                chunk_index: hit.chunk_index,
                content: hit.content.clone(),
                language: hit.language.clone(),
                start_line: hit.start_line,
                end_line: hit.end_line,
                bm25_score: 0.0,
                vector_score: 0.0,
                combined_score: 0.0,
                rerank_score: None,
            });

            entry.bm25_score = entry.bm25_score.max(hit.score);
            entry.combined_score += rrf_score;
        }

        // Process vector results for this query variant
        for (rank, hit) in qr.vector_hits.iter().enumerate() {
            let key: Key = (hit.repo_id, hit.file_path.clone(), hit.chunk_index);
            let rrf_score = qr.weight * (1.0 / (k + rank as f32 + 1.0));

            if rank == 0 {
                top_ranked.insert(key.clone(), true);
            }

            let entry = score_map.entry(key).or_insert_with(|| SearchHit {
                repo_id: hit.repo_id,
                repo_name: hit.repo_name.clone(),
                file_path: hit.file_path.clone(),
                chunk_index: hit.chunk_index,
                content: hit.content.clone(),
                language: hit.language.clone(),
                start_line: hit.start_line,
                end_line: hit.end_line,
                bm25_score: 0.0,
                vector_score: 0.0,
                combined_score: 0.0,
                rerank_score: None,
            });

            entry.vector_score = entry.vector_score.max(hit.score);
            entry.combined_score += rrf_score;
        }
    }

    // Apply top-rank bonus
    for (key, hit) in score_map.iter_mut() {
        if top_ranked.contains_key(key) {
            hit.combined_score += top_rank_bonus;
        }
    }

    let mut results: Vec<SearchHit> = score_map.into_values().collect();
    results.sort_by(|a, b| {
        b.combined_score
            .partial_cmp(&a.combined_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(limit);
    results
}
