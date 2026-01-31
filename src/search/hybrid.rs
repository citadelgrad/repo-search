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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::bm25::Bm25Hit;
    use crate::search::vector::VectorHit;
    use uuid::Uuid;

    fn make_bm25_hit(id: Uuid, path: &str, chunk: usize, score: f32) -> Bm25Hit {
        Bm25Hit {
            repo_id: id,
            repo_name: "test-repo".to_string(),
            file_path: path.to_string(),
            chunk_index: chunk,
            content: format!("content of {path} chunk {chunk}"),
            language: "rust".to_string(),
            start_line: 1,
            end_line: 100,
            score,
        }
    }

    fn make_vector_hit(id: Uuid, path: &str, chunk: usize, score: f32) -> VectorHit {
        VectorHit {
            repo_id: id,
            repo_name: "test-repo".to_string(),
            file_path: path.to_string(),
            chunk_index: chunk,
            content: format!("content of {path} chunk {chunk}"),
            language: "rust".to_string(),
            start_line: 1,
            end_line: 100,
            score,
        }
    }

    #[test]
    fn test_empty_inputs() {
        let results = multi_query_rrf_fusion(&[], 10);
        assert!(results.is_empty());

        let qr = QueryResults {
            bm25_hits: vec![],
            vector_hits: vec![],
            weight: 1.0,
        };
        let results = multi_query_rrf_fusion(&[qr], 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_only_single_query() {
        let id = Uuid::new_v4();
        let qr = QueryResults {
            bm25_hits: vec![
                make_bm25_hit(id, "src/main.rs", 0, 5.0),
                make_bm25_hit(id, "src/lib.rs", 0, 3.0),
            ],
            vector_hits: vec![],
            weight: 2.0,
        };

        let results = multi_query_rrf_fusion(&[qr], 10);
        assert_eq!(results.len(), 2);
        // Rank 0 gets higher RRF score
        assert!(results[0].combined_score > results[1].combined_score);
        assert_eq!(results[0].file_path, "src/main.rs");
    }

    #[test]
    fn test_vector_only_single_query() {
        let id = Uuid::new_v4();
        let qr = QueryResults {
            bm25_hits: vec![],
            vector_hits: vec![
                make_vector_hit(id, "src/a.rs", 0, 0.95),
                make_vector_hit(id, "src/b.rs", 0, 0.80),
            ],
            weight: 1.0,
        };

        let results = multi_query_rrf_fusion(&[qr], 10);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].file_path, "src/a.rs");
        assert!(results[0].vector_score > results[1].vector_score);
    }

    #[test]
    fn test_combined_bm25_and_vector_boost() {
        let id = Uuid::new_v4();
        // File A: rank 0 in BM25, rank 1 in vector
        // File B: rank 1 in BM25, rank 0 in vector
        // File C: only in BM25 at rank 2
        let qr = QueryResults {
            bm25_hits: vec![
                make_bm25_hit(id, "a.rs", 0, 5.0),
                make_bm25_hit(id, "b.rs", 0, 3.0),
                make_bm25_hit(id, "c.rs", 0, 1.0),
            ],
            vector_hits: vec![
                make_vector_hit(id, "b.rs", 0, 0.95),
                make_vector_hit(id, "a.rs", 0, 0.80),
            ],
            weight: 1.0,
        };

        let results = multi_query_rrf_fusion(&[qr], 10);
        assert_eq!(results.len(), 3);
        // a.rs and b.rs should be top 2 (both appear in BM25 + vector)
        let top_paths: Vec<&str> = results.iter().take(2).map(|r| r.file_path.as_str()).collect();
        assert!(top_paths.contains(&"a.rs"));
        assert!(top_paths.contains(&"b.rs"));
        // c.rs should be last (only in BM25)
        assert_eq!(results[2].file_path, "c.rs");
    }

    #[test]
    fn test_original_query_weighted_higher() {
        let id = Uuid::new_v4();
        // Original query (weight 2.0) finds file A
        let original = QueryResults {
            bm25_hits: vec![make_bm25_hit(id, "a.rs", 0, 5.0)],
            vector_hits: vec![],
            weight: 2.0,
        };
        // Expanded query (weight 1.0) finds file B
        let expanded = QueryResults {
            bm25_hits: vec![make_bm25_hit(id, "b.rs", 0, 5.0)],
            vector_hits: vec![],
            weight: 1.0,
        };

        let results = multi_query_rrf_fusion(&[original, expanded], 10);
        assert_eq!(results.len(), 2);
        // a.rs should rank higher because of 2x weight
        assert_eq!(results[0].file_path, "a.rs");
        assert!(results[0].combined_score > results[1].combined_score);
    }

    #[test]
    fn test_top_rank_bonus_applied() {
        let id = Uuid::new_v4();
        let qr = QueryResults {
            bm25_hits: vec![
                make_bm25_hit(id, "first.rs", 0, 10.0),
                make_bm25_hit(id, "second.rs", 0, 9.0),
            ],
            vector_hits: vec![],
            weight: 1.0,
        };

        let results = multi_query_rrf_fusion(&[qr], 10);
        // first.rs is rank 0 so gets top_rank_bonus (+0.05)
        let first = &results[0];
        let k = 60.0f32;
        let expected_rrf = 1.0 / (k + 1.0); // rank 0
        let expected_with_bonus = expected_rrf + 0.05;
        assert!((first.combined_score - expected_with_bonus).abs() < 0.001);
    }

    #[test]
    fn test_limit_respected() {
        let id = Uuid::new_v4();
        let bm25_hits: Vec<_> = (0..50)
            .map(|i| make_bm25_hit(id, &format!("file_{i}.rs"), 0, 50.0 - i as f32))
            .collect();

        let qr = QueryResults {
            bm25_hits,
            vector_hits: vec![],
            weight: 1.0,
        };

        let results = multi_query_rrf_fusion(&[qr], 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_multi_query_fusion_deduplication() {
        let id = Uuid::new_v4();
        // Same file appears in multiple query variants
        let q1 = QueryResults {
            bm25_hits: vec![make_bm25_hit(id, "common.rs", 0, 5.0)],
            vector_hits: vec![],
            weight: 2.0,
        };
        let q2 = QueryResults {
            bm25_hits: vec![make_bm25_hit(id, "common.rs", 0, 4.0)],
            vector_hits: vec![],
            weight: 1.0,
        };

        let results = multi_query_rrf_fusion(&[q1, q2], 10);
        // Should be deduplicated to 1 entry with accumulated score
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_path, "common.rs");
        // Score should be sum of both queries' RRF + top_rank_bonus
        let k = 60.0f32;
        let expected = 2.0 * (1.0 / (k + 1.0)) + 1.0 * (1.0 / (k + 1.0)) + 0.05;
        assert!((results[0].combined_score - expected).abs() < 0.001);
    }

    #[test]
    fn test_bm25_score_keeps_max() {
        let id = Uuid::new_v4();
        let q1 = QueryResults {
            bm25_hits: vec![make_bm25_hit(id, "a.rs", 0, 8.0)],
            vector_hits: vec![],
            weight: 1.0,
        };
        let q2 = QueryResults {
            bm25_hits: vec![make_bm25_hit(id, "a.rs", 0, 3.0)],
            vector_hits: vec![],
            weight: 1.0,
        };

        let results = multi_query_rrf_fusion(&[q1, q2], 10);
        assert_eq!(results[0].bm25_score, 8.0); // max, not overwritten
    }
}
