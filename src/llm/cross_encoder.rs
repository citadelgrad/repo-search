//! Cross-encoder reranker via OpenAI-compatible `/v1/rerank` endpoint.
//!
//! Sends a single batch request with all query-document pairs instead of
//! making N individual LLM chat calls. Typical latency: 50-100ms vs 1-3s.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::RerankerConfig;

/// Result of reranking a single document.
#[derive(Debug, Clone)]
pub struct RerankResult {
    /// Index into the original documents array.
    pub index: usize,
    /// Relevance score (0.0 - 1.0 after sigmoid normalization).
    pub score: f32,
}

/// Rerank documents against a query using a cross-encoder model.
///
/// Returns results sorted by score descending. Returns Err if the
/// reranker endpoint is unreachable or returns an error.
pub async fn rerank(
    client: &reqwest::Client,
    config: &RerankerConfig,
    query: &str,
    documents: &[String],
    top_n: usize,
) -> Result<Vec<RerankResult>> {
    let base_url = config
        .base_url
        .as_deref()
        .context("Reranker base_url not configured")?;

    let model = config
        .model
        .as_deref()
        .unwrap_or("default");

    let url = format!("{}/v1/rerank", base_url.trim_end_matches('/'));

    let req_body = RerankRequest {
        model: model.to_string(),
        query: query.to_string(),
        documents: documents.to_vec(),
        top_n,
    };

    let timeout = std::time::Duration::from_secs(config.timeout_secs.min(30));

    let resp = client
        .post(&url)
        .timeout(timeout)
        .json(&req_body)
        .send()
        .await
        .context("Failed to reach reranker endpoint")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Reranker returned {status}: {body}");
    }

    let body: RerankResponse = resp
        .json()
        .await
        .context("Failed to parse reranker response")?;

    let mut results: Vec<RerankResult> = body
        .results
        .into_iter()
        .map(|r| RerankResult {
            index: r.index,
            score: sigmoid(r.relevance_score),
        })
        .collect();

    // Sort by score descending
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    Ok(results)
}

/// Sigmoid normalization: maps raw logits to 0-1 range.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ─── Request/Response types ────────────────────────────

#[derive(Serialize)]
struct RerankRequest {
    model: String,
    query: String,
    documents: Vec<String>,
    top_n: usize,
}

#[derive(Deserialize)]
struct RerankResponse {
    results: Vec<RerankResultRaw>,
}

#[derive(Deserialize)]
struct RerankResultRaw {
    index: usize,
    relevance_score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_zero() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_large_positive() {
        let s = sigmoid(10.0);
        assert!(s > 0.999);
    }

    #[test]
    fn test_sigmoid_large_negative() {
        let s = sigmoid(-10.0);
        assert!(s < 0.001);
    }

    #[test]
    fn test_sigmoid_known_value() {
        // sigmoid(1) ≈ 0.7310586
        let s = sigmoid(1.0);
        assert!((s - 0.7310586).abs() < 1e-5);
    }

    #[test]
    fn test_sigmoid_symmetry() {
        // sigmoid(x) + sigmoid(-x) = 1
        let x = 2.5f32;
        let sum = sigmoid(x) + sigmoid(-x);
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
